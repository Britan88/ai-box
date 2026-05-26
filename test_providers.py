import asyncio
import base64
import json
import logging
import os
import sqlite3
import struct
import time
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from api_interface import call, get_models

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_providers")

DB_PATH = os.getenv("DB_NAME")
PROVIDERS = ("cf", "aiio")
TEST_WAV_PATH = "hello_world.wav"
TEST_PNG_PATH = "test.png"
CF_API = "https://api.cloudflare.com/client/v4"
TEXT_PROMPT_PATH = "prompts/text_prompt.md"
IMAGE_PROMPT_PATH = "prompts/img_prompt.md"


IMAGE_MODEL_SUBSTRINGS = (
    "flux",
    "stable-diffusion",
    "dreamshaper",
    "leonardo",
)
IMAGE_INPAINTING_SUBSTRINGS = ("inpainting",)
EMBEDDING_MODEL_SUBSTRINGS = (
    "bge-",
    "bge_",
    "embedding",
)
STT_MODEL_SUBSTRINGS = (
    "whisper",
    "nova-",
    "nova3",
    "deepgram/nova",
)
STT_DEEPGRAM_FLUX_SUBSTRINGS = ("deepgram/flux",)
TTS_MODEL_SUBSTRINGS = (
    "melotts",
    "aura-",
)
VISION_MODEL_SUBSTRINGS = (
    "llava",
    "uform",
)
CLASSIFICATION_TEXT_SUBSTRINGS = ("distilbert",)
CLASSIFICATION_IMAGE_SUBSTRINGS = ("resnet",)
TRANSLATION_SUBSTRINGS = (
    "m2m100",
    "indictrans2",
)
SKIP_SUBSTRINGS = ("smart-turn",)
FLUX_MULTIPART_SUBSTRINGS = ("flux-2-klein",)
TTS_PROMPT_SUBSTRINGS = ("melotts",)

TASK_TYPE_MAP = {
    "Text Generation": "text",
    "text-generation": "text",
    "Translation": "translation",
    "Summarization": "text",
    "Text Classification": "classification_text",
    "Image Classification": "classification_image",
    "Image Generation": "image",
    "image-generation": "image",
    "Text Embeddings": "embedding",
    "Feature Extraction": "embedding",
    "Speech Recognition": "stt",
    "Automatic Speech Recognition": "stt",
    "Text-to-Speech": "tts",
}


def _load_text_prompt(path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        return [{"role": "user", "content": content}]
    except FileNotFoundError:
        return [{"role": "user", "content": "Say hello in one sentence."}]


TEXT_PROMPT = _load_text_prompt(TEXT_PROMPT_PATH)


def _load_image_prompt(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "a beautiful mountain landscape at sunset"


IMAGE_PROMPT = _load_image_prompt(IMAGE_PROMPT_PATH)


def detect_type_cf(model: dict) -> str:
    model_id = (model.get("id") or "").lower()
    task = model.get("task") or ""

    for s in SKIP_SUBSTRINGS:
        if s in model_id:
            return "skip"
    for s in IMAGE_INPAINTING_SUBSTRINGS:
        if s in model_id:
            return "image_inpainting"
    for s in FLUX_MULTIPART_SUBSTRINGS:
        if s in model_id:
            return "image_multipart"
    for s in STT_DEEPGRAM_FLUX_SUBSTRINGS:
        if s in model_id:
            return "stt_deepgram_flux"
    for s in STT_MODEL_SUBSTRINGS:
        if s in model_id:
            return "stt"
    for s in TTS_PROMPT_SUBSTRINGS:
        if s in model_id:
            return "tts_prompt"
    for s in TTS_MODEL_SUBSTRINGS:
        if s in model_id:
            return "tts"
    for s in VISION_MODEL_SUBSTRINGS:
        if s in model_id:
            return "vision"
    for s in CLASSIFICATION_TEXT_SUBSTRINGS:
        if s in model_id:
            return "classification_text"
    for s in CLASSIFICATION_IMAGE_SUBSTRINGS:
        if s in model_id:
            return "classification_image"
    for s in TRANSLATION_SUBSTRINGS:
        if s in model_id:
            return "translation"
    for s in EMBEDDING_MODEL_SUBSTRINGS:
        if s in model_id:
            return "embedding"
    for s in IMAGE_MODEL_SUBSTRINGS:
        if s in model_id:
            return "image"

    return TASK_TYPE_MAP.get(task, "text")


def detect_type_aiio(_model: dict) -> str:
    return "text"


def load_file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def read_wav_sample_rate(wav_bytes: bytes) -> int:
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    logger.debug(f"[read_wav_sample_rate] sample_rate={sample_rate}")
    return sample_rate


def read_wav_pcm(wav_bytes: bytes) -> bytes:
    # skip WAV header (44 bytes standard) and return raw PCM
    return wav_bytes[44:]


def format_error(payload: dict | None, response: str) -> str:
    payload_str = json.dumps(payload, ensure_ascii=False) if payload else "n/a"
    return f"{payload_str} | {response}"


def write_result(
    db_path: str,
    provider: str,
    model_type: str,
    model_id: str,
    status: str,
    avtime: int,
    error: str | None,
    result: str | None = None,
):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO models (provider, type, model, status, avtime, error, result) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (provider, model_type, model_id, status, avtime, error, result),
    )
    conn.commit()
    conn.close()


async def _cf_post_json(
    key: dict, model_id: str, payload: dict
) -> tuple[int, dict | bytes]:
    headers = {
        "Authorization": f"Bearer {key['api_key']}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{CF_API}/accounts/{key['account_id']}/ai/run/{model_id}",
            headers=headers,
            json=payload,
        )
    content_type = r.headers.get("content-type", "")
    logger.debug(
        f"[_cf_post_json] model={model_id} | HTTP {r.status_code} | content-type={content_type}"
    )
    if (
        "image/" in content_type
        or "audio/" in content_type
        or "application/octet-stream" in content_type
    ):
        return r.status_code, r.content
    return r.status_code, r.json()


async def _cf_post_multipart(
    key: dict, model_id: str, data: dict, files: dict | None = None
) -> tuple[int, dict | bytes]:
    headers = {"Authorization": f"Bearer {key['api_key']}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{CF_API}/accounts/{key['account_id']}/ai/run/{model_id}",
            headers=headers,
            data=data,
            files=files,
        )
    content_type = r.headers.get("content-type", "")
    logger.debug(
        f"[_cf_post_multipart] model={model_id} | HTTP {r.status_code} | content-type={content_type}"
    )
    if (
        "image/" in content_type
        or "audio/" in content_type
        or "application/octet-stream" in content_type
    ):
        return r.status_code, r.content
    return r.status_code, r.json()


def _assert_cf_success(status_code: int, result: dict | bytes):
    if isinstance(result, bytes):
        if status_code >= 400:
            raise RuntimeError(f"HTTP {status_code} | binary error len={len(result)}")
        return
    if status_code >= 400 or not result.get("success"):
        raise RuntimeError(str(result))


async def test_text(provider, model_id):
    t0 = time.perf_counter()
    log_payload = {"messages": TEXT_PROMPT, "max_tokens": 100}
    try:
        response = await call(provider, DB_PATH, model_id, TEXT_PROMPT, max_tokens=100)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        result = response if isinstance(response, str) else None
        return "ok", elapsed_ms, None, result
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return "error", elapsed_ms, format_error(log_payload, str(e)), None


async def test_image(provider: str, model_id: str) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    payload = {"prompt": IMAGE_PROMPT}
    try:
        key = api_cf._get_random_key(DB_PATH)
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        result_b64 = (
            base64.b64encode(result).decode() if isinstance(result, bytes) else None
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(f"[test_image] OK | model={model_id} | elapsed={elapsed_ms}ms")
        return "ok", elapsed_ms, None, result_b64
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(payload, str(e))
        logger.error(f"[test_image] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


async def test_image_multipart(
    provider: str, model_id: str
) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    log_payload = {"prompt": IMAGE_PROMPT, "multipart": True}
    try:
        key = api_cf._get_random_key(DB_PATH)
        status_code, result = await _cf_post_multipart(
            key,
            model_id,
            data={"prompt": IMAGE_PROMPT},
        )
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            f"[test_image_multipart] OK | model={model_id} | elapsed={elapsed_ms}ms"
        )
        # после _assert_cf_success
        result_b64 = (
            base64.b64encode(result).decode() if isinstance(result, bytes) else None
        )
        return "ok", elapsed_ms, None, result_b64
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(log_payload, str(e))
        logger.error(f"[test_image_multipart] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


async def test_embedding(provider: str, model_id: str) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    payload = {"text": "hello world test embedding"}
    try:
        key = api_cf._get_random_key(DB_PATH)
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(f"[test_embedding] OK | model={model_id} | elapsed={elapsed_ms}ms")
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(payload, str(e))
        logger.error(f"[test_embedding] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None, None


async def test_stt(provider: str, model_id: str) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    log_payload = {"audio": "<wav_bytes_list>"}
    try:
        key = api_cf._get_random_key(DB_PATH)
        wav_bytes = load_file_bytes(TEST_WAV_PATH)
        payload = {"audio": list(wav_bytes)}
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        transcript = result.get("text", "") if isinstance(result, dict) else ""
        logger.info(
            f"[test_stt] OK | model={model_id} | transcript={transcript!r} | elapsed={elapsed_ms}ms"
        )
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(log_payload, str(e))
        logger.error(f"[test_stt] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None, None


async def test_image_inpainting(
    provider: str, model_id: str
) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    log_payload = {
        "prompt": IMAGE_PROMPT,
        "image": f"<b64 from {TEST_PNG_PATH}>",
        "mask_image": f"<b64 from {TEST_PNG_PATH}>",
    }
    try:
        key = api_cf._get_random_key(DB_PATH)
        png_b64 = load_file_b64(TEST_PNG_PATH)
        payload = {"prompt": IMAGE_PROMPT, "image": png_b64, "mask_image": png_b64}
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            f"[test_image_inpainting] OK | model={model_id} | elapsed={elapsed_ms}ms"
        )
        # после _assert_cf_success
        result_b64 = (
            base64.b64encode(result).decode() if isinstance(result, bytes) else None
        )
        return "ok", elapsed_ms, None, result_b64
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(log_payload, str(e))
        logger.error(f"[test_image_inpainting] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


async def test_stt_nova3(provider: str, model_id: str) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    log_payload = {"audio": "<pcm16_b64>"}
    try:
        key = api_cf._get_random_key(DB_PATH)
        wav_bytes = load_file_bytes(TEST_WAV_PATH)
        pcm16_b64 = convert_wav_to_pcm16_b64(wav_bytes)
        payload = {"audio": pcm16_b64}
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        transcript = result.get("text", "") if isinstance(result, dict) else ""
        logger.info(
            f"[test_stt_nova3] OK | model={model_id} | transcript={transcript!r} | elapsed={elapsed_ms}ms"
        )
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(log_payload, str(e))
        logger.error(f"[test_stt_nova3] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


async def test_stt_deepgram_flux(
    provider: str, model_id: str
) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    log_payload = {"audio": "<pcm16_b64>", "sample_rate": "?", "encoding": "linear16"}
    try:
        key = api_cf._get_random_key(DB_PATH)
        wav_bytes = load_file_bytes(TEST_WAV_PATH)
        sample_rate = int(read_wav_sample_rate(wav_bytes))
        pcm16_b64 = convert_wav_to_pcm16_b64(wav_bytes)
        log_payload = {
            "audio": "<pcm16_b64>",
            "sample_rate": sample_rate,
            "encoding": "linear16",
        }
        payload = {
            "audio": pcm16_b64,
            "sample_rate": sample_rate,
            "encoding": "linear16",
        }
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            f"[test_stt_deepgram_flux] OK | model={model_id} | elapsed={elapsed_ms}ms"
        )
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(log_payload, str(e))
        logger.error(f"[test_stt_deepgram_flux] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


def convert_wav_to_pcm16_b64(wav_bytes: bytes) -> str:
    # read WAV header fields
    audio_format = struct.unpack_from("<H", wav_bytes, 20)[0]
    num_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]

    logger.debug(
        f"[convert_wav_to_pcm16_b64] format={audio_format} channels={num_channels} bits={bits_per_sample}"
    )

    # find data chunk (skip past fmt chunk)
    offset = 12
    pcm_data = b""
    while offset < len(wav_bytes) - 8:
        chunk_id = wav_bytes[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
        if chunk_id == b"data":
            pcm_data = wav_bytes[offset + 8 : offset + 8 + chunk_size]
            break
        offset += 8 + chunk_size

    # format 3 = IEEE float32, convert to int16
    if audio_format == 3 and bits_per_sample == 32:
        num_samples = len(pcm_data) // 4
        floats = struct.unpack_from(f"<{num_samples}f", pcm_data)
        # clamp and convert float32 [-1.0, 1.0] -> int16
        ints = [max(-32768, min(32767, int(f * 32767))) for f in floats]
        pcm_data = struct.pack(f"<{num_samples}h", *ints)
        logger.debug(
            f"[convert_wav_to_pcm16_b64] converted float32 -> pcm16 | samples={num_samples}"
        )
    elif audio_format == 1 and bits_per_sample == 16:
        logger.debug(
            f"[convert_wav_to_pcm16_b64] already pcm16 | bytes={len(pcm_data)}"
        )
    else:
        logger.warning(
            f"[convert_wav_to_pcm16_b64] unknown format={audio_format} bits={bits_per_sample} — passing as-is"
        )

    return base64.b64encode(pcm_data).decode()


async def test_tts(provider: str, model_id: str) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    payload = {"text": "hello world"}
    try:
        key = api_cf._get_random_key(DB_PATH)
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(f"[test_tts] OK | model={model_id} | elapsed={elapsed_ms}ms")
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(payload, str(e))
        logger.error(f"[test_tts] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


async def test_tts_prompt(provider: str, model_id: str) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    payload = {"prompt": "hello world"}
    try:
        key = api_cf._get_random_key(DB_PATH)
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(f"[test_tts_prompt] OK | model={model_id} | elapsed={elapsed_ms}ms")
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(payload, str(e))
        logger.error(f"[test_tts_prompt] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


async def test_vision(provider: str, model_id: str) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    log_payload = {
        "image": f"<bytes from {TEST_PNG_PATH}>",
        "prompt": "describe this image",
        "max_tokens": 100,
    }
    try:
        key = api_cf._get_random_key(DB_PATH)
        png_bytes = load_file_bytes(TEST_PNG_PATH)
        payload = {
            "image": list(png_bytes),
            "prompt": "describe this image",
            "max_tokens": 100,
        }
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(f"[test_vision] OK | model={model_id} | elapsed={elapsed_ms}ms")
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(log_payload, str(e))
        logger.error(f"[test_vision] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


async def test_classification_text(
    provider: str, model_id: str
) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    payload = {"text": "hello world"}
    try:
        key = api_cf._get_random_key(DB_PATH)
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            f"[test_classification_text] OK | model={model_id} | elapsed={elapsed_ms}ms"
        )
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(payload, str(e))
        logger.error(f"[test_classification_text] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


async def test_classification_image(
    provider: str, model_id: str
) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    log_payload = {"image": f"<bytes from {TEST_PNG_PATH}>"}
    try:
        key = api_cf._get_random_key(DB_PATH)
        png_bytes = load_file_bytes(TEST_PNG_PATH)
        payload = {"image": list(png_bytes)}
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            f"[test_classification_image] OK | model={model_id} | elapsed={elapsed_ms}ms"
        )
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(log_payload, str(e))
        logger.error(f"[test_classification_image] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


async def test_translation(provider: str, model_id: str) -> tuple[str, int, str | None]:
    import api_cf

    t0 = time.perf_counter()
    payload = {"text": "hello world", "source_lang": "en", "target_lang": "fr"}
    try:
        key = api_cf._get_random_key(DB_PATH)
        status_code, result = await _cf_post_json(key, model_id, payload)
        _assert_cf_success(status_code, result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            f"[test_translation] OK | model={model_id} | elapsed={elapsed_ms}ms"
        )
        return "ok", elapsed_ms, None, None
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        error = format_error(payload, str(e))
        logger.error(f"[test_translation] ERROR | model={model_id} | {error}")
        return "error", elapsed_ms, error, None


def _is_nova3(model_id: str) -> bool:
    return "nova-3" in model_id.lower()


TYPE_HANDLERS = {
    "text": test_text,
    "image": test_image,
    "image_multipart": test_image_multipart,
    "image_inpainting": test_image_inpainting,
    "embedding": test_embedding,
    "stt": test_stt,
    "stt_nova3": test_stt_nova3,
    "stt_deepgram_flux": test_stt_deepgram_flux,
    "tts": test_tts,
    "tts_prompt": test_tts_prompt,
    "vision": test_vision,
    "classification_text": test_classification_text,
    "classification_image": test_classification_image,
    "translation": test_translation,
}


async def test_model(provider: str, model: dict) -> dict:
    model_id = model["id"]
    model_type = detect_type_cf(model) if provider == "cf" else detect_type_aiio(model)

    if model_type == "stt" and _is_nova3(model_id):
        model_type = "stt_nova3"

    logger.info(
        f"[test_model] provider={provider} | model={model_id} | type={model_type}"
    )

    if model_type == "skip":
        logger.info(f"[test_model] SKIP | model={model_id}")
        write_result(DB_PATH, provider, model_type, model_id, "skip", 0, None)
        return {
            "model": model_id,
            "type": model_type,
            "status": "skip",
            "avtime": 0,
            "error": None,
        }

    handler = TYPE_HANDLERS.get(model_type, test_text)
    # надо
    status, avtime, error, result = await handler(provider, model_id)
    write_result(DB_PATH, provider, model_type, model_id, status, avtime, error, result)
    return {
        "model": model_id,
        "type": model_type,
        "status": status,
        "avtime": avtime,
        "error": error,
    }


async def test_provider(provider: str) -> dict:
    logger.info(f"[test_provider] START | provider={provider}")
    t0 = time.perf_counter()

    try:
        models = await get_models(provider, DB_PATH)
        logger.info(
            f"[test_provider] models fetched | provider={provider} | count={len(models)}"
        )
    except Exception as e:
        logger.error(
            f"[test_provider] failed to get models | provider={provider} | {e}"
        )
        return {
            "provider": provider,
            "status": "error",
            "error": str(e),
            "models": [],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }

    results = []
    for model in models:
        result = await test_model(provider, model)
        results.append(result)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info(
        f"[test_provider] DONE | provider={provider} | ok={ok}/{len(results)} | elapsed={elapsed_ms}ms"
    )

    return {
        "provider": provider,
        "total": len(results),
        "ok": ok,
        "errors": len(results) - ok,
        "elapsed_ms": elapsed_ms,
        "models": results,
    }


async def run():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = f"test_results_{timestamp}.json"

    logger.info(f"[run] START | providers={PROVIDERS} | output={output_path}")

    report = {"timestamp": timestamp, "db": DB_PATH, "providers": []}

    for provider in PROVIDERS:
        result = await test_provider(provider)
        report["providers"].append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"[run] DONE | output={output_path}")


if __name__ == "__main__":
    asyncio.run(run())
