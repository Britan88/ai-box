import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime, timezone

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

TEXT_PROMPT = [{"role": "user", "content": "Say hello in one sentence."}]

TASK_TYPE_MAP = {
    "Text Generation": "text",
    "text-generation": "text",
    "Translation": "text",
    "Summarization": "text",
    "Text Classification": "text",
    "Image Generation": "image",
    "image-generation": "image",
    "Text Embeddings": "embedding",
    "Feature Extraction": "embedding",
    "Speech Recognition": "stt",
    "Automatic Speech Recognition": "stt",
}


def detect_type_cf(model: dict) -> str:
    task = model.get("task") or ""
    return TASK_TYPE_MAP.get(task, "text")


def detect_type_aiio(_model: dict) -> str:
    return "text"


def load_wav_b64() -> str:
    with open(TEST_WAV_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


async def test_text(provider: str, model_id: str) -> dict:
    logger.info(f"[test_text] provider={provider} | model={model_id}")
    t0 = time.perf_counter()
    try:
        response = await call(provider, DB_PATH, model_id, TEXT_PROMPT, max_tokens=100)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(f"[test_text] OK | model={model_id} | elapsed={elapsed_ms}ms")
        return {
            "status": "ok",
            "request": {"messages": TEXT_PROMPT, "max_tokens": 100},
            "response": response,
            "elapsed_ms": elapsed_ms,
        }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.error(f"[test_text] ERROR | model={model_id} | {e}")
        return {
            "status": "error",
            "request": {"messages": TEXT_PROMPT, "max_tokens": 100},
            "response": str(e),
            "elapsed_ms": elapsed_ms,
        }


async def test_image(provider: str, model_id: str) -> dict:
    import api_cf

    logger.info(f"[test_image] provider={provider} | model={model_id}")
    t0 = time.perf_counter()
    try:
        key = api_cf._get_random_key(DB_PATH)
        import httpx

        CF_API = "https://api.cloudflare.com/client/v4"
        headers = {
            "Authorization": f"Bearer {key['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {"prompt": "a red apple on a white background"}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{CF_API}/accounts/{key['account_id']}/ai/run/{model_id}",
                headers=headers,
                json=payload,
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.debug(f"[test_image] HTTP {r.status_code} | model={model_id}")
        if not r.is_success:
            raise RuntimeError(f"HTTP {r.status_code} | {r.text}")
        content_type = r.headers.get("content-type", "")
        if "image/" in content_type:
            image_b64 = base64.b64encode(r.content).decode()
        else:
            data = r.json()
            if not data.get("success"):
                raise RuntimeError(f"api error: {data.get('errors')}")
            result = data.get("result", {})
            image_b64 = (
                result.get("image") or result.get("b64_json") or result.get("output")
            )
        logger.info(f"[test_image] OK | model={model_id} | elapsed={elapsed_ms}ms")
        return {
            "status": "ok",
            "request": {"prompt": "a red apple on a white background"},
            "response": f"image_b64_len={len(image_b64)}",
            "elapsed_ms": elapsed_ms,
        }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.error(f"[test_image] ERROR | model={model_id} | {e}")
        return {
            "status": "error",
            "request": {"prompt": "a red apple on a white background"},
            "response": str(e),
            "elapsed_ms": elapsed_ms,
        }


async def test_embedding(provider: str, model_id: str) -> dict:
    import httpx

    import api_cf

    logger.info(f"[test_embedding] provider={provider} | model={model_id}")
    t0 = time.perf_counter()
    try:
        key = api_cf._get_random_key(DB_PATH)
        CF_API = "https://api.cloudflare.com/client/v4"
        headers = {
            "Authorization": f"Bearer {key['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {"text": "hello world test embedding"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{CF_API}/accounts/{key['account_id']}/ai/run/{model_id}",
                headers=headers,
                json=payload,
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.debug(f"[test_embedding] HTTP {r.status_code} | model={model_id}")
        if not r.is_success:
            raise RuntimeError(f"HTTP {r.status_code} | {r.text}")
        data = r.json()
        if not data.get("success"):
            raise RuntimeError(f"api error: {data.get('errors')}")
        result = data.get("result", {})
        shape = f"vectors={len(result.get('data', []))}"
        logger.info(
            f"[test_embedding] OK | model={model_id} | {shape} | elapsed={elapsed_ms}ms"
        )
        return {
            "status": "ok",
            "request": {"text": "hello world test embedding"},
            "response": shape,
            "elapsed_ms": elapsed_ms,
        }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.error(f"[test_embedding] ERROR | model={model_id} | {e}")
        return {
            "status": "error",
            "request": {"text": "hello world test embedding"},
            "response": str(e),
            "elapsed_ms": elapsed_ms,
        }


async def test_stt(provider: str, model_id: str) -> dict:
    import httpx

    import api_cf

    logger.info(f"[test_stt] provider={provider} | model={model_id}")
    t0 = time.perf_counter()
    try:
        key = api_cf._get_random_key(DB_PATH)
        CF_API = "https://api.cloudflare.com/client/v4"
        headers = {"Authorization": f"Bearer {key['api_key']}"}
        wav_b64 = load_wav_b64()
        payload = {"audio": wav_b64}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{CF_API}/accounts/{key['account_id']}/ai/run/{model_id}",
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.debug(f"[test_stt] HTTP {r.status_code} | model={model_id}")
        if not r.is_success:
            raise RuntimeError(f"HTTP {r.status_code} | {r.text}")
        data = r.json()
        if not data.get("success"):
            raise RuntimeError(f"api error: {data.get('errors')}")
        transcript = data.get("result", {}).get("text", "")
        logger.info(
            f"[test_stt] OK | model={model_id} | transcript={transcript!r} | elapsed={elapsed_ms}ms"
        )
        return {
            "status": "ok",
            "request": {"audio": "hello_world.wav"},
            "response": transcript,
            "elapsed_ms": elapsed_ms,
        }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.error(f"[test_stt] ERROR | model={model_id} | {e}")
        return {
            "status": "error",
            "request": {"audio": "hello_world.wav"},
            "response": str(e),
            "elapsed_ms": elapsed_ms,
        }


async def test_model(provider: str, model: dict) -> dict:
    model_id = model["id"]
    model_type = detect_type_cf(model) if provider == "cf" else detect_type_aiio(model)

    logger.info(
        f"[test_model] provider={provider} | model={model_id} | type={model_type}"
    )

    if model_type == "text":
        result = await test_text(provider, model_id)
    elif model_type == "image":
        result = await test_image(provider, model_id)
    elif model_type == "embedding":
        result = await test_embedding(provider, model_id)
    elif model_type == "stt":
        result = await test_stt(provider, model_id)
    else:
        result = await test_text(provider, model_id)

    return {
        "model": model_id,
        "type": model_type,
        **result,
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
        "status": "ok",
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

    report = {
        "timestamp": timestamp,
        "db": DB_PATH,
        "providers": [],
    }

    for provider in PROVIDERS:
        result = await test_provider(provider)
        report["providers"].append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"[run] DONE | output={output_path}")


if __name__ == "__main__":
    asyncio.run(run())
