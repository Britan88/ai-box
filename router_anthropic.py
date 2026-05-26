# router_anthropic.py
import json
import logging
import sqlite3
import time
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import api_aiio
import api_cf

logger = logging.getLogger("router_anthropic")

router = APIRouter()

CF_API = "https://api.cloudflare.com/client/v4"
AIIO_COMPLETIONS_URL = "https://api.intelligence.io.solutions/api/v1/chat/completions"

ACTIVE_PROVIDERS: list[str] = ["cf"]


class ContentBlock(BaseModel):
    type: str
    text: str


class AnthropicMessage(BaseModel):
    role: str
    content: str | list


class AnthropicRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 1000
    temperature: float = 1.0
    stream: bool = False
    system: str | list | None = None


def get_db_path() -> str:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return os.getenv("DB_NAME")


def resolve_model(db_path: str, model: str) -> tuple[str, str]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in ACTIVE_PROVIDERS)
    cursor.execute(
        f"SELECT provider, model FROM models WHERE model = ? AND status = 'ok' AND provider IN ({placeholders}) LIMIT 1",
        [model, *ACTIVE_PROVIDERS],
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        logger.error(
            f"[resolve_model] not found | model={model} | providers={ACTIVE_PROVIDERS}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found or not available in providers {ACTIVE_PROVIDERS}",
        )

    logger.debug(
        f"[resolve_model] resolved | model={model} | provider={row['provider']}"
    )
    return row["provider"], row["model"]


def _anthropic_to_openai_messages(
    messages: list[AnthropicMessage], system: str | None
) -> list[dict]:
    result = []
    if system:
        result.append({"role": "system", "content": system})
    for m in messages:
        if isinstance(m.content, str):
            result.append({"role": m.role, "content": m.content})
        elif isinstance(m.content, list):
            text = " ".join(
                block.get("text", "")
                for block in m.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            result.append({"role": m.role, "content": text})
    return result


def _wrap_anthropic_response(model: str, content: str) -> dict:
    return {
        "id": f"msg_{int(time.time())}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _anthropic_to_openai_messages(
    messages: list[AnthropicMessage], system: str | list | None
) -> list[dict]:
    result = []
    if system:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_text = " ".join(
                block.get("text", "")
                for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            system_text = str(system)
        if system_text:
            result.append({"role": "system", "content": system_text})
    for m in messages:
        if isinstance(m.content, str):
            result.append({"role": m.role, "content": m.content})
        elif isinstance(m.content, list):
            text = " ".join(
                block.get("text", "")
                for block in m.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            result.append({"role": m.role, "content": text})
    return result


async def _stream_anthropic_cf(
    key: dict, model: str, payload: dict, original_model: str
) -> AsyncGenerator[str, None]:
    msg_id = f"msg_{int(time.time())}"

    # yield immediately before any network call
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': original_model, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    headers = {
        "Authorization": f"Bearer {key['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{CF_API}/accounts/{key['account_id']}/ai/run/{model}"

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            logger.debug(f"[_stream_anthropic_cf] HTTP {r.status_code} | model={model}")
            if r.status_code >= 400:
                body = await r.aread()
                raise HTTPException(status_code=r.status_code, detail=body.decode())
            async for line in r.aiter_lines():
                logger.debug(f"[_stream_anthropic_cf] raw line | {line!r}")
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta_text = chunk.get("response", "") or ""
                    if not delta_text:
                        choices = chunk.get("choices", [])
                        if choices:
                            delta_text = (
                                choices[0].get("delta", {}).get("content", "") or ""
                            )
                    if delta_text:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_text}})}\n\n"
                except Exception as e:
                    logger.debug(
                        f"[_stream_anthropic_cf] parse error | {e} | line={line!r}"
                    )

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


async def _stream_anthropic_aiio(
    api_key: str, model: str, payload: dict, original_model: str
) -> AsyncGenerator[str, None]:
    msg_id = f"msg_{int(time.time())}"

    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': original_model, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=90) as client:
        async with client.stream(
            "POST", AIIO_COMPLETIONS_URL, headers=headers, json=payload
        ) as r:
            logger.debug(
                f"[_stream_anthropic_aiio] HTTP {r.status_code} | model={model}"
            )
            if r.status_code >= 400:
                body = await r.aread()
                raise HTTPException(status_code=r.status_code, detail=body.decode())
            async for line in r.aiter_lines():
                logger.debug(f"[_stream_anthropic_aiio] raw line | {line!r}")
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta_text = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                        or ""
                    )
                    if delta_text:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_text}})}\n\n"
                except Exception as e:
                    logger.debug(
                        f"[_stream_anthropic_aiio] parse error | {e} | line={line!r}"
                    )

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


@router.post("/v1/messages")
async def messages(req: AnthropicRequest):
    db_path = get_db_path()
    provider, model = resolve_model(db_path, req.model)

    logger.info(
        f"[messages] provider={provider} | model={model} | stream={req.stream} | messages={len(req.messages)}"
    )

    openai_messages = _anthropic_to_openai_messages(req.messages, req.system)
    payload = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }

    if req.stream:
        if provider == "cf":
            key = api_cf._get_random_key(db_path)
            return StreamingResponse(
                _stream_anthropic_cf(key, model, payload, req.model),
                media_type="text/event-stream",
            )
        if provider == "aiio":
            api_key = api_aiio._get_random_key(db_path)
            return StreamingResponse(
                _stream_anthropic_aiio(api_key, model, payload, req.model),
                media_type="text/event-stream",
            )

    if provider == "cf":
        key = api_cf._get_random_key(db_path)
        headers = {
            "Authorization": f"Bearer {key['api_key']}",
            "Content-Type": "application/json",
        }
        url = f"{CF_API}/accounts/{key['account_id']}/ai/run/{model}"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
        logger.debug(f"[messages] cf HTTP {r.status_code}")
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        logger.debug(f"[messages] cf raw response | {data}")

        if not data.get("success"):
            raise HTTPException(
                status_code=502, detail=f"CF error: {data.get('errors')}"
            )
        content = data.get("result", {}).get("response", "")
        return _wrap_anthropic_response(req.model, content)

    if provider == "aiio":
        api_key = api_aiio._get_random_key(db_path)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(AIIO_COMPLETIONS_URL, headers=headers, json=payload)
        logger.debug(f"[messages] aiio HTTP {r.status_code}")
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return _wrap_anthropic_response(req.model, content)

    raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")
