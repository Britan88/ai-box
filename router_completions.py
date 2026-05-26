# router_completions.py
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

logger = logging.getLogger("router_completions")

router = APIRouter()

ACTIVE_PROVIDERS: list[str] = ["cf"]

CF_API = "https://api.cloudflare.com/client/v4"
AIIO_COMPLETIONS_URL = "https://api.intelligence.io.solutions/api/v1/chat/completions"


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 1000
    temperature: float = 1.0
    stream: bool = False


def get_db_path() -> str:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return os.getenv("DB_NAME")


def resolve_model(db_path: str, model: str) -> tuple[str, str]:
    """Returns (provider, model). Raises if not found."""
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
            f"[resolve_model] model not found or not ok | model={model} | providers={ACTIVE_PROVIDERS}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found or not available in providers {ACTIVE_PROVIDERS}",
        )

    logger.debug(
        f"[resolve_model] resolved | model={model} | provider={row['provider']}"
    )
    return row["provider"], row["model"]


async def _stream_cf(key: dict, model: str, payload: dict) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {key['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{CF_API}/accounts/{key['account_id']}/ai/run/{model}"

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            logger.debug(f"[_stream_cf] HTTP {r.status_code} | model={model}")
            if r.status_code >= 400:
                body = await r.aread()
                raise HTTPException(status_code=r.status_code, detail=body.decode())
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"
                elif line == "data: [DONE]":
                    yield "data: [DONE]\n\n"
                    return


async def _stream_aiio(
    api_key: str, model: str, payload: dict
) -> AsyncGenerator[str, None]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST", AIIO_COMPLETIONS_URL, headers=headers, json=payload
        ) as r:
            logger.debug(f"[_stream_aiio] HTTP {r.status_code} | model={model}")
            if r.status_code >= 400:
                body = await r.aread()
                raise HTTPException(status_code=r.status_code, detail=body.decode())
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    yield line + "\n\n"
                elif line == "data: [DONE]":
                    yield "data: [DONE]\n\n"
                    return


async def _call_cf(key: dict, model: str, payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {key['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{CF_API}/accounts/{key['account_id']}/ai/run/{model}"

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)

    logger.debug(f"[_call_cf] HTTP {r.status_code} | model={model}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    if not data.get("success"):
        raise HTTPException(
            status_code=502, detail=f"CF API error: {data.get('errors')}"
        )

    response_text = data.get("result", {}).get("response", "")
    return _wrap_openai_response(model, response_text)


async def _call_aiio(api_key: str, model: str, payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(AIIO_COMPLETIONS_URL, headers=headers, json=payload)

    logger.debug(f"[_call_aiio] HTTP {r.status_code} | model={model}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return r.json()


def _wrap_openai_response(model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }


@router.post("/v1/chat/completions")
async def chat_completions(req: CompletionRequest):
    db_path = get_db_path()
    provider, model = resolve_model(db_path, req.model)

    logger.info(
        f"[chat_completions] provider={provider} | model={model} | stream={req.stream} | messages={len(req.messages)}"
    )

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if provider == "cf":
        key = api_cf._get_random_key(db_path)
        model_type = _get_cf_model_type(db_path, model)
        logger.debug(f"[chat_completions] model_type={model_type}")

        if model_type == "image":
            prompt = messages[-1]["content"] if messages else ""
            payload = {"prompt": prompt}
        else:
            payload = {
                "messages": messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "stream": req.stream,
            }

        url = f"{CF_API}/accounts/{key['account_id']}/ai/run/{model}"
        logger.debug(f"[chat_completions] url=[{url}] | model_type={model_type}")

        if req.stream and model_type != "image":
            return StreamingResponse(
                _stream_cf(key, model, payload), media_type="text/event-stream"
            )

        headers = {
            "Authorization": f"Bearer {key['api_key']}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)

        logger.debug(
            f"[chat_completions] HTTP {r.status_code} | content-type={r.headers.get('content-type')}"
        )

        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)

        content_type = r.headers.get("content-type", "")

        if "image/" in content_type or "application/octet-stream" in content_type:
            import base64

            image_b64 = base64.b64encode(r.content).decode()
            return _wrap_openai_response(model, image_b64)

        data = r.json()
        if not data.get("success"):
            raise HTTPException(
                status_code=502, detail=f"CF API error: {data.get('errors')}"
            )

        result = data.get("result", {})

        if model_type == "image":
            import base64

            image_b64 = (
                result.get("image")
                or result.get("b64_json")
                or result.get("output")
                or ""
            )
            if isinstance(image_b64, str) and not image_b64.startswith("data:"):
                pass
            return _wrap_openai_response(model, image_b64)

        return _wrap_openai_response(model, result.get("response", ""))

    if provider == "aiio":
        api_key = api_aiio._get_random_key(db_path)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": req.stream,
        }
        return await _call_aiio(api_key, model, payload)

    raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")


def _get_cf_model_type(db_path: str, model: str) -> str:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT type FROM models WHERE model = ? AND provider = 'cf' LIMIT 1", (model,)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return "text"
    return row["type"]
