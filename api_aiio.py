import logging
import random
import sqlite3
from datetime import datetime, timezone

import httpx

AIIO_COMPLETIONS_URL = "https://api.intelligence.io.solutions/api/v1/chat/completions"
AIIO_MODELS_URL = (
    "https://api.intelligence.io.solutions/api/v1/models?page=1&page_size=200"
)

logger = logging.getLogger("api_aiio")


def _get_random_key(db_path: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, api_key FROM aiio
        WHERE api_key IS NOT NULL AND api_key != ''
          AND (expire IS NULL OR expire = '' OR expire > ?)
        """,
        (now,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    logger.debug(f"[_get_random_key] available keys: {len(rows)}")
    if not rows:
        raise RuntimeError("no valid aiio keys in db")
    key = random.choice(rows)
    logger.debug(f"[_get_random_key] selected id={key['id']}")
    return key["api_key"]


async def get_models(db_path: str) -> list[dict]:
    api_key = _get_random_key(db_path)
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(AIIO_MODELS_URL, headers=headers)
        logger.debug(f"[get_models] HTTP {r.status_code}")
        r.raise_for_status()
        data = r.json()

    models = [
        {"id": m.get("id"), "name": m.get("id")}
        for m in data.get("data", [])
        if m.get("id")
    ]
    logger.info(f"[get_models] total models: {len(models)}")
    return models


async def call(
    db_path: str, model: str, messages: list[dict], max_tokens: int = 1000
) -> str:
    api_key = _get_random_key(db_path)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }

    logger.info(f"[call] model={model} | messages={len(messages)}")

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(AIIO_COMPLETIONS_URL, headers=headers, json=payload)
        logger.debug(f"[call] HTTP {r.status_code}")
        r.raise_for_status()
        data = r.json()

    response_text = data["choices"][0]["message"]["content"]
    logger.info(f"[call] response_len={len(response_text)}")
    return response_text
