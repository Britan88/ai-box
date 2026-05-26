import logging
import random
import sqlite3

import httpx

CF_API = "https://api.cloudflare.com/client/v4"

logger = logging.getLogger("api_cf")


def _get_random_key(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, api_key, account_id FROM cf
        WHERE api_key IS NOT NULL AND api_key != ''
          AND account_id IS NOT NULL AND account_id != ''
          AND (expire IS NULL OR expire = '' OR expire > datetime('now'))
        """
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    logger.debug(f"[_get_random_key] available keys: {len(rows)}")
    if not rows:
        raise RuntimeError("no valid cf keys in db")
    key = random.choice(rows)
    logger.debug(
        f"[_get_random_key] selected id={key['id']} | account_id={key['account_id']}"
    )
    return key


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


async def get_models(db_path: str) -> list[dict]:
    key = _get_random_key(db_path)
    account_id = key["account_id"]
    headers = _headers(key["api_key"])

    results = []
    page = 1

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            r = await client.get(
                f"{CF_API}/accounts/{account_id}/ai/models/search",
                headers=headers,
                params={"page": page, "per_page": 50},
            )
            logger.debug(f"[get_models] HTTP {r.status_code} | page={page}")
            r.raise_for_status()
            data = r.json()
            if not data.get("success"):
                raise RuntimeError(f"cf api error: {data.get('errors')}")
            items = data.get("result", [])
            results.extend(items)
            info = data.get("result_info", {})
            if page >= info.get("total_pages", 1) or not items:
                break
            page += 1

    logger.info(f"[get_models] total models: {len(results)}")
    return [
        {
            "id": m.get("name"),
            "name": m.get("name"),
            "task": m.get("task", {}).get("name") if m.get("task") else None,
        }
        for m in results
    ]


async def call(
    db_path: str, model: str, messages: list[dict], max_tokens: int = 1000
) -> str:
    key = _get_random_key(db_path)
    account_id = key["account_id"]
    headers = _headers(key["api_key"])

    payload = {"messages": messages, "max_tokens": max_tokens}

    logger.info(
        f"[call] model={model} | account_id={account_id} | messages={len(messages)}"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{CF_API}/accounts/{account_id}/ai/run/{model}",
            headers=headers,
            json=payload,
        )
        logger.debug(f"[call] HTTP {r.status_code}")
        r.raise_for_status()
        data = r.json()

    if not data.get("success"):
        raise RuntimeError(f"cf ai run failed: {data.get('errors')}")

    response_text = data.get("result", {}).get("response", "")
    logger.info(f"[call] response_len={len(response_text)}")
    return response_text
