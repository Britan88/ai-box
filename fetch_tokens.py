import asyncio
import json
import logging
import random
import sqlite3

import httpx

from database import sync_schema

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fetch_tokens")

CF_API = "https://api.cloudflare.com/client/v4"

TOKEN_PAYLOAD = {
    "name": "auto-generated-token",
    "policies": [
        {
            "effect": "allow",
            "resources": {"com.cloudflare.api.account.zone.*": "*"},
            "permission_groups": [
                {"name": "Zone Read"},
                {"name": "Analytics Read"},
            ],
        }
    ],
}


def load_rows(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, email, global_api_key FROM cf WHERE global_api_key IS NOT NULL AND (api_key IS NULL OR api_key = '')"
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    logger.info(
        f"[load_rows] found {len(rows)} rows to process | ids={[r['id'] for r in rows]}"
    )
    return rows


def write_api_key(db_path: str, row_id: int, api_key: str, account_id: str):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE cf SET api_key = ?, account_id = ?, problems = NULL WHERE id = ?",
        (api_key, account_id, row_id),
    )
    conn.commit()
    conn.close()
    logger.info(
        f"[write_api_key] id={row_id} | api_key={api_key[:10]}... | account_id={account_id}"
    )


def write_problem(db_path: str, row_id: int, problem: dict):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE cf SET problems = ? WHERE id = ?",
        (json.dumps(problem, ensure_ascii=False), row_id),
    )
    conn.commit()
    conn.close()
    logger.warning(f"[write_problem] id={row_id} | problem={problem}")


async def fetch_permission_groups(client: httpx.AsyncClient, headers: dict) -> dict:
    r = await client.get(f"{CF_API}/user/tokens/permission_groups", headers=headers)
    logger.debug(f"[fetch_permission_groups] HTTP {r.status_code}")
    r.raise_for_status()
    groups = r.json().get("result", [])
    result = {g["name"]: g["id"] for g in groups}
    logger.debug(f"[fetch_permission_groups] loaded {len(result)} groups")
    return result


async def fetch_first_account(
    client: httpx.AsyncClient, headers: dict
) -> tuple[str, str]:
    r = await client.get(f"{CF_API}/accounts", headers=headers, params={"per_page": 1})
    logger.debug(f"[fetch_first_account] HTTP {r.status_code}")
    r.raise_for_status()
    accounts = r.json().get("result", [])
    if not accounts:
        raise ValueError("no accounts found")
    account_id = accounts[0]["id"]
    account_name = accounts[0]["name"]
    logger.debug(
        f"[fetch_first_account] account_id={account_id} | account_name={account_name}"
    )
    return account_id, account_name


async def create_token(
    client: httpx.AsyncClient, headers: dict, account_id: str, groups: dict
) -> str:
    def resolve_group(name: str) -> dict:
        gid = groups.get(name)
        if not gid:
            raise ValueError(f"permission group not found: '{name}'")
        return {"id": gid, "name": name}

    payload = {
        "name": "auto-generated-token",
        "policies": [
            {
                "effect": "allow",
                "resources": {"com.cloudflare.api.account.zone.*": "*"},
                "permission_groups": [
                    resolve_group("Zone Read"),
                    resolve_group("Analytics Read"),
                ],
            },
            {
                "effect": "allow",
                "resources": {f"com.cloudflare.api.account.{account_id}": "*"},
                "permission_groups": [
                    resolve_group("Account Settings Read"),
                    resolve_group("Account Analytics Read"),
                    resolve_group("Analytics Read"),
                    resolve_group("Workers Scripts Read"),
                    resolve_group("Workers AI Read"),
                ],
            },
        ],
    }

    logger.debug(f"[create_token] POST /user/tokens | account_id={account_id}")
    r = await client.post(f"{CF_API}/user/tokens", headers=headers, json=payload)
    logger.debug(f"[create_token] HTTP {r.status_code}")
    data = r.json()

    if not data.get("success"):
        raise ValueError(f"token creation failed: {data.get('errors', [])}")

    token_value = data["result"]["value"]
    logger.info(f"[create_token] token created | value={token_value[:10]}...")
    return token_value


async def process_row(db_path: str, row: dict):
    row_id = row["id"]
    email = row["email"]
    global_api_key = row["global_api_key"]

    logger.info(f"[process_row] START | id={row_id} | email={email}")

    headers = {
        "X-Auth-Email": email,
        "X-Auth-Key": global_api_key,
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r_user = await client.get(f"{CF_API}/user", headers=headers)
            logger.debug(f"[process_row] /user HTTP {r_user.status_code} | id={row_id}")

            if r_user.status_code == 403:
                raise ValueError(
                    f"invalid email or global_api_key | HTTP 403 | body={r_user.text}"
                )

            r_user.raise_for_status()

            account_id, account_name = await fetch_first_account(client, headers)
            groups = await fetch_permission_groups(client, headers)
            token_value = await create_token(client, headers, account_id, groups)

        write_api_key(db_path, row_id, token_value, account_id)
        logger.info(
            f"[process_row] DONE | id={row_id} | account_id={account_id} | account_name={account_name}"
        )

    except Exception as e:
        problem = {
            "type": type(e).__name__,
            "message": str(e),
            "row": {"id": row_id, "email": email},
        }
        logger.error(f"[process_row] ERROR | id={row_id} | {problem}")
        write_problem(db_path, row_id, problem)


async def run(db_path: str):
    sync_schema(db_path)
    rows = load_rows(db_path)

    if not rows:
        logger.info("[run] nothing to process")
        return

    for i, row in enumerate(rows):
        await process_row(db_path, row)

        if i < len(rows) - 1:
            delay = random.uniform(3, 5)
            logger.debug(f"[run] sleeping {delay:.2f}s before next row")
            await asyncio.sleep(delay)

    logger.info("[run] all rows processed")


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    db_path = os.getenv("DB_NAME")
    logger.info(f"[main] db_path={db_path}")
    asyncio.run(run(db_path))
