"""
Admin API — Model Router
Provides statistics, account info, model test results, and playground proxy.
Mount on main FastAPI app: app.include_router(admin_router)
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("router_admin")

router = APIRouter(prefix="/admin", tags=["admin"])

DB_NAME = os.getenv("DB_NAME", "local.db")

CF_API = "https://api.cloudflare.com/client/v4"
AIIO_COMPLETIONS_URL = "https://api.intelligence.io.solutions/api/v1/chat/completions"

PROMPT_TEXT_PATH = os.getenv("TEXT_PROMPT_PATH", "prompts/text_prompt.md")
PROMPT_IMAGE_PATH = os.getenv("IMAGE_PROMPT_PATH", "prompts/img_prompt.md")

# in-memory test job registry: job_id -> {status, total, done, results, error}
_test_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


@contextmanager
def get_db(db_path: str = None):
    path = db_path or DB_NAME
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def rows_as_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PlaygroundRequest(BaseModel):
    provider: str  # "cf" | "aiio"
    model: str
    messages: list[dict]
    max_tokens: int = 1000
    temperature: float = 1.0
    stream: bool = False
    system: Optional[str] = None  # prepend as system message
    model_type: Optional[str] = None  # "text" | "image" | "embedding" | ...


# ---------------------------------------------------------------------------
# /admin/stats — общая сводка
# ---------------------------------------------------------------------------


@router.get("/stats")
def get_stats():
    """
    Aggregated overview:
    - account counts per provider (total / active / expired / problems)
    - models table: total / ok / error / skip — by provider and by type
    - last_updated: newest row in models table
    """
    with get_db() as conn:
        cur = conn.cursor()

        # ── CF accounts ──────────────────────────────────────────────────
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN api_key IS NOT NULL AND api_key != '' THEN 1 ELSE 0 END) AS has_key,
                SUM(CASE WHEN expire IS NOT NULL AND expire != ''
                              AND expire <= datetime('now') THEN 1 ELSE 0 END) AS expired,
                SUM(CASE WHEN problems IS NOT NULL AND problems != '' THEN 1 ELSE 0 END) AS has_problems
            FROM cf
        """)
        cf_row = dict(cur.fetchone())

        # ── aiio accounts ────────────────────────────────────────────────
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN api_key IS NOT NULL AND api_key != '' THEN 1 ELSE 0 END) AS has_key,
                SUM(CASE WHEN expire IS NOT NULL AND expire != ''
                              AND expire <= datetime('now') THEN 1 ELSE 0 END) AS expired,
                SUM(CASE WHEN problems IS NOT NULL AND problems != '' THEN 1 ELSE 0 END) AS has_problems
            FROM aiio
        """)
        aiio_row = dict(cur.fetchone())

        # ── models summary ───────────────────────────────────────────────
        cur.execute("""
            SELECT
                provider,
                type,
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'ok'    THEN 1 ELSE 0 END) AS ok,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error,
                SUM(CASE WHEN status = 'skip'  THEN 1 ELSE 0 END) AS skip,
                ROUND(AVG(CASE WHEN status = 'ok' AND avtime IS NOT NULL THEN avtime END)) AS avg_avtime_ms,
                MIN(CASE WHEN status = 'ok' AND avtime IS NOT NULL THEN avtime END) AS min_avtime_ms,
                MAX(CASE WHEN status = 'ok' AND avtime IS NOT NULL THEN avtime END) AS max_avtime_ms
            FROM models
            GROUP BY provider, type
            ORDER BY provider, type
        """)
        models_by_type = rows_as_dicts(cur.fetchall())

        cur.execute("""
            SELECT
                provider,
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'ok'    THEN 1 ELSE 0 END) AS ok,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error,
                SUM(CASE WHEN status = 'skip'  THEN 1 ELSE 0 END) AS skip
            FROM models
            GROUP BY provider
        """)
        models_by_provider = rows_as_dicts(cur.fetchall())

        cur.execute("SELECT COUNT(*) AS total FROM models")
        models_total = cur.fetchone()["total"]

        # check if models table has rowid/updated_at — fall back to rowid
        cur.execute("SELECT MAX(id) AS last_id FROM models")
        last_model_id = cur.fetchone()["last_id"]

    return {
        "accounts": {
            "cf": {
                "total": cf_row["total"],
                "with_key": cf_row["has_key"],
                "expired": cf_row["expired"],
                "active": (cf_row["has_key"] or 0) - (cf_row["expired"] or 0),
                "has_problems": cf_row["has_problems"],
            },
            "aiio": {
                "total": aiio_row["total"],
                "with_key": aiio_row["has_key"],
                "expired": aiio_row["expired"],
                "active": (aiio_row["has_key"] or 0) - (aiio_row["expired"] or 0),
                "has_problems": aiio_row["has_problems"],
            },
        },
        "models": {
            "total": models_total,
            "last_tested_id": last_model_id,
            "by_provider": models_by_provider,
            "by_type": models_by_type,
        },
    }


# ---------------------------------------------------------------------------
# /admin/accounts/{provider} — список аккаунтов
# ---------------------------------------------------------------------------


@router.get("/accounts/{provider}")
def get_accounts(
    provider: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    has_problems: Optional[bool] = Query(None),
    expired: Optional[bool] = Query(None),
    search: Optional[str] = Query(None, description="search in email"),
):
    """
    Paginated list of CF or aiio accounts.
    Sensitive fields (password, otp_secret, recovery) are redacted.
    """
    if provider not in ("cf", "aiio"):
        raise HTTPException(status_code=400, detail="provider must be 'cf' or 'aiio'")

    conditions = []
    params = []

    if has_problems is True:
        conditions.append("problems IS NOT NULL AND problems != ''")
    elif has_problems is False:
        conditions.append("(problems IS NULL OR problems = '')")

    if expired is True:
        conditions.append(
            "expire IS NOT NULL AND expire != '' AND expire <= datetime('now')"
        )
    elif expired is False:
        conditions.append("(expire IS NULL OR expire = '' OR expire > datetime('now'))")

    if search:
        conditions.append("email LIKE ?")
        params.append(f"%{search}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    offset = (page - 1) * per_page

    # redact sensitive columns
    if provider == "cf":
        select_cols = """
            id, account_id, email,
            CASE WHEN api_key IS NOT NULL AND api_key != '' THEN '***' ELSE NULL END AS api_key,
            expire, ai_quota,
            CASE WHEN global_api_key IS NOT NULL AND global_api_key != '' THEN '***' ELSE NULL END AS global_api_key,
            problems
        """
    else:
        select_cols = """
            id, email,
            CASE WHEN api_key IS NOT NULL AND api_key != '' THEN '***' ELSE NULL END AS api_key,
            expire, ai_quota,
            CASE WHEN global_api_key IS NOT NULL AND global_api_key != '' THEN '***' ELSE NULL END AS global_api_key,
            problems
        """

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {provider} {where}", params)
        total = cur.fetchone()["cnt"]
        cur.execute(
            f"SELECT {select_cols} FROM {provider} {where} ORDER BY id LIMIT ? OFFSET ?",
            params + [per_page, offset],
        )
        rows = rows_as_dicts(cur.fetchall())

    # parse ai_quota JSON if present
    for row in rows:
        if row.get("ai_quota"):
            try:
                row["ai_quota"] = json.loads(row["ai_quota"])
            except Exception:
                pass

    return {
        "provider": provider,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, -(-total // per_page)),
        "accounts": rows,
    }


# ---------------------------------------------------------------------------
# /admin/accounts/{provider}/{account_id} — детали аккаунта
# ---------------------------------------------------------------------------


@router.get("/accounts/{provider}/{account_id}")
def get_account(provider: str, account_id: int):
    if provider not in ("cf", "aiio"):
        raise HTTPException(status_code=400, detail="provider must be 'cf' or 'aiio'")

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {provider} WHERE id = ?", (account_id,))
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="account not found")

    data = dict(row)
    # redact
    for field in ("password", "otp_secret", "recovery"):
        if data.get(field):
            data[field] = "***"
    if data.get("ai_quota"):
        try:
            data["ai_quota"] = json.loads(data["ai_quota"])
        except Exception:
            pass

    return data


# ---------------------------------------------------------------------------
# /admin/accounts/cf/{account_id}/zones  — зоны аккаунта через CF API
# /admin/accounts/cf/{account_id}/workers — воркеры аккаунта через CF API
# account_id здесь — числовой id строки в БД (не CF account_id)
# ---------------------------------------------------------------------------


async def _get_cf_key_for(db_id: int) -> dict:
    """Fetch api_key + cf account_id for a specific DB row id."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT api_key, account_id FROM cf WHERE id = ?", (db_id,))
        row = cur.fetchone()
    if not row or not row["api_key"] or not row["account_id"]:
        raise HTTPException(
            status_code=404, detail="account not found or has no api_key"
        )
    return dict(row)


@router.get("/accounts/cf/{account_id}/zones")
async def get_cf_account_zones(account_id: int, page: int = Query(1, ge=1)):
    key = await _get_cf_key_for(account_id)
    headers = {
        "Authorization": f"Bearer {key['api_key']}",
        "Content-Type": "application/json",
    }
    cf_account_id = key["account_id"]

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{CF_API}/zones",
            headers=headers,
            params={"account.id": cf_account_id, "per_page": 50, "page": page},
        )

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    if not data.get("success"):
        raise HTTPException(status_code=502, detail=str(data.get("errors")))

    zones = [
        {
            "id": z["id"],
            "name": z["name"],
            "status": z["status"],
            "paused": z.get("paused", False),
            "plan": z.get("plan", {}).get("name"),
            "type": z.get("type"),
            "modified_on": z.get("modified_on"),
            "created_on": z.get("created_on"),
        }
        for z in data.get("result", [])
    ]

    info = data.get("result_info", {})
    return {
        "total": info.get("total_count", len(zones)),
        "page": info.get("page", page),
        "per_page": info.get("per_page", 50),
        "pages": info.get("total_pages", 1),
        "zones": zones,
    }


@router.get("/accounts/cf/{account_id}/workers")
async def get_cf_account_workers(account_id: int):
    key = await _get_cf_key_for(account_id)
    headers = {
        "Authorization": f"Bearer {key['api_key']}",
        "Content-Type": "application/json",
    }
    cf_account_id = key["account_id"]

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{CF_API}/accounts/{cf_account_id}/workers/scripts",
            headers=headers,
        )

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    if not data.get("success"):
        raise HTTPException(status_code=502, detail=str(data.get("errors")))

    workers = [
        {
            "id": w["id"],
            "modified_on": w.get("modified_on"),
            "created_on": w.get("created_on"),
            "usage_model": w.get("usage_model"),
        }
        for w in data.get("result", [])
    ]

    return {"total": len(workers), "workers": workers}


# ---------------------------------------------------------------------------
# /admin/models — результаты тестирования
# ---------------------------------------------------------------------------


@router.get("/models")
def get_models(
    provider: Optional[str] = Query(None, description="cf | aiio"),
    type: Optional[str] = Query(None, description="text | image | embedding | ..."),
    status: Optional[str] = Query(None, description="ok | error | skip"),
    page: int = Query(1, ge=1),
    per_page: int = Query(100, ge=1, le=500),
    order_by: str = Query("avtime", description="avtime | model | status | type"),
    order_dir: str = Query("asc", description="asc | desc"),
):
    """
    Full model test results table with filtering and pagination.
    """
    valid_order = {"avtime", "model", "status", "type", "id", "provider"}
    if order_by not in valid_order:
        order_by = "avtime"
    if order_dir not in ("asc", "desc"):
        order_dir = "asc"

    conditions = []
    params = []

    if provider:
        conditions.append("provider = ?")
        params.append(provider)
    if type:
        conditions.append("type = ?")
        params.append(type)
    if status:
        conditions.append("status = ?")
        params.append(status)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (page - 1) * per_page

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) AS cnt FROM models {where}", params)
        total = cur.fetchone()["cnt"]
        cur.execute(
            f"""
            SELECT id, provider, type, model, status, avtime, error, result
            FROM models {where}
            ORDER BY {order_by} {order_dir}
            LIMIT ? OFFSET ?
            """,
            params + [per_page, offset],
        )
        rows = rows_as_dicts(cur.fetchall())

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, -(-total // per_page)),
        "filters": {"provider": provider, "type": type, "status": status},
        "models": rows,
    }


# ---------------------------------------------------------------------------
# /admin/models/available — только рабочие модели (status=ok)
# ---------------------------------------------------------------------------


@router.get("/models/available")
def get_available_models(
    provider: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
):
    """
    Shortcut: only models with status='ok', grouped by provider and type.
    """
    conditions = ["status = 'ok'"]
    params = []

    if provider:
        conditions.append("provider = ?")
        params.append(provider)
    if type:
        conditions.append("type = ?")
        params.append(type)

    where = "WHERE " + " AND ".join(conditions)

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT provider, type, model, avtime
            FROM models {where}
            ORDER BY provider, type, avtime ASC
            """,
            params,
        )
        rows = rows_as_dicts(cur.fetchall())

    # group by provider → type → models
    grouped: dict = {}
    for r in rows:
        p = r["provider"]
        t = r["type"] or "unknown"
        grouped.setdefault(p, {}).setdefault(t, []).append(
            {"model": r["model"], "avtime_ms": r["avtime"]}
        )

    return {
        "total": len(rows),
        "providers": grouped,
    }


# ---------------------------------------------------------------------------
# /admin/models/errors — только ошибочные
# ---------------------------------------------------------------------------


@router.get("/models/errors")
def get_model_errors(
    provider: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    conditions = ["status = 'error'"]
    params = []

    if provider:
        conditions.append("provider = ?")
        params.append(provider)

    where = "WHERE " + " AND ".join(conditions)
    offset = (page - 1) * per_page

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) AS cnt FROM models {where}", params)
        total = cur.fetchone()["cnt"]
        cur.execute(
            f"""
            SELECT id, provider, type, model, avtime, error, result
            FROM models {where}
            ORDER BY provider, model
            LIMIT ? OFFSET ?
            """,
            params + [per_page, offset],
        )
        rows = rows_as_dicts(cur.fetchall())

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, -(-total // per_page)),
        "errors": rows,
    }


# ---------------------------------------------------------------------------
# /admin/models/{model_id} — инфо по конкретной модели
# (model name URL-encoded, e.g. %40cf%2Fmeta%2Fllama...)
# ---------------------------------------------------------------------------


@router.get("/models/detail")
def get_model_detail(model: str = Query(..., description="full model id")):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM models WHERE model = ? ORDER BY provider",
            (model,),
        )
        rows = rows_as_dicts(cur.fetchall())

    if not rows:
        raise HTTPException(status_code=404, detail="model not found in test results")

    return {"model": model, "results": rows}


# ---------------------------------------------------------------------------
# /admin/accounts/{provider}/{account_id}/problems — update problems field
# ---------------------------------------------------------------------------


@router.delete("/accounts/{provider}/{account_id}/problems")
def clear_problems(provider: str, account_id: int):
    """Clear the problems field for an account (mark as resolved)."""
    if provider not in ("cf", "aiio"):
        raise HTTPException(status_code=400, detail="provider must be 'cf' or 'aiio'")

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE {provider} SET problems = NULL WHERE id = ?", (account_id,)
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="account not found")
        conn.commit()

    return {"success": True, "account_id": account_id, "provider": provider}


# ---------------------------------------------------------------------------
# /admin/playground — proxy single completion, non-streaming
# ---------------------------------------------------------------------------


@router.post("/playground")
async def playground(req: PlaygroundRequest):
    """
    Direct completion call for playground use.
    Picks a random valid key from DB for the given provider, calls the API,
    returns the response with latency info.
    Supports stream=True — returns SSE.
    """
    t0 = time.perf_counter()

    if req.provider not in ("cf", "aiio"):
        raise HTTPException(status_code=400, detail="provider must be 'cf' or 'aiio'")

    # build messages with optional system prompt
    messages = list(req.messages)
    if req.system and not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": req.system}] + messages

    if req.provider == "cf":
        return await _playground_cf(req, messages, t0)
    return await _playground_aiio(req, messages, t0)


async def _get_cf_key() -> dict:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, api_key, account_id FROM cf
            WHERE api_key IS NOT NULL AND api_key != ''
              AND account_id IS NOT NULL AND account_id != ''
              AND (expire IS NULL OR expire = '' OR expire > datetime('now'))
            ORDER BY RANDOM() LIMIT 1
        """)
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=503, detail="no valid CF keys in db")
    return dict(row)


async def _get_aiio_key() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT api_key FROM aiio
            WHERE api_key IS NOT NULL AND api_key != ''
              AND (expire IS NULL OR expire = '' OR expire > ?)
            ORDER BY RANDOM() LIMIT 1
        """,
            (now,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=503, detail="no valid aiio keys in db")
    return row["api_key"]


async def _playground_cf(req: PlaygroundRequest, messages: list, t0: float):
    key = await _get_cf_key()
    headers = {
        "Authorization": f"Bearer {key['api_key']}",
        "Content-Type": "application/json",
    }

    is_image = (req.model_type or "").lower() in ("image", "imagegen", "text-to-image")

    if is_image:
        # CF image models expect {"prompt": "..."} not messages
        user_prompt = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        payload = {"prompt": user_prompt}
    else:
        payload = {
            "messages": messages,
            "max_tokens": req.max_tokens,
            "stream": req.stream,
        }

    url = f"{CF_API}/accounts/{key['account_id']}/ai/run/{req.model}"

    if req.stream and not is_image:

        async def generate():
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as r:
                    if r.status_code >= 400:
                        body = await r.aread()
                        yield f"data: {json.dumps({'error': r.status_code, 'detail': body.decode()})}\n\n"
                        return
                    async for line in r.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    if is_image:
        import base64

        return {
            "provider": "cf",
            "model": req.model,
            "account_id": key["account_id"],
            "key_id": key["id"],
            "image_b64": base64.b64encode(r.content).decode(),
            "latency_ms": elapsed_ms,
        }

    data = r.json()
    if not data.get("success"):
        raise HTTPException(
            status_code=502, detail=f"CF API error: {data.get('errors')}"
        )

    result = data.get("result", {})
    return {
        "provider": "cf",
        "model": req.model,
        "account_id": key["account_id"],
        "key_id": key["id"],
        "response": result.get("response", ""),
        "usage": result.get("usage"),
        "latency_ms": elapsed_ms,
    }


async def _playground_aiio(req: PlaygroundRequest, messages: list, t0: float):
    api_key = await _get_aiio_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }

    if req.stream:

        async def generate():
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", AIIO_COMPLETIONS_URL, headers=headers, json=payload
                ) as r:
                    if r.status_code >= 400:
                        body = await r.aread()
                        yield f"data: {json.dumps({'error': r.status_code, 'detail': body.decode()})}\n\n"
                        return
                    async for line in r.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(AIIO_COMPLETIONS_URL, headers=headers, json=payload)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    data = r.json()
    choice = data.get("choices", [{}])[0]
    return {
        "provider": "aiio",
        "model": req.model,
        "response": choice.get("message", {}).get("content", ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "latency_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# /admin/health — db ping + счётчики в одном запросе
# ---------------------------------------------------------------------------


@router.get("/health")
def admin_health():
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS n FROM cf")
            cf_count = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM aiio")
            aiio_count = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM models WHERE status = 'ok'")
            models_ok = cur.fetchone()["n"]
        db_ok = True
    except Exception as e:
        cf_count = aiio_count = models_ok = 0
        db_ok = False
        logger.error(f"[admin/health] db error: {e}")

    return {
        "status": "ok" if db_ok else "db_error",
        "db": DB_NAME,
        "cf_accounts": cf_count,
        "aiio_accounts": aiio_count,
        "models_ok": models_ok,
        "time": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# /admin/prompts — read / write prompt files
# ---------------------------------------------------------------------------


@router.get("/prompts")
def get_prompts():
    def _read(path: str) -> str:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    return {
        "text": {"path": PROMPT_TEXT_PATH, "content": _read(PROMPT_TEXT_PATH)},
        "image": {"path": PROMPT_IMAGE_PATH, "content": _read(PROMPT_IMAGE_PATH)},
    }


class PromptUpdate(BaseModel):
    type: str  # "text" | "image"
    content: str


@router.put("/prompts")
def update_prompt(req: PromptUpdate):
    if req.type not in ("text", "image"):
        raise HTTPException(status_code=400, detail="type must be 'text' or 'image'")
    path = PROMPT_TEXT_PATH if req.type == "text" else PROMPT_IMAGE_PATH
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(req.content)
    return {"ok": True, "path": path}


# ---------------------------------------------------------------------------
# /admin/test/run — async test runner
# /admin/test/status/{job_id} — poll job status
# ---------------------------------------------------------------------------


class TestRunRequest(BaseModel):
    models: list[str] | None = None  # explicit model ids; if None — use type filter
    types: list[str] | None = None  # e.g. ["text", "image"]; ignored when models set
    provider: str | None = None  # filter by provider when using types


@router.post("/test/run")
async def test_run(req: TestRunRequest):
    # resolve model list from DB
    with get_db() as conn:
        cur = conn.cursor()
        if req.models:
            placeholders = ",".join("?" for _ in req.models)
            cur.execute(
                f"SELECT provider, type, model FROM models WHERE model IN ({placeholders}) GROUP BY model",
                req.models,
            )
        else:
            conditions = ["status != 'skip'"]
            params: list = []
            if req.types:
                placeholders = ",".join("?" for _ in req.types)
                conditions.append(f"type IN ({placeholders})")
                params.extend(req.types)
            if req.provider:
                conditions.append("provider = ?")
                params.append(req.provider)
            where = "WHERE " + " AND ".join(conditions)
            cur.execute(
                f"SELECT provider, type, model FROM models {where} GROUP BY model",
                params,
            )
        model_rows = rows_as_dicts(cur.fetchall())

    if not model_rows:
        raise HTTPException(status_code=404, detail="no models match the filter")

    job_id = str(uuid.uuid4())[:8]
    _test_jobs[job_id] = {
        "status": "running",
        "total": len(model_rows),
        "done": 0,
        "results": [],
        "error": None,
    }

    asyncio.create_task(_run_test_job(job_id, model_rows))
    return {"job_id": job_id, "total": len(model_rows)}


async def _run_test_job(job_id: str, model_rows: list[dict]):
    import importlib

    try:
        tp = importlib.import_module("test_providers")
    except ImportError as e:
        _test_jobs[job_id]["status"] = "error"
        _test_jobs[job_id]["error"] = f"cannot import test_providers: {e}"
        return

    # reload prompts from files so dashboard edits take effect without restart
    tp.TEXT_PROMPT = tp._load_text_prompt(PROMPT_TEXT_PATH)
    tp.IMAGE_PROMPT = tp._load_image_prompt(PROMPT_IMAGE_PATH)

    IMAGE_TYPES = {"image", "image_multipart", "image_inpainting"}

    job = _test_jobs[job_id]
    for row in model_rows:
        try:
            status, avtime, error, result = await tp.TYPE_HANDLERS.get(
                row["type"], tp.test_text
            )(row["provider"], row["model"])
            tp.write_result(
                DB_NAME,
                row["provider"],
                row["type"],
                row["model"],
                status,
                avtime,
                error,
                result,
            )
            job["results"].append(
                {
                    "model": row["model"],
                    "type": row["type"],
                    "provider": row["provider"],
                    "status": status,
                    "avtime": avtime,
                    "error": error,
                    "result": result,
                }
            )
        except Exception as e:
            logger.error(f"[test_job] {row['model']} failed: {e}")
            job["results"].append(
                {
                    "model": row["model"],
                    "type": row["type"],
                    "provider": row["provider"],
                    "status": "error",
                    "avtime": 0,
                    "error": str(e),
                    "result": None,
                }
            )
        job["done"] += 1

    job["status"] = "done"


@router.get("/test/status/{job_id}")
def test_status(job_id: str):
    job = _test_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job
