import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from database import sync_schema
from router_admin import router as admin_router
from router_anthropic import router as anthropic_router
from router_completions import router as completions_router

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ================== CONFIG ==================
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
PORT = int(os.getenv("PORT", 8000))

logger.info(f"config loaded | DB_NAME: {DB_NAME} | PORT: {PORT}")

# ================== DATABASE ==================
sync_schema(DB_NAME)

# ================== FASTAPI APP ==================
app = FastAPI(title="AI Box", version="1.0")

# Монтирование статики (рекомендуется создать папку static)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Подключаем роутеры
app.include_router(completions_router)
app.include_router(anthropic_router)
app.include_router(admin_router)


# ================== MAIN ROUTES ==================
@app.get("/", response_class=FileResponse)
async def root():
    """Главная страница"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return {
            "status": "ok",
            "message": "AI Box Server is running",
            "docs": "/docs",
            "admin": "/admin"
        }


@app.get("/admin", response_class=FileResponse)
async def admin_dashboard():
    """Админка"""
    if os.path.exists("admin-dash.html"):
        return FileResponse("admin-dash.html")
    elif os.path.exists("static/admin-dash.html"):
        return FileResponse("static/admin-dash.html")
    else:
        return RedirectResponse(url="/admin/stats")


@app.get("/health")
def health():
    logger.debug("health check called")
    return {"status": "ok", "db": DB_NAME}


# ================== START ==================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,           # убрать на продакшене
        log_level="info"
    )
