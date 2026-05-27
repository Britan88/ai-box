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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
DB_NAME = os.getenv("DB_NAME")
PORT = int(os.getenv("PORT", 8000))

sync_schema(DB_NAME)

app = FastAPI(title="AI Box")

# Монтируем статику
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
logger.info("✅ Static folder mounted")

app.include_router(completions_router)
app.include_router(anthropic_router)
app.include_router(admin_router)


@app.get("/")
async def root():
    return RedirectResponse(url="/admin")


@app.get("/admin", response_class=FileResponse)
async def admin_dashboard():
    return FileResponse("static/admin-dash.html")


@app.get("/health")
def health():
    return {"status": "ok", "db": DB_NAME}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
