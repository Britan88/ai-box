import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from database import sync_schema
from router_admin import router as admin_router
from router_anthropic import router as anthropic_router
from router_completions import router as completions_router

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()
DB_NAME = os.getenv("DB_NAME")
PORT = int(os.getenv("PORT"))

logger.info(f"config loaded | DB_NAME: {DB_NAME} | PORT: {PORT}")

sync_schema(DB_NAME)

app = FastAPI()
app.mount("/static", StaticFiles(directory=".", html=True), name="static")
app.include_router(completions_router)
app.include_router(anthropic_router)
app.include_router(admin_router)


@app.get("/health")
def health():
    logger.debug("health check called")
    return {"status": "ok", "db": DB_NAME}
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Сервер работает",
        "docs": "/docs",
        "redoc": "/redoc",
        "static": "/static"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
