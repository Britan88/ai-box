import logging

from api_aiio import call as aiio_call
from api_aiio import get_models as aiio_get_models
from api_cf import call as cf_call
from api_cf import get_models as cf_get_models

logger = logging.getLogger("api_interface")

PROVIDERS = ("cf", "aiio")


def _validate_provider(provider: str):
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: '{provider}' | available: {PROVIDERS}")


async def get_models(provider: str, db_path: str) -> list[dict]:
    _validate_provider(provider)
    logger.info(f"[get_models] provider={provider}")
    if provider == "cf":
        return await cf_get_models(db_path)
    return await aiio_get_models(db_path)


async def call(
    provider: str,
    db_path: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1000,
) -> str:
    _validate_provider(provider)
    logger.info(
        f"[call] provider={provider} | model={model} | messages={len(messages)}"
    )
    if provider == "cf":
        return await cf_call(db_path, model, messages, max_tokens)
    return await aiio_call(db_path, model, messages, max_tokens)
