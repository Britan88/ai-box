## Сжатый контекст проекта

### Стек
- Python, FastAPI, SQLite, httpx, asyncio
- `.env`: `DB_NAME`, `PORT`

### Файлы
| файл | назначение |
|---|---|
| `main.py` | FastAPI app, `/health` |
| `database.py` | sync_schema — создаёт/дополняет таблицы |
| `fetch_tokens.py` | берёт строки CF где нет api_key, получает токен через global_api_key, пишет api_key + account_id |
| `api_cf.py` | рандомный ключ из `cf`, get_models, call |
| `api_aiio.py` | рандомный ключ из `aiio`, get_models, call |
| `api_interface.py` | единый интерфейс `call(provider, ...)` / `get_models(provider, ...)` |
| `test_providers.py` | тестирует все модели обоих провайдеров, пишет в таблицу `models` |

### Схема БД

**`cf`**: `id` INT PK AUTOINCREMENT, `email` TEXT, `password` TEXT, `api_key` TEXT, `expire` TEXT, `ai_quota` TEXT, `global_api_key` TEXT, `otp_secret` TEXT, `recovery` TEXT, `problems` TEXT, `account_id` TEXT

**`aiio`**: `id` INT PK AUTOINCREMENT, `email` TEXT, `password` TEXT, `api_key` TEXT, `expire` TEXT, `ai_quota` TEXT, `global_api_key` TEXT, `otp_secret` TEXT, `recovery` TEXT, `problems` TEXT

**`models`**: `id` INT PK AUTOINCREMENT, `provider` TEXT, `type` TEXT, `model` TEXT, `status` TEXT, `avtime` INT, `error` TEXT

### Логика выбора ключа
- `cf`: фильтр `api_key NOT NULL`, `account_id NOT NULL`, `expire` не истёк
- `aiio`: фильтр `api_key NOT NULL`, `expire` не истёк
- рандомный `random.choice` из результата

### Список доступных моделей
Модели считаются доступными если в таблице `models` `status = 'ok'`

### Статусы моделей в тестах
- `ok` — успешный вызов
- `error` — ошибка, пишется `payload | response`
- `skip` — модель пропущена (`smart-turn`)

### Типы моделей CF и их payload
| тип | payload |
|---|---|
| `text` | `{messages, max_tokens}` |
| `image` | `{prompt}` JSON |
| `image_multipart` | `prompt` form-data |
| `image_inpainting` | `{prompt, image: array, mask_image: array}` JSON — WIP |
| `embedding` | `{text}` |
| `stt` | `{audio: bytes_list}` |
| `stt_nova3` | `{audio: {body, contentType}}` — WIP |
| `stt_deepgram_flux` | нестандартный формат — WIP |
| `tts` | `{text}` |
| `tts_prompt` | `{prompt}` (melotts) |
| `vision` | `{image: bytes_list, prompt, max_tokens}` |
| `classification_text` | `{text}` |
| `classification_image` | `{image: bytes_list}` |
| `translation` | `{text, source_lang, target_lang}` |

### Нерешённые ошибки
- `stable-diffusion-v1-5-inpainting` — mask_image формат
- `deepgram/nova-3` — audio формат
- `deepgram/flux` — sample_rate тип

---

## `API_CF.md`

```markdown
# api_cf — Cloudflare Workers AI

## Источник ключей
Таблица: `cf`
Фильтр: `api_key NOT NULL AND api_key != '' AND account_id NOT NULL AND account_id != '' AND (expire IS NULL OR expire = '' OR expire > datetime('now'))`
Выбор: random

## get_models(db_path) -> list[dict]
GET `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search`
Пагинация: page/per_page=50
Возвращает: [{id, name, task}]

## call(db_path, model, messages, max_tokens) -> str
POST `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}`
Headers: `Authorization: Bearer {api_key}`
Body: `{messages: [{role, content}], max_tokens}`
Возвращает: `result.response`

## Доступные модели
Модели считаются доступными если в таблице `models` status = 'ok' AND provider = 'cf'
```

---

## `API_AIIO.md`

```markdown
# api_aiio — intelligence.io.solutions

## Источник ключей
Таблица: `aiio`
Фильтр: `api_key NOT NULL AND api_key != '' AND (expire IS NULL OR expire = '' OR expire > ?)`
Выбор: random

## get_models(db_path) -> list[dict]
GET `https://api.intelligence.io.solutions/api/v1/models?page=1&page_size=200`
Headers: `Authorization: Bearer {api_key}`
Парсинг: `data[].id`
Возвращает: [{id, name}]

## call(db_path, model, messages, max_tokens) -> str
POST `https://api.intelligence.io.solutions/api/v1/chat/completions`
Headers: `Authorization: Bearer {api_key}`
Body: `{model, messages, max_tokens, stream: false}`
Возвращает: `choices[0].message.content`

## Доступные модели
Модели считаются доступными если в таблице `models` status = 'ok' AND provider = 'aiio'

## Ошибки
402 Payment Required — ключ без квоты, пишется как error
```

---

## `API_INTERFACE.md`

```markdown
# api_interface — универсальный интерфейс

## Провайдеры
- `cf` → api_cf
- `aiio` → api_aiio

## get_models(provider, db_path) -> list[dict]
Возвращает список моделей от провайдера

## call(provider, db_path, model, messages, max_tokens=1000) -> str
Вызывает модель, возвращает текстовый ответ

## Доступные модели (общий принцип)
SELECT model, provider FROM models WHERE status = 'ok'
```
