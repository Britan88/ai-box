# Model Router API

Local FastAPI proxy that routes LLM and AI model requests to Cloudflare Workers AI and intelligence.io.solutions (aiio).

## Setup

```env
DB_NAME=local.db
PORT=8000
```

Run:
```bash
python main.py
```

---

## Endpoints

### `GET /health`
Health check.

**Response:**
```json
{"status": "ok", "db": "local.db"}
```

---

### `POST /v1/chat/completions`
OpenAI-compatible completions endpoint.

**Request:**
```json
{
  "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 1000,
  "temperature": 1.0,
  "stream": false
}
```

**Response (text model):**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello!"},
      "finish_reason": "stop"
    }
  ],
  "usage": null
}
```

**Response (image model):**
```json
{
  "choices": [
    {
      "message": {"role": "assistant", "content": "<base64 image string>"}
    }
  ]
}
```

**Streaming:** set `"stream": true` — returns `text/event-stream` SSE.

---

### `POST /v1/messages`
Anthropic-compatible messages endpoint. Used by Claude Code and other Anthropic SDK clients.

**Request:**
```json
{
  "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 1000,
  "stream": false,
  "system": "You are a helpful assistant."
}
```

**Response:**
```json
{
  "id": "msg_1234567890",
  "type": "message",
  "role": "assistant",
  "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  "content": [{"type": "text", "text": "Hello!"}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
```

**Streaming:** set `"stream": true` — returns Anthropic SSE format (`message_start`, `content_block_delta`, `message_stop`).

---

## Providers

| Provider | Tables | Models source |
|---|---|---|
| `cf` | `cf` | Cloudflare Workers AI |
| `aiio` | `aiio` | intelligence.io.solutions |

Active providers are set in `router_completions.py` and `router_anthropic.py`:
```python
ACTIVE_PROVIDERS: list[str] = ["cf"]
```

---

## Model availability

Models are available if they appear in the `models` table with `status = 'ok'`.

Run model tests to populate:
```bash
python test_providers.py
```

---

## Claude Code integration

```bat
@echo off
set ANTHROPIC_BASE_URL=http://localhost:8000
set ANTHROPIC_AUTH_TOKEN=any
set ANTHROPIC_API_KEY=any
set ANTHROPIC_MODEL=@cf/meta/llama-3.3-70b-instruct-fp8-fast
set ANTHROPIC_SMALL_FAST_MODEL=@cf/meta/llama-3.2-1b-instruct
set CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
call "C:\Users\%USERNAME%\AppData\Roaming\npm\claude.cmd" %*
```

---

## Database schema

### `cf`
| column | type | description |
|---|---|---|
| id | INTEGER PK | autoincrement |
| email | TEXT | CF account email |
| password | TEXT | |
| api_key | TEXT | CF API token |
| expire | TEXT | ISO8601 expiry |
| ai_quota | TEXT | JSON quota info |
| global_api_key | TEXT | CF Global API Key |
| otp_secret | TEXT | |
| recovery | TEXT | |
| problems | TEXT | last error state |
| account_id | TEXT | CF account ID |

### `aiio`
| column | type | description |
|---|---|---|
| id | INTEGER PK | autoincrement |
| email | TEXT | |
| password | TEXT | |
| api_key | TEXT | aiio API key |
| expire | TEXT | ISO8601 expiry |
| ai_quota | TEXT | JSON quota info |
| global_api_key | TEXT | |
| otp_secret | TEXT | |
| recovery | TEXT | |
| problems | TEXT | last error state |

### `models`
| column | type | description |
|---|---|---|
| id | INTEGER PK | autoincrement |
| provider | TEXT | `cf` or `aiio` |
| type | TEXT | `text`, `image`, `embedding`, `stt`, `tts`, etc. |
| model | TEXT | model ID |
| status | TEXT | `ok`, `error`, `skip` |
| avtime | INTEGER | response time ms |
| error | TEXT | `payload \| response` on error |

---

## Scripts

| script | description |
|---|---|
| `main.py` | FastAPI server |
| `database.py` | schema sync on startup |
| `fetch_tokens.py` | fetch CF API tokens using global_api_key |
| `test_providers.py` | test all models, write results to `models` table |
| `api_cf.py` | CF Workers AI client |
| `api_aiio.py` | aiio client |
| `api_interface.py` | unified provider interface |
| `router_completions.py` | OpenAI-compatible router |
| `router_anthropic.py` | Anthropic-compatible router |
