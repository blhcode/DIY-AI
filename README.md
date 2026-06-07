# DIY AI

Local AI assistant for home repairs and DIY projects. Describe a problem or build idea and get:

- A project plan with difficulty, time, and cost estimates
- Step-by-step building or repair instructions
- A parts and materials list
- **Multiple store options per part** — pick where you want to shop

Powered by **Ollama** (local or LAN) and **DuckDuckGo** web search — no paid API keys.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running on this machine or another on your network
- A tool-capable model (recommended: `llama3.1:8b`)

```bash
ollama pull llama3.1:8b
ollama serve
```

## Quick start

```bash
cd "/home/isaiah/Projects/DIY AI"
cp diy.env.example diy.env
# Edit diy.env — set OLLAMA_BASE_URL to your Ollama IP
bash start.sh
```

Open the URL printed in the terminal (default port **8780**).

## Configuration (`diy.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint. Use your LAN IP for remote Ollama, e.g. `http://192.168.1.50:11434/v1` |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model name (run `ollama list` on the Ollama host) |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Per-request timeout |
| `DEFAULT_COUNTRY` | `AU` | Country preset: `AU`, `US`, or `UK` — sets default stores, DuckDuckGo region, and currency |
| `AVAILABLE_STORES` | (country default) | Comma-separated store override. Leave empty for country defaults (AU: Bunnings, Mitre 10, Amazon Australia, Total Tools, Home Timber & Hardware) |
| `DEFAULT_LOCATION` | (empty) | City/state for local searches (e.g. `Sydney`, `Melbourne`) |
| `DIY_PORT` | `8780` | Server port |

**Important:** `OLLAMA_BASE_URL` must end in `/v1`, not `/api/chat`.

## How it works

1. You describe a repair or DIY project in the chat UI.
2. The **DIY AI agent** (Ollama with tool calling) searches DuckDuckGo for guides, parts, and store listings.
3. A **synthesis pass** formats results into a structured plan (steps, parts, store options).
4. You **select a store per part** (or apply one store to all) and generate a shopping list.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Ollama status + configured stores |
| GET | `/api/stores` | List of stores from config |
| POST | `/api/chat` | `{ "messages": [{ "role": "user", "content": "..." }] }` |

## Limitations

- **Prices and stock** come from DuckDuckGo search snippets — always verify on the retailer website or in store.
- **Tool-calling quality** depends on your Ollama model; smaller models may skip store searches.
- **Safety:** The agent includes disclaimers for electrical, plumbing, gas, and structural work. Hire licensed professionals when appropriate.
- DuckDuckGo rate limits or blocks may occasionally reduce search quality.

## Project structure

```
diy.env.example     # Config template
start.sh            # Launch script
frontend/index.html # Web UI
src/
  api/main.py       # FastAPI routes
  config.py         # Settings from diy.env
  agent/            # Ollama agent + synthesizer
  providers/        # DuckDuckGo search + store URL builders
  planner/models.py # Plan JSON schema
```

## License

MIT — use freely for personal projects.
