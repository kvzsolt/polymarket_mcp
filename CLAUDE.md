Read `@AGENTS.md` before creating files, modifying architecture, or writing tests.

## Quick Commands
- **Install dependencies:** `uv sync`
- **Inspect MCP tools:** `uv run fastmcp list mcp_server.py`
- **Run MCP server (stdio):** `uv run python mcp_server.py`
- **Run test suite:** `uv run pytest -q` (offline, `tests/test_all_tools.py`)
- **Run live end-to-end tests:** `uv run pytest -m live` (`tests/test_mcp_client.py`: real stdio server + real APIs; never sends orders)
- **Call one tool live without pytest:** `PATH=.venv/bin:$PATH uv run fastmcp call mcp_server.py search_markets tag=politics limit=5`
- **Typecheck & lint:** `uv run ruff check . && uv run mypy .`

## Code Style & Conventions
- **Python 3.11+:** Use `from __future__ import annotations`, `Annotated[..., Field(...)]` on all `@server.tool` parameters, and explicit return types on every function. No `Any` in public function signatures except raw JSON dict inputs to `TextFormatter`.
- **Environment Loading:** Always anchor `.env` loading to `Path(__file__).resolve().parent / ".env"` because MCP hosts spawn stdio processes from arbitrary working directories.
- **Adding a New Tool (Checklist):**
  1. Add the async HTTP method or pricing calculation to the appropriate `*_client.py`.
  2. Add a static formatting method to `TextFormatter` in `mcp_server.py`.
  3. Register the `@server.tool` inside `create_server()` in `mcp_server.py` using the injected gateway.
  4. Add 1 happy-path test and 1 error-translation test in `tests/test_all_tools.py` using `httpx2.MockTransport`.
## Environment (`.env` next to `mcp_server.py`)
- `FRED_API_KEY` (required for FRED tools).
- `POLYNODE_API_KEY` (required for `search_x_context`, `search_online_context`, `get_market_holders`; X Search needs a paid plan, 1 req/s, monthly quota).
- `CONGRESS_API_KEY` (required for `get_bill_status`, `get_nomination_status`; free at api.congress.gov/sign-up).
- `FEC_API_KEY` (optional, defaults to `DEMO_KEY`; the same api.data.gov key as Congress works).
- `COURTLISTENER_API_TOKEN` (required for `search_court_dockets`, `search_court_opinions`, `get_docket_entries`; v4 search/RECAP returns 401 without it).
- `THE_ODDS_API_KEY` (required for `get_bookmaker_odds`; free tier 500 credits/month).
- `SEC_USER_AGENT` (optional, recommended for `search_sec_filings`, `get_company_submissions`, e.g. `"YourOrg research@yourdomain.com"`; no API key required, polite `<= 10 req/s` throttling enforced, falls back safely with an inline note when unset).
- `QUANT_LEDGER_PATH` (optional, defaults to `.quant_ledger.sqlite3` beside `mcp_server.py`; tests inject `:memory:` or `tmp_path` for `record_forecast`, `get_active_theses`, `resolve_and_score_calibration`).
- Keyless: Polymarket Gamma/CLOB reads (`search_markets`, `search_events`, `get_market_resolution_rules`, `simulate_clob_fill`, `scan_neg_risk_event`), Binance/Deribit (`price_crypto_expiry` up to 365d + one-touch barriers), SEC EDGAR (`search_sec_filings`, `get_company_submissions`), Federal Register, GDELT (max 1 request / 5 s), Fed RSS, PredictIt. Pure deterministic math (no HTTP): `calculate_merger_spread_prob`, `analyze_poll_dropout` (fetch the RCP table with Firecrawl MCP and pass rows in).
- `POLY_MAX_ORDER_USD` (default `100`): hard cap enforced before any order is validated or sent.
- `POLY_PRIVATE_KEY`: signer key, needed only for `place_order(dry_run=False)`, `list_open_orders`, and `cancel_order(dry_run=False)`.
- `POLY_DEPOSIT_WALLET` (optional): defaults to the signer's Deposit Wallet. Live orders and order cancellations are refused unless the SDK classifies the pair as `DEPOSIT_WALLET` (signature type 3); first live use deploys an undeployed Deposit Wallet via the relayer.
- `POLY_API_KEY` / `POLY_API_SECRET` / `POLY_API_PASSPHRASE` (optional, all or none): CLOB L2 creds; derived from the key when omitted.
- Live orders and resting order management use the official SDK `polymarket-client` (`AsyncSecureClient`), behind the injectable `OrderSender` and `OrderManager` protocols in `poly_client.py`.

## Multi-MCP Wiring (Claude Code & Hermes)
- **Agent Skill:** `skills/SKILL.md` (mirrored at `.claude/skills/polymarket-quant/SKILL.md` for Claude Code auto-discovery).
- **Claude Code (`~/.claude.json` or `.mcp.json`):**
  ```json
  {
    "mcpServers": {
      "polymarket-quant": {
        "command": "uv",
        "args": ["--directory", "/path/to/polymarket_mcp", "run", "python", "mcp_server.py"]
      },
      "falcon": {
        "url": "https://mcp.polygun.com/mcp"
      },
      "firecrawl": {
        "command": "npx",
        "args": ["-y", "firecrawl-mcp"]
      }
    }
  }
  ```
- **Hermes Sync:** Run `hermes import-agent claude-code --sync` to import `.claude/skills/polymarket-quant/SKILL.md` and the multi-MCP configuration directly into Hermes.
