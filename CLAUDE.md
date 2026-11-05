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
- `COURTLISTENER_API_TOKEN` (required for `search_court_dockets`; v4 search returns 401 without it).
- `THE_ODDS_API_KEY` (required for `get_bookmaker_odds`; free tier 500 credits/month).
- Keyless: Federal Register, GDELT (max 1 request / 5 s), Fed RSS, PredictIt. `analyze_poll_dropout` does no HTTP: fetch the RCP table with the Firecrawl MCP and pass the rows in.
- `POLY_MAX_ORDER_USD` (default `100`): hard cap enforced before any order is validated or sent.
- `POLY_PRIVATE_KEY`: signer key, needed only for `place_order(dry_run=False)`.
- `POLY_DEPOSIT_WALLET` (optional): defaults to the signer's Deposit Wallet. Live orders are refused unless the SDK classifies the pair as `DEPOSIT_WALLET` (signature type 3); first live use deploys an undeployed Deposit Wallet via the relayer.
- `POLY_API_KEY` / `POLY_API_SECRET` / `POLY_API_PASSPHRASE` (optional, all or none): CLOB L2 creds; derived from the key when omitted.
- Live orders use the official SDK `polymarket-client` (`AsyncSecureClient`), behind the injectable `OrderSender` in `poly_client.py`.
