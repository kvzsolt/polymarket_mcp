# AGENTS.md — Polymarket Quant MCP Server

## 1. Architecture & Anti-Overengineering Rules
- **Flat Repository Structure:** Keep all modules at the repository root (`mcp_server.py`, `fred_client.py`, `poly_client.py`, `crypto_client.py`, `polynode_client.py`, `gov_client.py`, `politics_intel_client.py`, `sec_client.py`, `ledger_client.py`) and all tests in `tests/`. Do NOT create nested subpackages (`src/`, `core/`, `adapters/`, `services/`, `utils/`).
- **Strict Separation of Responsibilities:**
  1. `*_client.py`: Pure async HTTP/SQLite clients (`httpx2` / `httpx`, `sqlite3`), typed `Settings` loaded from env, domain math (e.g., `ExpiryPricer`, `OrderBookMath`, `ResolutionRuleAuditor`, `MergerMath`, `CalibrationMath`, `PollDropoutCalculator`), and domain exceptions (`FredError`, `PolyError`, `CryptoError`, `PolynodeError`, `GovError`, `PoliticsIntelError`, `SecError`, `LedgerError`). Zero FastMCP imports allowed here.
  2. `mcp_server.py`: Contains `SERVER_INSTRUCTIONS`, `Gateway` wrappers, `TextFormatter`, `create_server()`, and `@server.tool` functions.
- **Per-Call Settings & Transport Injection:**
  - `create_server(settings_loader=..., transport=...)` MUST accept injectable settings loaders and an optional `httpx.AsyncBaseTransport`.
  - Load settings inside the `Gateway.call()` method on each tool invocation so the MCP server can always boot and list its tools even if an API key is missing from `.env`.
  - Wrap all domain client exceptions (`FredError`, `PolyError`, `CryptoError`, `PolynodeError`, `GovError`, `PoliticsIntelError`, `SecError`, `LedgerError`) into `fastmcp.exceptions.ToolError` inside the gateway so FastMCP returns the exact error reason to the LLM.
- **Compact Text Output (`TextFormatter`):**
  - MCP tools MUST return compact, human/LLM-readable strings formatted by `TextFormatter`, never raw multi-kilobyte JSON dicts.
- **Trading Safety:**
  - Every execution tool (`place_order`, `cancel_order`) MUST default to `dry_run: bool = True` and enforce a hard `MAX_ORDER_USD` cap from settings before signing or sending any request.

## 2. Testing Rules (Read Before Writing Any Test)
Coding agents routinely write brittle, bloated tests. Follow these rules strictly:

### What to Do:
1. **Test Through `create_server()` and `httpx.MockTransport`:**
   - Test tools by instantiating `server = create_server(settings_loader=fake_settings, transport=httpx.MockTransport(handler))` and calling the tool or client directly.
   - Verify three things per tool:
     a) Outbound HTTP request correctness (URL path, query params, headers).
     b) Compact text output formatting from `TextFormatter` on valid canned responses (including edge cases like `"."` missing values in FRED or thin orderbooks in Polymarket).
     c) Domain errors (`4xx`/`5xx` HTTP responses, missing API keys) raising `ToolError` with the expected message.
2. **Test Pure Quant Math Deterministically:**
   - Pass fixed arrays of closes or orderbook levels into pricers and slippage calculators. Assert exact numerical outputs (`empirical`, `normal_7d`, `vwap_fill_price`, `kelly_fraction`).

### What NOT to Do (Strictly Forbidden):
- **NO `unittest.mock.patch` or `MagicMock`:** Never monkeypatch internal methods, global variables, or `httpx.AsyncClient`. Always use constructor injection (`settings_loader` and `MockTransport`).
- **NO Live Network Calls in Tests:** Tests must run offline in `< 1 second`.
- **NO Trivial/Redundant Tests:** Do not test that Pydantic `Field` works, do not test Python stdlib dataclasses, and do not write 15 parametrized permutations for the same string formatter. Keep total tests per tool to 2–3 high-signal cases (happy path + edge case + error translation).