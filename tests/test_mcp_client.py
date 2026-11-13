"""Live end-to-end checks: real ``mcp_server.py`` subprocess over stdio, real APIs.

The server is started like an MCP host does it: from an unrelated working
directory with only the SDK's minimal environment, so it must find ``.env`` by
its own path. Deselected by default; run with ``uv run pytest -m live``.
Assertions check structure and settled history, never the latest values.
No live order is ever sent: ``place_order`` is not exercised here. Tools that
need a key skip when it is absent from the ``.env``; quota-metered calls
(PolyNode X Search, The Odds API) request the minimum.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from fastmcp.exceptions import ToolError

SERVER = Path(__file__).resolve().parents[1] / "mcp_server.py"
ENV = dotenv_values(SERVER.parent / ".env")
TOOLS = {
    "get_series",
    "search_series",
    "search_markets",
    "search_events",
    "get_market_resolution_rules",
    "simulate_clob_fill",
    "scan_neg_risk_event",
    "place_order",
    "list_open_orders",
    "cancel_order",
    "price_crypto_expiry",
    "search_sec_filings",
    "get_company_submissions",
    "calculate_merger_spread_prob",
    "search_x_context",
    "search_online_context",
    "get_market_holders",
    "search_federal_register",
    "get_bill_status",
    "get_nomination_status",
    "get_candidate_financials",
    "search_court_dockets",
    "search_court_opinions",
    "get_docket_entries",
    "search_gdelt_news",
    "get_gdelt_volume",
    "get_fed_releases",
    "search_predictit",
    "get_bookmaker_odds",
    "analyze_poll_dropout",
    "record_forecast",
    "get_active_theses",
    "resolve_and_score_calibration",
}


def needs(key: str) -> pytest.MarkDecorator:
    """Skip unless ``key`` is set in the server's ``.env``."""
    return pytest.mark.skipif(not ENV.get(key), reason=f"{key} not set in .env")


pytestmark = pytest.mark.live


def call(tmp_path: Path, tool: str | None, **arguments: Any) -> Any:
    """Spawn the server, then list tools (``tool=None``) or call one."""

    async def go() -> Any:
        transport = PythonStdioTransport(SERVER, python_cmd=sys.executable, cwd=str(tmp_path), keep_alive=False)
        async with Client(transport, timeout=60) as client:
            if tool is None:
                return {t.name for t in await client.list_tools()}
            return (await client.call_tool(tool, arguments)).data

    return asyncio.run(go())


def test_server_boots_over_stdio_with_all_tools(tmp_path: Path) -> None:
    assert call(tmp_path, None) == TOOLS


def test_fred_series_and_error_passthrough(tmp_path: Path) -> None:
    text = call(
        tmp_path, "get_series", series_id="UNRATE", observation_start="2019-01-01", observation_end="2019-12-31"
    )
    assert text.splitlines()[0] == "FRED series UNRATE (12 observations)"
    with pytest.raises(ToolError, match="FRED error 400: .*series does not exist"):
        call(tmp_path, "get_series", series_id="ZZNOTASERIES9")


def test_search_markets_returns_liquid_rows(tmp_path: Path) -> None:
    lines = call(tmp_path, "search_markets", min_volume_usd=100_000, limit=5).splitlines()
    assert lines[0].startswith("market_id | question")
    assert 1 <= len(lines) - 1 <= 5
    assert all(len(row.split(" | ")) == 7 for row in lines[1:])


def test_price_crypto_expiry_prices_btc(tmp_path: Path) -> None:
    lines = call(tmp_path, "price_crypto_expiry", symbol="BTCUSDT", strike=1_000_000.0, horizon_hours=24).splitlines()
    rows = {line.split(" | ")[0]: line.split(" | ") for line in lines[2:] if " | " in line}
    assert {"empirical", "normal_7d", "normal_full"} <= set(rows)
    assert all(float(rows[m][1]) < 0.01 for m in ("empirical", "normal_7d", "normal_full"))  # $1M BTC in a day
    assert rows["empirical"][4] == "976"  # 1000 closes -> 976 overlapping 24h windows


def test_clob_fill_simulation(tmp_path: Path) -> None:
    row = call(tmp_path, "search_markets", min_volume_usd=1_000_000, limit=1).splitlines()[1]
    yes_token = row.split(" | ")[-1].split()[0]
    try:
        text = call(tmp_path, "simulate_clob_fill", token_id=yes_token, side="BUY", amount_usd=10, fair_prob=0.5)
    except ToolError as exc:
        if "Could not reach Polymarket" in str(exc):
            pytest.skip(f"CLOB unreachable from this network (geoblock?): {exc}")
        raise
    assert text.startswith("BUY $10.00 -> ")


def test_federal_register_and_fed_feed(tmp_path: Path) -> None:
    lines = call(tmp_path, "search_federal_register", term="tariff", limit=3).splitlines()
    assert lines[0].startswith("Federal Register 'tariff': 3 published")
    assert any(line.startswith("[PUBLISHED] ") for line in lines)
    releases = call(tmp_path, "get_fed_releases", limit=3).splitlines()
    assert len(releases) == 3 and all("https://www.federalreserve.gov/" in r for r in releases)


def test_predictit_and_fec_demo_key(tmp_path: Path) -> None:
    text = call(tmp_path, "search_predictit", keyword="2026")
    assert text.startswith(("PredictIt matches for '2026'", "No PredictIt markets"))
    rows = call(tmp_path, "get_candidate_financials", query="ossoff", cycle=2026).splitlines()
    assert any(row.startswith("OSSOFF") for row in rows[1:])


def test_gdelt_articles(tmp_path: Path) -> None:
    try:
        text = call(tmp_path, "search_gdelt_news", query="federal reserve", timespan="3d", limit=3)
    except ToolError as exc:
        if "429" in str(exc):
            pytest.skip(f"GDELT rate limit: {exc}")
        raise
    assert text.startswith(("GDELT 'federal reserve'", "No GDELT articles"))


def test_poll_dropout_runs_in_process_without_keys(tmp_path: Path) -> None:
    polls = [
        {"pollster": "A", "end_date": "2026-10-30", "value": 47},
        {"pollster": "B", "end_date": "2026-10-20", "value": 45},
    ]
    assert call(tmp_path, "analyze_poll_dropout", polls=polls).startswith("polls 2 | current avg 46.000")


@needs("POLYNODE_API_KEY")
def test_polynode_x_search(tmp_path: Path) -> None:
    try:
        text = call(tmp_path, "search_x_context", query="from:Polymarket", max_results=1)
    except ToolError as exc:
        if "error 502" in str(exc):  # documented as a transient PolyNode upstream failure
            pytest.skip(f"PolyNode X upstream down: {exc}")
        raise
    assert text.startswith(("PolyNode X Search", "No X posts"))


@needs("CONGRESS_API_KEY")
def test_congress_bill_status(tmp_path: Path) -> None:
    assert call(tmp_path, "get_bill_status", congress=118, bill_type="hr", bill_number=2).startswith("HR.2 (118th)")


@needs("COURTLISTENER_API_TOKEN")
def test_court_dockets(tmp_path: Path) -> None:
    assert call(tmp_path, "search_court_dockets", query="tariff", limit=2).startswith("CourtListener dockets 'tariff'")


@needs("THE_ODDS_API_KEY")
def test_bookmaker_odds(tmp_path: Path) -> None:
    text = call(tmp_path, "get_bookmaker_odds")
    assert text.startswith(("Bookmaker outrights", "No bookmaker odds"))
