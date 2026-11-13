"""Offline tests for every MCP tool.

Only constructor injection: fake settings loaders, ``httpx2.MockTransport`` for
HTTP, and a recording ``OrderSender`` for live orders. Canned bodies mirror the
real wire formats (FRED v1, Gamma, CLOB ``/book``, PolyNode, Federal Register,
Congress.gov, FEC, CourtListener, GDELT, Fed RSS, PredictIt, The Odds API;
checked against live responses or published docs on 2026-11-05).
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.exceptions import ToolError

from crypto_client import CryptoSettings, ExpiryPricer
from fred_client import FredError, FredSettings
from gov_client import GovSettings
from ledger_client import CalibrationMath, ForecastRecord, LedgerSettings
from mcp_server import create_server
from politics_intel_client import PoliticsIntelError, PoliticsIntelSettings, Poll, PollDropoutCalculator
from poly_client import CancelResult, Level, OrderBookMath, OrderResult, PolyError, PolySettings, RestingOrder
from polynode_client import PolynodeSettings
from sec_client import SecSettings

FRED = FredSettings(api_key="k" * 32)
POLY = PolySettings(max_order_usd=100.0)
POLYNODE = PolynodeSettings(api_key="pn_live_test")
GOV = GovSettings(congress_api_key="cg_test", courtlistener_token="cl_test")
INTEL = PoliticsIntelSettings(odds_api_key="odds_test")
SEC = SecSettings(user_agent="QuantTest research@example.com", using_default_user_agent=False, min_interval_seconds=0.0)

Answer = tuple[int, Any] | httpx.Response
Route = Answer | Callable[[httpx.Request], Answer]


class RecordingSender:
    """Fake ``OrderSender``: records live orders instead of signing them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, float]] = []

    async def __call__(
        self, settings: PolySettings, token_id: str, side: str, price: float, size: float
    ) -> OrderResult:
        self.calls.append((token_id, side, price, size))
        return OrderResult(True, "0xorder", "live")


class RecordingOrderManager:
    """Fake ``OrderManager``: records open-order queries and cancellations offline."""

    def __init__(self, orders: tuple[RestingOrder, ...] = ()) -> None:
        self.orders = orders
        self.list_calls: list[str | None] = []
        self.cancel_calls: list[str] = []

    async def list_open_orders(
        self, settings: PolySettings, condition_id: str | None = None
    ) -> tuple[RestingOrder, ...]:
        self.list_calls.append(condition_id)
        if condition_id is None:
            return self.orders
        return tuple(o for o in self.orders if o.condition_id == condition_id)

    async def cancel_order(self, settings: PolySettings, order_id: str) -> CancelResult:
        self.cancel_calls.append(order_id)
        return CancelResult(order_id=order_id, dry_run=False, canceled=True, status="canceled", wallet="0xdep")


def make_server(
    routes: dict[str, Route],
    seen: list[httpx.Request] | None = None,
    sender: RecordingSender | None = None,
    fred: Callable[[], FredSettings] = lambda: FRED,
    polynode: Callable[[], PolynodeSettings] = lambda: POLYNODE,
    gov: Callable[[], GovSettings] = lambda: GOV,
    intel: Callable[[], PoliticsIntelSettings] = lambda: INTEL,
    *,
    poly: Callable[[], PolySettings] = lambda: POLY,
    sec: Callable[[], SecSettings] = lambda: SEC,
    ledger: Callable[[], LedgerSettings] | None = None,
    manager: RecordingOrderManager | None = None,
) -> FastMCP:
    """Server whose HTTP is answered by ``routes`` keyed on URL path (JSON tuple or a ready Response)."""
    mem_ledger = LedgerSettings(db_path=":memory:")
    ledger_loader = ledger or (lambda: mem_ledger)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        route = routes[request.url.path]
        answer = route(request) if callable(route) else route
        if isinstance(answer, httpx.Response):
            return answer
        status, body = answer
        return httpx.Response(status, json=body)

    return create_server(
        fred,
        poly,
        lambda: CryptoSettings(),
        polynode,
        gov,
        intel,
        transport=httpx.MockTransport(handler),
        order_sender=sender,
        sec_settings_loader=sec,
        ledger_settings_loader=ledger_loader,
        order_manager=manager,
    )


def call(server: FastMCP, tool: str, **arguments: Any) -> str:
    async def go() -> str:
        async with Client(server) as client:
            return (await client.call_tool(tool, arguments)).data

    return asyncio.run(go())


def book(bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> tuple[int, dict[str, Any]]:
    """CLOB /book body; levels given worst-first, like the real wire."""
    return 200, {
        "bids": [{"price": p, "size": s} for p, s in bids],
        "asks": [{"price": p, "size": s} for p, s in asks],
        "tick_size": "0.01",
        "min_order_size": "5",
    }


# --- FRED -------------------------------------------------------------------


def test_get_series_sends_query_and_skips_missing_values() -> None:
    seen: list[httpx.Request] = []
    body = {"observations": [{"date": "2024-01-01", "value": "3.7"}, {"date": "2024-02-01", "value": "."}]}
    server = make_server({"/fred/series/observations": (200, body)}, seen)

    text = call(server, "get_series", series_id="UNRATE", observation_start="2024-01-01", units="pc1")

    assert text == "FRED series UNRATE (2 observations)\n2024-01-01: 3.7"
    params = seen[0].url.params
    assert (params["series_id"], params["observation_start"], params["units"], params["file_type"]) == (
        "UNRATE",
        "2024-01-01",
        "pc1",
        "json",
    )


def test_search_series_formats_hits() -> None:
    body = {"seriess": [{"id": "UNRATE", "title": "Unemployment Rate"}]}
    assert call(make_server({"/fred/series/search": (200, body)}), "search_series", search_text="jobs") == (
        "UNRATE: Unemployment Rate"
    )


def test_fred_rejection_and_missing_key_become_tool_errors() -> None:
    rejected = (400, {"error_code": 400, "error_message": "Bad Request.  The series does not exist."})
    with pytest.raises(ToolError, match="FRED error 400: Bad Request.  The series does not exist."):
        call(make_server({"/fred/series/observations": rejected}), "get_series", series_id="NOPE")

    def no_key() -> FredSettings:
        return FredSettings.from_env({})

    with pytest.raises(ToolError, match="FRED_API_KEY is not set"):
        call(make_server({}, fred=no_key), "search_series", search_text="gdp")
    with pytest.raises(FredError):
        no_key()


# --- search_markets ---------------------------------------------------------


def market(mid: str, question: str, bid: float | None, ask: float | None) -> dict[str, Any]:
    return {
        "id": mid,
        "question": question,
        "endDate": "2026-12-10T00:00:00Z",
        "bestBid": bid,
        "bestAsk": ask,
        "volumeNum": 250_000.4,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "clobTokenIds": '["111", "222"]',
    }


def test_search_markets_filters_noise_and_formats_rows() -> None:
    seen: list[httpx.Request] = []
    markets = [
        market("1", "Will the Fed cut rates in December?", 0.62, 0.64),
        market("2", "Will the Fed hike rates in December?", None, 0.05),  # one-sided book
        market("3", "Will the Fed pause in December?", 0.20, 0.45),  # spread > 10c
        market("4", "Will BTC hit 100k?", 0.30, 0.31),  # query mismatch
    ]
    routes: dict[str, Route] = {"/tags/slug/economy": (200, {"id": "100328"}), "/markets": (200, markets)}

    text = call(make_server(routes, seen), "search_markets", query="fed december", tag="economy", max_days_to_expiry=60)

    assert text.splitlines()[1:] == [
        "1 | Will the Fed cut rates in December? | 2026-12-10 | 0.620/0.640 | 0.360/0.380 | $250,000 | 111 222"
    ]
    params = seen[1].url.params
    assert (params["tag_id"], params["volume_num_min"], params["active"], params["closed"]) == (
        "100328",
        "10000",
        "true",
        "false",
    )
    assert "end_date_max" in params and "end_date_min" in params


def test_search_markets_unknown_tag_is_a_tool_error() -> None:
    routes: dict[str, Route] = {"/tags/slug/nope": (404, {"type": "not found error", "error": "tag not found"})}
    with pytest.raises(ToolError, match="Polymarket error 404: tag not found"):
        call(make_server(routes), "search_markets", tag="nope")


# --- simulate_clob_fill -----------------------------------------------------


def test_book_walk_vwap_slippage_and_shrunk_kelly_are_exact() -> None:
    asks = (Level(0.50, 100), Level(0.52, 100), Level(0.55, 1000))
    buy = OrderBookMath.walk(asks, "BUY", 80)  # $50 clears level 1, $30 eats into level 2
    vwap = 80 / (100 + 30 / 0.52)
    assert (buy.shares, buy.vwap, buy.post_trade_best) == (pytest.approx(100 + 30 / 0.52), pytest.approx(vwap), 0.52)
    assert buy.slippage_bps == pytest.approx((vwap - 0.50) / 0.50 * 10_000)  # 146.34 bps

    report = OrderBookMath.edge(buy, fair_prob=0.60, fair_prob_sd=0.05)
    edge = 0.60 - vwap
    assert report.edge == pytest.approx(edge)
    assert report.kelly_full == pytest.approx(edge / (1 - vwap))  # 0.1881
    assert report.kelly_shrunk == pytest.approx(edge / (1 - vwap) * edge**2 / (edge**2 + 0.05**2))  # 0.1457

    sell = OrderBookMath.walk((Level(0.48, 50), Level(0.45, 1000)), "SELL", 30)
    assert sell.vwap == pytest.approx(30 / (50 + 6 / 0.45))
    assert sell.slippage_bps == pytest.approx((0.48 - 30 / (50 + 6 / 0.45)) / 0.48 * 10_000)


def test_simulate_clob_fill_on_thin_book_reports_partial_fill() -> None:
    seen: list[httpx.Request] = []
    routes: dict[str, Route] = {"/book": book(bids=[("0.40", "10")], asks=[("0.55", "10"), ("0.50", "20")])}

    text = call(
        make_server(routes, seen), "simulate_clob_fill", token_id="111", side="BUY", amount_usd=100, fair_prob=0.7
    )

    assert seen[0].url.params["token_id"] == "111"
    fill_line, edge_line = text.splitlines()
    assert fill_line == (
        "BUY $15.50 -> 30.00 sh | vwap 0.5167 | best 0.5000 | slippage 333.3 bps | post-trade best n/a | "
        "PARTIAL $15.50 of $100.00"
    )
    assert edge_line.startswith("fair 0.7000 | edge +0.1833")


def test_simulate_clob_fill_missing_book_is_a_tool_error() -> None:
    routes: dict[str, Route] = {"/book": (404, {"error": "No orderbook exists for the requested token id"})}
    with pytest.raises(ToolError, match="Polymarket error 404: No orderbook exists"):
        call(make_server(routes), "simulate_clob_fill", token_id="9", side="SELL", amount_usd=10, fair_prob=0.5)


# --- scan_neg_risk_event ----------------------------------------------------


def test_scan_neg_risk_event_sums_vwaps_and_flags_arb() -> None:
    legs = {"y1": "0.30", "n1": "0.72", "y2": "0.30", "n2": "0.71", "y3": "0.35", "n3": "0.66"}
    event = {
        "negRisk": True,
        "markets": [
            {"groupItemTitle": f"opt{i}", "active": True, "closed": False, "clobTokenIds": f'["y{i}", "n{i}"]'}
            for i in (1, 2, 3)
        ]
        + [{"groupItemTitle": "old", "active": True, "closed": True, "clobTokenIds": '["x", "z"]'}],
    }
    routes: dict[str, Route] = {
        "/events/slug/fed-cuts": (200, event),
        "/book": lambda r: book(bids=[], asks=[(legs[r.url.params["token_id"]], "1000")]),
    }

    lines = call(make_server(routes), "scan_neg_risk_event", event_slug="fed-cuts", size_usd=50).splitlines()

    assert lines[0] == "event fed-cuts | 3 outcomes | negRisk=True"
    assert lines[1] == "opt1 | YES 0.3000 | NO 0.7200"
    assert lines[-2] == "sum(YES) 0.9500 | bundle edge +0.0500  <-- ARB: buy all"
    assert lines[-1] == "sum(NO) 2.0900 | bundle edge -0.0900"


def test_scan_neg_risk_unknown_event_is_a_tool_error() -> None:
    routes: dict[str, Route] = {"/events/slug/nope": (404, {"type": "not found error", "error": "slug not found"})}
    with pytest.raises(ToolError, match="Polymarket error 404: slug not found"):
        call(make_server(routes), "scan_neg_risk_event", event_slug="nope")


# --- place_order ------------------------------------------------------------

ORDER_ROUTES: dict[str, Route] = {
    "/markets/123": (200, {"clobTokenIds": '["111", "222"]', "acceptingOrders": True, "closed": False}),
    "/book": book(bids=[("0.38", "100")], asks=[("0.42", "100"), ("0.40", "50")]),
}
ORDER = {"market_id": "123", "token_id": "111", "side": "BUY", "amount_usd": 50, "limit_price": 0.41}


def test_place_order_dry_run_validates_without_sending() -> None:
    sender = RecordingSender()

    text = call(make_server(ORDER_ROUTES, sender=sender), "place_order", **ORDER, reasoning="fed cut mispriced")

    assert sender.calls == []
    assert text.splitlines() == [
        "BUY 121.95 sh @ 0.41 ($50.00) token 111 market 123",
        (
            "crosses now: BUY $20.00 -> 50.00 sh | vwap 0.4000 | best 0.4000 | slippage 0.0 bps | "
            "post-trade best 0.4200 | PARTIAL $20.00 of $50.00"
        ),
        "rests on book: $30.00",
        "DRY RUN: validated, not sent.",
        "reasoning: fed cut mispriced",
    ]


def test_place_order_live_sends_floored_shares_through_the_sender() -> None:
    sender = RecordingSender()

    text = call(make_server(ORDER_ROUTES, sender=sender), "place_order", **ORDER, dry_run=False)

    assert sender.calls == [("111", "BUY", 0.41, 121.95)]
    assert text.splitlines()[3] == "SENT: order 0xorder status live"


def test_place_order_guards_reject_before_sending() -> None:
    sender = RecordingSender()
    server = make_server(ORDER_ROUTES, sender=sender)

    with pytest.raises(ToolError, match="exceeds POLY_MAX_ORDER_USD 100.0"):
        call(server, "place_order", **{**ORDER, "amount_usd": 150}, dry_run=False)
    with pytest.raises(ToolError, match="does not belong to market 123"):
        call(server, "place_order", **{**ORDER, "token_id": "999"}, dry_run=False)
    assert sender.calls == []


# --- price_crypto_expiry ----------------------------------------------------


def closes_from(returns: list[float], start: float = 100.0) -> list[float]:
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * math.exp(r))
    return closes


def test_expiry_pricer_probabilities_are_exact() -> None:
    a, b = 0.01, 0.02  # older 32 hourly returns are +/-a, the last 168 are +/-b
    closes = closes_from([a, -a] * 16 + [b, -b] * 84)
    strike = closes[-1] * math.exp(0.015)  # r* = +0.015

    p = ExpiryPricer.price("BTCUSDT", closes, strike, horizon_hours=1, implied_vol=0.5)
    by_name = {e.name: e for e in p.estimates}

    assert p.r_star == pytest.approx(0.015)
    assert by_name["empirical"].p_yes == pytest.approx(84 / 200)  # only the +b hours beat r*
    sigma_7d = b * math.sqrt(168 / 167)  # zero-mean sample sd of +/-b
    sigma_full = math.sqrt((32 * a**2 + 168 * b**2) / 199)
    sigma_iv = 0.5 * math.sqrt(1 / 8760)
    for name, sigma in (("normal_7d", sigma_7d), ("normal_full", sigma_full), ("implied", sigma_iv)):
        assert by_name[name].sigma_h == pytest.approx(sigma)
        assert by_name[name].p_yes == pytest.approx(1 - statistics.NormalDist(0, sigma).cdf(0.015))
        assert by_name[name].p_no == pytest.approx(statistics.NormalDist(0, sigma).cdf(0.015))


def kline(close: float, close_ms: int) -> list[Any]:
    return [close_ms - 3_599_999, "0", "0", "0", f"{close:.8f}", "1", close_ms, "0", 1, "0", "0", "0"]


def test_price_crypto_expiry_uses_binance_and_nearest_deribit_option() -> None:
    seen: list[httpx.Request] = []
    now_ms = 1_800_000_000_000
    closes = closes_from([0.004, -0.004] * 15)
    hour = 3_600_000
    instruments = [
        {"instrument_name": "BTC-SOON-100-C", "strike": 100.0, "expiration_timestamp": now_ms + hour},  # too early
        {"instrument_name": "BTC-LATER-90-C", "strike": 90.0, "expiration_timestamp": now_ms + 30 * hour},
        {"instrument_name": "BTC-NEXT-102-C", "strike": 102.0, "expiration_timestamp": now_ms + 25 * hour},
        {"instrument_name": "BTC-NEXT-120-C", "strike": 120.0, "expiration_timestamp": now_ms + 25 * hour},
    ]
    routes: dict[str, Route] = {
        "/api/v3/klines": (200, [kline(c, now_ms - (len(closes) - 1 - i) * hour) for i, c in enumerate(closes)]),
        "/api/v2/public/get_instruments": (200, {"result": instruments}),
        "/api/v2/public/get_order_book": (200, {"result": {"mark_iv": 50.0}}),
    }

    text = call(
        make_server(routes, seen),
        "price_crypto_expiry",
        symbol="BTCUSDT",
        strike=101.0,
        horizon_hours=24,
        fill_price_no=0.4,
    )

    assert dict(seen[0].url.params) == {"symbol": "BTCUSDT", "interval": "1h", "limit": "1000"}
    assert seen[1].url.params["currency"] == "BTC"
    assert seen[2].url.params["instrument_name"] == "BTC-NEXT-102-C"  # first expiry after 24h, nearest strike
    lines = text.splitlines()
    assert lines[0] == "BTCUSDT spot 100.00 | strike 101.00 | r* +0.00995 | horizon 24h"
    assert [line.split(" | ")[0] for line in lines[2:6]] == ["empirical", "normal_7d", "normal_full", "implied"]
    assert lines[2] == "empirical | 0.0000 | 1.0000 | n/a | 7 | +0.6000"  # 24h windows net to 0 < r*
    assert lines[-1] == (
        "implied vol: Deribit BTC-NEXT-102-C mark_iv=50.0% "
        "(expiry +1.0h vs target; skew +0.00 iv_pts/10%K, BL digital adj +0.0000)"
    )


def test_crypto_errors_binance_fails_the_tool_deribit_only_degrades() -> None:
    bad_symbol: dict[str, Route] = {"/api/v3/klines": (400, {"code": -1121, "msg": "Invalid symbol."})}
    with pytest.raises(ToolError, match="HTTP 400 .*: Invalid symbol."):
        call(make_server(bad_symbol), "price_crypto_expiry", symbol="NOPEUSDT", strike=1.0, horizon_hours=2)

    deribit_down: dict[str, Route] = {
        "/api/v3/klines": (200, [kline(c, 1_000 + i) for i, c in enumerate(closes_from([0.01, -0.01] * 5))]),
        "/api/v2/public/get_instruments": (400, {"error": {"message": "Invalid params", "data": {"reason": "bad"}}}),
    }
    text = call(make_server(deribit_down), "price_crypto_expiry", symbol="BTCUSDT", strike=100.0, horizon_hours=2)
    assert text.splitlines()[-1].startswith("implied vol: implied vol unavailable (HTTP 400 ")
    assert "implied |" not in text


# --- PolyNode ---------------------------------------------------------------


def test_search_x_context_sends_operators_and_formats_posts() -> None:
    seen: list[httpx.Request] = []
    q = "from:AtlasIntel since:2026-11-01 until:2026-11-05"
    tweet = {
        "id": "1",
        "text": "New national poll:\nApprove 48%, Disapprove 50%.",
        "created_at": "Thu Nov 05 12:00:00 +0000 2026",
        "public_metrics": {"like_count": 420, "retweet_count": 37, "impression_count": "9000"},
        "author": {"id": "9", "name": "AtlasIntel", "username": "AtlasIntel"},
    }
    routes: dict[str, Route] = {"/v2/x/search": (200, {"query": q, "tweets": [tweet], "result_count": 1})}

    text = call(
        make_server(routes, seen),
        "search_x_context",
        query="from:AtlasIntel",
        max_results=5,
        since="2026-11-01",
        until="2026-11-05",
    )

    assert (seen[0].url.params["q"], seen[0].url.params["max"]) == (q, "5")
    assert seen[0].headers["x-api-key"] == "pn_live_test"
    assert text.splitlines() == [
        f"PolyNode X Search '{q}' (1 posts)",
        "[Thu Nov 05 12:00:00 +0000 2026] @AtlasIntel (likes=420, rt=37): New national poll: Approve 48%, Disapprove 50%.",
    ]


def test_polynode_missing_key_and_free_tier_become_tool_errors() -> None:
    with pytest.raises(ToolError, match="POLYNODE_API_KEY is not set"):
        call(make_server({}, polynode=lambda: PolynodeSettings.from_env({})), "search_x_context", query="x")
    routes: dict[str, Route] = {"/v2/x/search": (402, {})}
    with pytest.raises(ToolError, match="PolyNode error 402 on /v2/x/search: this endpoint needs a paid PolyNode plan"):
        call(make_server(routes), "search_x_context", query="x")


def test_search_online_context_flags_partial_results() -> None:
    seen: list[httpx.Request] = []
    body = {
        "status": "partial",
        "unresponsive_engines": ["bing"],
        "results": [
            {
                "title": "Senate vote  delayed",
                "url": "https://example.com/a",
                "domain": "example.com",
                "snippet": "The vote\nslipped to Friday.",
                "published_at": None,
            }
        ],
    }

    text = call(
        make_server({"/v2/search/online": (200, body)}, seen),
        "search_online_context",
        query="senate vote",
        max_results=3,
        time_range="week",
    )

    assert dict(seen[0].url.params) == {"q": "senate vote", "max_results": "3", "time_range": "week"}
    assert text.splitlines() == [
        "PolyNode Online Search 'senate vote' (1 results, status partial) | unresponsive engines: bing",
        "[-] example.com | Senate vote delayed | https://example.com/a",
        "  The vote slipped to Friday.",
    ]


def test_get_market_holders_scales_raw_units_and_measures_concentration() -> None:
    seen: list[httpx.Request] = []
    yes, no = "1000000000000000000001", "2000000000000000000002"

    def row(user: str, token: str, amount: int, avg: int, pnl: int = 0) -> dict[str, Any]:
        return {"user": user, "token_id": token, "amount": str(amount), "avg_price": str(avg), "realized_pnl": str(pnl)}

    rows = [
        row("0xa", yes, 6_000_000_000, 400_000),
        row("0xb", yes, 3_000_000_000, 500_000),
        row("0xd", no, 2_000_000_000, 600_000, -1_500_000),
        row("0xc", yes, 1_000_000_000, 0),  # unknown entry price: counts for size, not for avg entry
        row("0xe", yes, 0, 450_000),  # exited: ignored
    ]
    routes: dict[str, Route] = {"/v3/markets/slug/fed-dec/positions": (200, {"data": rows, "has_more": True})}

    text = call(make_server(routes, seen), "get_market_holders", market_slug="fed-dec", limit=5, top_n=2)

    assert dict(seen[0].url.params) == {"limit": "5", "sort": "amount", "status": "open"}
    assert text.splitlines() == [
        "market fed-dec | 5 rows (more exist; stats cover fetched rows)",
        # top2 = 9000/10000, HHI = .6^2 + .3^2 + .1^2, avg entry = (6000*.4 + 3000*.5) / 9000
        "token 00000001 | holders 3 | shares 10,000 | top2 90.0% | HHI 0.460 | avg entry 0.4333",
        "token 00000002 | holders 1 | shares 2,000 | top2 100.0% | HHI 1.000 | avg entry 0.6000",
        "wallet | token | shares | avg_price | realized_pnl",
        "0xa | 00000001 | 6,000 | 0.4000 | $0.00",
        "0xb | 00000001 | 3,000 | 0.5000 | $0.00",
    ]


# --- Government and legal ---------------------------------------------------


def test_search_federal_register_reports_public_inspection_failure() -> None:
    seen: list[httpx.Request] = []
    doc = {
        "title": "Adjusting Imports of Steel",
        "type": "Presidential Document",
        "publication_date": "2026-11-05",
        "document_number": "2026-22631",
    }
    routes: dict[str, Route] = {
        "/api/v1/documents.json": (200, {"count": 1, "results": [doc]}),
        "/api/v1/public-inspection-documents.json": (503, {"message": "Service Unavailable"}),
    }

    text = call(make_server(routes, seen), "search_federal_register", term="tariff", limit=2)

    assert (seen[0].url.params["conditions[term]"], seen[0].url.params["per_page"]) == ("tariff", "2")
    assert text.splitlines() == [
        "Federal Register 'tariff': 1 published, 0 on public inspection",
        "public inspection unavailable: Federal Register error 503: Service Unavailable",
        "[PUBLISHED] 2026-11-05: Adjusting Imports of Steel (type=Presidential Document, doc=2026-22631)",
    ]


def test_get_bill_status_formats_latest_action_and_needs_key() -> None:
    seen: list[httpx.Request] = []
    bill = {
        "type": "HR",
        "number": "1",
        "congress": 119,
        "title": "One Big Beautiful Bill Act",
        "introducedDate": "2025-05-20",
        "originChamber": "House",
        "sponsors": [{"fullName": "Rep. Arrington, Jodey C. [R-TX-19]"}],
        "policyArea": {"name": "Economics and Public Finance"},
        "latestAction": {"actionDate": "2025-07-04", "text": "Became Public Law No: 119-21."},
    }

    text = call(
        make_server({"/v3/bill/119/hr/1": (200, {"bill": bill})}, seen),
        "get_bill_status",
        congress=119,
        bill_type="hr",
        bill_number=1,
    )

    assert (seen[0].url.params["api_key"], seen[0].url.params["format"]) == ("cg_test", "json")
    assert text.splitlines()[-1] == "latest action 2025-07-04: Became Public Law No: 119-21."
    with pytest.raises(ToolError, match="CONGRESS_API_KEY is not set"):
        call(make_server({}, gov=lambda: GovSettings()), "get_bill_status", congress=119, bill_type="s", bill_number=5)


def test_get_nomination_status_formats_nomination() -> None:
    nomination = {
        "citation": "PN12",
        "congress": 119,
        "description": "Jane Doe, of Ohio, to be a Member of the Federal Reserve Board.",
        "receivedDate": "2025-01-20",
        "organization": "Federal Reserve System",
        "latestAction": {"actionDate": "2025-03-01", "text": "Confirmed by the Senate by Yea-Nay Vote. 52 - 47."},
    }
    routes: dict[str, Route] = {"/v3/nomination/119/12": (200, {"nomination": nomination})}

    assert call(make_server(routes), "get_nomination_status", congress=119, nomination_number=12).splitlines() == [
        "PN12 (119th) Jane Doe, of Ohio, to be a Member of the Federal Reserve Board.",
        "received 2025-01-20 | organization Federal Reserve System",
        "latest action 2025-03-01: Confirmed by the Senate by Yea-Nay Vote. 52 - 47.",
    ]


def test_get_candidate_financials_resorts_by_cash_on_hand() -> None:
    seen: list[httpx.Request] = []

    def cand(name: str, receipts: float, spent: float, cash: str) -> dict[str, Any]:
        return {
            "name": name,
            "candidate_id": name[:3],
            "party": "DEM",
            "office": "S",
            "state": "GA",
            "receipts": receipts,
            "disbursements": spent,
            "cash_on_hand_end_period": cash,
            "debts_owed_by_committee": "0.00",
            "coverage_end_date": "2026-10-14",
        }

    # API order is by receipts; the second candidate has more cash left.
    body = {
        "results": [cand("BIG SPENDER", 1_000_000, 950_000, "50000.00"), cand("SAVER", 400_000, 100_000, "300000.00")]
    }

    text = call(make_server({"/v1/candidates/totals/": (200, body)}, seen), "get_candidate_financials", query="ga")

    params = seen[0].url.params
    assert (params["api_key"], params["sort"], params["election_year"]) == ("DEMO_KEY", "-receipts", "2026")
    assert (params["q"], params["per_page"]) == ("ga", "10")
    assert text.splitlines()[1:] == [
        "SAVER | SAV | DEM S-GA | $400,000 | $100,000 | 25% | $300,000 | $0 | 2026-10-14",
        "BIG SPENDER | BIG | DEM S-GA | $1,000,000 | $950,000 | 95% | $50,000 | $0 | 2026-10-14",
    ]


def test_fec_rejection_becomes_tool_error() -> None:
    routes: dict[str, Route] = {"/v1/candidates/totals/": (422, {"message": "Cannot sort on value", "status": 422})}
    with pytest.raises(ToolError, match="FEC error 422: Cannot sort on value"):
        call(make_server(routes), "get_candidate_financials", query="ossoff")


def test_search_court_dockets_sends_token_and_needs_one() -> None:
    seen: list[httpx.Request] = []
    dockets = [
        {
            "caseName": "V.O.S. Selections v. Trump",
            "court": "Court of International Trade",
            "docketNumber": "25-00066",
            "dateFiled": "2025-04-14",
            "docket_absolute_url": "/docket/1/vos-selections-v-trump/",
        },
        {"caseName": "Other", "court": "D.D.C.", "docketNumber": "1", "dateFiled": "2025-01-01"},
    ]
    routes: dict[str, Route] = {"/api/rest/v4/search/": (200, {"count": 2, "results": dockets})}

    text = call(make_server(routes, seen), "search_court_dockets", query="tariff", limit=1)

    assert seen[0].headers["authorization"] == "Token cl_test"
    assert (seen[0].url.params["type"], seen[0].url.params["order_by"]) == ("r", "dateFiled desc")
    assert text.splitlines() == [
        "CourtListener dockets 'tariff' (2 total, showing 1)",
        (
            "2025-04-14 | Court of International Trade | 25-00066 | V.O.S. Selections v. Trump | "
            "https://www.courtlistener.com/docket/1/vos-selections-v-trump/"
        ),
    ]
    with pytest.raises(ToolError, match="COURTLISTENER_API_TOKEN is not set"):
        call(make_server({}, gov=lambda: GovSettings()), "search_court_dockets", query="x")


# --- GDELT, Fed RSS, PredictIt, The Odds API, polling -----------------------

GDELT = "/api/v2/doc/doc"


def test_search_gdelt_news_formats_articles_and_surfaces_plain_text_errors() -> None:
    seen: list[httpx.Request] = []
    article = {
        "url": "https://ex.com/a",
        "title": "Tariff talks resume",
        "seendate": "20261105T120000Z",
        "domain": "ex.com",
        "sourcecountry": "United States",
    }

    text = call(
        make_server({GDELT: (200, {"articles": [article]})}, seen),
        "search_gdelt_news",
        query="tariff",
        timespan="6h",
        limit=2,
    )

    params = seen[0].url.params
    assert (params["mode"], params["timespan"], params["maxrecords"], params["sort"]) == (
        "artlist",
        "6h",
        "2",
        "DateDesc",
    )
    assert text.splitlines()[1] == "20261105T120000Z | United States ex.com | Tariff talks resume | https://ex.com/a"

    throttled = httpx.Response(429, text="Please limit requests to one every 5 seconds or contact kalev.")
    with pytest.raises(ToolError, match="GDELT error 429: Please limit requests to one every 5 seconds"):
        call(make_server({GDELT: throttled}), "search_gdelt_news", query="tariff")
    bad_query = httpx.Response(200, text="Your search contained a keyword that was too short.")
    with pytest.raises(ToolError, match="GDELT: Your search contained a keyword that was too short"):
        call(make_server({GDELT: bad_query}), "search_gdelt_news", query="a")


def test_get_gdelt_volume_summarizes_timeline() -> None:
    points = [
        {"date": "20261105T000000Z", "value": 0.1},
        {"date": "20261105T001500Z", "value": 0.4},
        {"date": "20261105T003000Z", "value": 0.25},
    ]
    body = {"timeline": [{"series": "Volume Intensity", "data": points}]}

    lines = call(make_server({GDELT: (200, body)}), "get_gdelt_volume", query="shutdown").splitlines()

    assert lines[0] == (
        "GDELT volume 'shutdown' (3 points, % of monitored coverage) | latest 0.2500 | mean 0.2500 | "
        "peak 0.4000 at 20261105T001500Z"
    )
    assert lines[1:] == ["20261105T000000Z: 0.1000", "20261105T001500Z: 0.4000", "20261105T003000Z: 0.2500"]


def test_get_fed_releases_parses_bom_prefixed_feed_and_rejects_bad_xml() -> None:
    feed = (
        b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8" ?><rss version="2.0"><channel><title>FRB</title>'
        b"<item><title> Federal Reserve issues FOMC statement </title>"
        b"<link><![CDATA[https://www.federalreserve.gov/a.htm]]></link><pubDate>Wed, 28 Oct 2026 18:00:00 GMT</pubDate></item>"
        b"<item><title>Second</title><link>https://b</link><pubDate>Tue</pubDate></item></channel></rss>"
    )
    route: dict[str, Route] = {"/feeds/press_all.xml": httpx.Response(200, content=feed)}

    assert call(make_server(route), "get_fed_releases", limit=1) == (
        "Wed, 28 Oct 2026 18:00:00 GMT | Federal Reserve issues FOMC statement | https://www.federalreserve.gov/a.htm"
    )
    broken: dict[str, Route] = {"/feeds/press_all.xml": httpx.Response(200, content=b"<rss><channel><item>")}
    with pytest.raises(ToolError, match="Fed RSS returned malformed XML"):
        call(make_server(broken), "get_fed_releases")


def test_search_predictit_matches_every_keyword_word() -> None:
    markets = [
        {
            "id": 7589,
            "name": "How many House seats will Republicans win in 2026?",
            "shortName": "GOP House seats",
            "status": "Open",
            "contracts": [{"name": "218-222", "lastTradePrice": 0.21, "bestBuyYesCost": 0.22, "bestBuyNoCost": 0.8}],
        },
        {"id": 1, "name": "Senate control 2026", "shortName": "Senate", "status": "Open", "contracts": []},
    ]

    text = call(
        make_server({"/api/marketdata/all/": (200, {"markets": markets})}), "search_predictit", keyword="house SEATS"
    )

    assert text.splitlines() == [
        "PredictIt matches for 'house SEATS' (1 markets):",
        "Market 7589: How many House seats will Republicans win in 2026? [Open]",
        "  - 218-222: last=0.21 buyYes=0.22 buyNo=0.80",
    ]


def test_get_bookmaker_odds_removes_vig_per_book_then_averages() -> None:
    seen: list[httpx.Request] = []

    def book_quotes(key: str, a: float, b: float) -> dict[str, Any]:
        outcomes = [{"name": "Candidate A", "price": a}, {"name": "Candidate B", "price": b}]
        return {"key": key, "markets": [{"key": "outrights", "outcomes": outcomes}]}

    event = {
        "sport_title": "US Presidential Elections Winner",
        "commence_time": "2028-11-07T00:00:00Z",
        "bookmakers": [book_quotes("fair", 2.0, 2.0), book_quotes("vig", 1.6, 2.5)],
    }
    path = "/v4/sports/politics_us_presidential_election_winner/odds"

    text = call(make_server({path: (200, [event])}, seen), "get_bookmaker_odds")

    params = seen[0].url.params
    assert (params["apiKey"], params["markets"], params["oddsFormat"]) == ("odds_test", "outrights", "decimal")
    # Book "vig": 1/1.6 + 1/2.5 = 1.025 overround -> A 0.6098, B 0.3902; averaged with the fair book's 0.5/0.5.
    assert text.splitlines()[2:] == [
        "  Candidate A: fair 0.5549 | best 2.00 | books 2",
        "  Candidate B: fair 0.4451 | best 2.50 | books 2",
    ]
    with pytest.raises(ToolError, match="THE_ODDS_API_KEY is not set"):
        call(make_server({}, intel=lambda: PoliticsIntelSettings()), "get_bookmaker_odds")


def test_analyze_poll_dropout_orders_by_date_not_input_order() -> None:
    polls = [
        {"pollster": "A", "end_date": "2026-10-30", "value": 47.0},
        {"pollster": "B", "end_date": "2026-10-20", "value": 44.0},  # oldest, given second
        {"pollster": "C", "end_date": "2026-10-28", "value": 46.0},
        {"pollster": "D", "end_date": "2026-10-25", "value": 45.0},
    ]

    assert call(make_server({}), "analyze_poll_dropout", polls=polls).splitlines() == [
        "polls 4 | current avg 45.500 | without oldest 46.000 | shift if dropped +0.500",
        "oldest: B 2026-10-20 value 44",
    ]


def test_poll_dropout_flags_tied_oldest_and_rejects_short_input() -> None:
    tied = [Poll("A", date(2026, 10, 30), 50.0), Poll("B", date(2026, 10, 1), 40.0), Poll("C", date(2026, 10, 1), 42.0)]
    analysis = PollDropoutCalculator.analyze(tied)
    assert (analysis.oldest.pollster, analysis.oldest_ties) == ("C", 1)  # stable sort: last of the tied pair drops
    with pytest.raises(PoliticsIntelError, match="At least 2 polls"):
        PollDropoutCalculator.analyze(tied[:1])
    with pytest.raises(PoliticsIntelError, match="finite"):  # NaN cannot arrive over MCP JSON; guard anyway
        PollDropoutCalculator.analyze([*tied[:1], Poll("D", date(2026, 10, 2), float("nan"))])


def test_search_federal_register_lists_pending_documents_first() -> None:
    pending = [
        {
            "title": f"Tariff proclamation {i}",
            "filed_at": "2026-11-05T08:45:00.000-05:00",
            "publication_date": "2026-11-06",
            "document_number": f"2026-2272{i}",
        }
        for i in (1, 2, 3)
    ]
    routes: dict[str, Route] = {
        "/api/v1/documents.json": (200, {"results": []}),
        "/api/v1/public-inspection-documents.json": (200, {"count": 3, "results": pending}),
    }

    lines = call(make_server(routes), "search_federal_register", term="tariff", limit=2).splitlines()

    assert lines == [
        "Federal Register 'tariff': 0 published, 2 on public inspection",
        "[PUBLIC INSPECTION] 2026-11-05T08:45:00.000-05:00: Tariff proclamation 1 (doc=2026-22721, pub 2026-11-06)",
        "[PUBLIC INSPECTION] 2026-11-05T08:45:00.000-05:00: Tariff proclamation 2 (doc=2026-22722, pub 2026-11-06)",
    ]


def test_get_bookmaker_odds_skips_books_that_cannot_be_devigged() -> None:
    def quotes(prices: dict[str, float]) -> dict[str, Any]:
        outcomes = [{"name": n, "price": p} for n, p in prices.items()]
        return {"key": "b", "markets": [{"key": "outrights", "outcomes": outcomes}]}

    # Book 1: 1/2 + 1/3 + 1/6 = 1, no vig. Book 2 quotes only A (sum 0.476 < 1); normalizing it would claim P(A)=1.
    # Book 3 omits longshot C but sums to 1/1.8 + 1/2.4 = 0.972... < 1, so it is skipped too.
    # Book 4 omits C with a real overround: 1/1.6 + 1/2.5 = 1.025 -> A 0.6098, B 0.3902.
    books = [
        quotes({"A": 2.0, "B": 3.0, "C": 6.0}),
        quotes({"A": 2.1}),
        quotes({"A": 1.8, "B": 2.4}),
        quotes({"A": 1.6, "B": 2.5}),
    ]
    event = {"sport_title": "Winner", "bookmakers": books}
    path = "/v4/sports/politics_us_presidential_election_winner/odds"

    lines = call(make_server({path: (200, [event])}), "get_bookmaker_odds").splitlines()

    assert lines[1:] == [
        "Winner | no-vig from 2 book(s), 2 skipped (quotes sum < 100%)",
        "  A: fair 0.5549 | best 2.10 | books 4",  # (0.5 + 0.6098) / 2
        "  B: fair 0.3618 | best 3.00 | books 3",  # (0.3333 + 0.3902) / 2
        "  C: fair 0.1667 | best 6.00 | books 1",
    ]
    malformed = {"bookmakers": [{"markets": [{"key": "outrights", "outcomes": [{"price": 2.0}]}]}]}
    with pytest.raises(ToolError, match="The Odds API returned malformed odds"):
        call(make_server({path: (200, [malformed])}), "get_bookmaker_odds")


def test_transport_failure_never_echoes_the_keyed_url() -> None:
    def refuse(request: httpx.Request) -> Answer:
        raise httpx.ConnectError(f"connection refused for {request.url}")  # the URL carries api_key=cg_test

    with pytest.raises(ToolError) as caught:
        call(make_server({"/v3/bill/119/hr/1": refuse}), "get_bill_status", congress=119, bill_type="hr", bill_number=1)
    assert str(caught.value) == "Could not reach Congress.gov (ConnectError)."
    assert "cg_test" not in str(caught.value)


def test_malformed_or_failed_payloads_become_tool_errors() -> None:
    cases: list[tuple[str, dict[str, Route], dict[str, Any], str]] = [
        (
            "search_online_context",
            {"/v2/search/online": (429, {})},
            {"query": "x"},
            "PolyNode error 429 on /v2/search/online: rate limited",
        ),
        (
            "get_market_holders",
            {"/v3/markets/slug/m/positions": (200, {"data": [{"user": "0xa", "amount": "x1"}]})},
            {"market_slug": "m"},
            "Malformed PolyNode position row",
        ),
        (
            "get_nomination_status",
            {"/v3/nomination/119/7": (200, {"request": {}})},
            {"congress": 119, "nomination_number": 7},
            r"Congress.gov returned no nomination for PN7 \(Congress 119\)",
        ),
        (
            "get_gdelt_volume",
            {GDELT: (200, {"timeline": [{"data": [{"date": "d"}]}]})},
            {"query": "x"},
            "GDELT returned a malformed timeline",
        ),
        (
            "search_predictit",
            {"/api/marketdata/all/": (503, {"message": "Service Unavailable"})},
            {"keyword": "x"},
            "PredictIt error 503: Service Unavailable",
        ),
    ]
    for tool, routes, arguments, message in cases:
        with pytest.raises(ToolError, match=message):
            call(make_server(routes), tool, **arguments)


# --- Slow-Horizon Discovery, Settlement Rules, Horizon Carry & Order Lifecycle ---


def test_search_markets_paginates_past_top_200_and_checks_horizon_window() -> None:
    seen: list[httpx.Request] = []
    page0 = [market(str(i), f"Short-term noise {i}", 0.50, 0.52) for i in range(200)]
    page1 = [market("205", "Will FTC block Acme merger in Q4?", 0.41, 0.44)]

    def gamma_markets(req: httpx.Request) -> Answer:
        return (200, page1 if req.url.params.get("offset") == "200" else page0)

    text = call(
        make_server({"/markets": gamma_markets}, seen),
        "search_markets",
        query="acme merger",
        min_days_to_expiry=10,
        max_days_to_expiry=180,
        order="volumeNum",
        limit=5,
    )

    assert len(seen) == 2
    assert (seen[0].url.params["order"], seen[1].url.params["offset"]) == ("volumeNum", "200")
    assert text.splitlines()[1].startswith("205 | Will FTC block Acme merger in Q4? |")
    with pytest.raises(ToolError, match="min_days_to_expiry must be strictly less than max_days_to_expiry"):
        call(make_server({}), "search_markets", min_days_to_expiry=60, max_days_to_expiry=30)


def test_search_events_uses_public_search_for_queries_and_events_for_catalog() -> None:
    seen: list[httpx.Request] = []
    ev = {
        "id": "901",
        "slug": "acme-merger-2026",
        "title": "Acme / WidgetCo Antitrust Outcome",
        "endDate": "2027-06-30T00:00:00Z",
        "negRisk": True,
        "volume": 420_000,
        "closed": False,
        "markets": [
            {"id": "55", "groupItemTitle": "Approved", "active": True, "closed": False, "bestBid": 0.61, "bestAsk": 0.64}
        ],
    }
    routes: dict[str, Route] = {"/public-search": (200, {"events": [ev]}), "/events": (200, [ev])}

    q_text = call(make_server(routes, seen), "search_events", query="antitrust", tag="business", min_volume_usd=10_000)
    cat_text = call(make_server(routes, seen), "search_events", min_volume_usd=10_000, limit=5)

    assert (seen[0].url.path, seen[0].url.params["q"], seen[0].url.params["events_tag"]) == (
        "/public-search",
        "antitrust",
        "business",
    )
    assert (seen[1].url.path, seen[1].url.params["order"]) == ("/events", "volume")
    assert q_text.splitlines()[1] == (
        "901 | acme-merger-2026 | Acme / WidgetCo Antitrust Outcome | 2027-06-30 | True | $420,000 | 1 | "
        "55:Approved (0.61/0.64)"
    )
    assert cat_text == q_text
    with pytest.raises(ToolError, match="Polymarket returned malformed events payload"):
        call(make_server({"/public-search": (200, {"events": "bad"})}), "search_events", query="x")


def test_get_market_resolution_rules_fetches_parent_event_and_flags_ambiguities() -> None:
    mkt = {
        "id": "777",
        "slug": "will-tariff-deal-hit-100-2026",
        "question": "Will the tariff merger deal hit $100 by Dec 31?",
        "description": (
            "Resolves YES if announced and enacted before 11:59 PM ET on Dec 31 according to a consensus of credible "
            "reporting; otherwise resolves 50-50 (p3) if ambiguous. Includes Other outcomes."
        ),
        "resolutionSource": "",
        "endDate": "2026-12-31T23:59:59Z",
        "negRisk": True,
        "negRiskMarketID": "0xneg",
        "conditionId": "0x" + "c" * 64,
        "questionID": "0x" + "d" * 64,
        "umaResolutionStatuses": '["proposed"]',
        "orderPriceMinTickSize": "0.01",
        "orderMinSize": "5",
        "clobTokenIds": '["t_yes", "t_no"]',
        "events": [{"slug": "tariff-deals-2026"}],
    }
    ev = {"slug": "tariff-deals-2026", "title": "2026 Tariff Deals", "description": "Parent event level rules."}
    routes: dict[str, Route] = {
        "/markets/slug/will-tariff-deal-hit-100-2026": (200, mkt),
        "/events/slug/tariff-deals-2026": (200, ev),
    }

    text = call(make_server(routes), "get_market_resolution_rules", market_id_or_slug="will-tariff-deal-hit-100-2026")

    assert "market 777 | slug will-tariff-deal-hit-100-2026 | end 2026-12-31T23:59:59Z | negRisk=True (negRiskMarketID=0xneg)" in text
    assert "tick_size: 0.01 | min_order_size: 5 | uma_statuses: proposed | source: UNSPECIFIED" in text
    assert "parent_event: tariff-deals-2026 — 2026 Tariff Deals" in text
    for flag in (
        "TIMEZONE_CUTOFF",
        "ANNOUNCED_VS_EFFECTIVE",
        "TOUCH_VS_CLOSE",
        "UNNAMED_CONSENSUS_SOURCE",
        "P3_50_50_FALLBACK",
        "NEG_RISK_OTHER_TRAP",
    ):
        assert flag in text
    with pytest.raises(ToolError, match="Polymarket error 404"):
        call(make_server({"/markets/999": (404, {"error": "not found"})}), "get_market_resolution_rules", market_id_or_slug="999")


def test_horizon_aware_kelly_carry_and_annualized_growth_are_exact() -> None:
    fill = OrderBookMath.walk((Level(0.50, 200),), "BUY", 50.0)
    report = OrderBookMath.edge(fill, fair_prob=0.60, fair_prob_sd=0.05, days_to_resolution=73.0)
    # edge = +0.10, shrunk = 0.20 * (0.01 / 0.0125) = 0.16, 365 / 73 = 5.0
    expected_log_g = 5.0 * (0.60 * math.log(1.0 + 0.16) + 0.40 * math.log(1.0 - 0.16))
    assert report.edge_per_day_bps == pytest.approx(1000.0 / 73.0)
    assert report.annualized_return == pytest.approx(1.0)
    assert report.annualized_log_growth == pytest.approx(expected_log_g)
    with pytest.raises(PolyError, match="days_to_resolution must be a positive finite number"):
        OrderBookMath.edge(fill, fair_prob=0.60, fair_prob_sd=0.05, days_to_resolution=0.0)

    routes: dict[str, Route] = {"/book": book(bids=[("0.48", "100")], asks=[("0.50", "200")])}
    lines = call(
        make_server(routes),
        "simulate_clob_fill",
        token_id="111",
        side="BUY",
        amount_usd=50.0,
        fair_prob=0.60,
        fair_prob_sd=0.05,
        days_to_resolution=73.0,
    ).splitlines()
    assert lines[2] == f"horizon 73d | carry +13.70 bps/d | ann_ret +100.00% | ann_log_growth {expected_log_g:+.2%}"


def test_list_open_orders_and_cancel_order_respect_dry_run_and_key_guard() -> None:
    cid = "0x" + "a" * 64
    resting = RestingOrder(
        order_id="0xord1",
        condition_id=cid,
        token_id="111",
        outcome="Yes",
        side="BUY",
        price=0.40,
        original_size=100.0,
        size_matched=25.0,
        status="LIVE",
        created_at="2026-11-10T00:00:00+00:00",
    )
    mgr = RecordingOrderManager((resting,))
    routes: dict[str, Route] = {"/markets/123": (200, {"conditionId": cid})}
    srv = make_server(routes, manager=mgr)

    listed = call(srv, "list_open_orders", market_id="123")
    assert mgr.list_calls == [cid]
    assert f"0xord1 | BUY | Yes | 0.4000 | 100.00 | 25.00 | $30.00 | LIVE | 111 | {cid}" in listed

    dry = call(srv, "cancel_order", order_id="0xord1")  # dry_run defaults to True
    assert dry.startswith("DRY RUN cancel order 0xord1:")
    assert mgr.cancel_calls == []

    live = call(srv, "cancel_order", order_id="0xord1", dry_run=False)
    assert live == "CANCELED: order 0xord1 | status canceled | deposit wallet 0xdep (sig type 3)"
    assert mgr.cancel_calls == ["0xord1"]

    # Default SdkOrderManager refuses live account calls without POLY_PRIVATE_KEY, while dry_run=True still works.
    keyless_srv = make_server(routes)
    assert call(keyless_srv, "cancel_order", order_id="0xord1").startswith("DRY RUN cancel order 0xord1:")
    with pytest.raises(ToolError, match="POLY_PRIVATE_KEY is required"):
        call(keyless_srv, "list_open_orders")
    with pytest.raises(ToolError, match="POLY_PRIVATE_KEY is required"):
        call(keyless_srv, "cancel_order", order_id="0xord1", dry_run=False)


# --- Long-Dated Crypto (365d) & One-Touch Barrier Pricing -------------------


def test_expiry_pricer_one_touch_reflection_and_pathwise_high_low_math() -> None:
    # 60 daily bars oscillating +/-2% (zero drift), so every 30-day terminal return is 0,
    # while intra-window highs spike +6% above close and lows dip -6% below close.
    closes = closes_from([0.02, -0.02] * 30, start=100.0)
    highs = [c * math.exp(0.06) for c in closes]
    lows = [c * math.exp(-0.06) for c in closes]
    up_strike = closes[-1] * math.exp(0.04)  # r* = +0.04: terminal 0% never hits, pathwise +6% always hits

    exp_up = ExpiryPricer.price(
        "BTCUSDT", closes, up_strike, horizon_hours=720, implied_vol=0.60, contract_type="expiry", bar_hours=24, highs=highs, lows=lows
    )
    touch_up = ExpiryPricer.price(
        "BTCUSDT", closes, up_strike, horizon_hours=720, implied_vol=0.60, contract_type="touch", bar_hours=24, highs=highs, lows=lows
    )
    exp_by = {e.name: e for e in exp_up.estimates}
    touch_by = {e.name: e for e in touch_up.estimates}

    assert (exp_by["empirical"].p_yes, touch_by["empirical"].p_yes) == (0.0, 1.0)
    for name in ("normal_30d", "normal_full", "implied"):
        sigma = touch_by[name].sigma_h
        assert sigma is not None
        tail = 1.0 - statistics.NormalDist(0.0, sigma).cdf(0.04)
        assert exp_by[name].p_yes == pytest.approx(tail)
        assert touch_by[name].p_yes == pytest.approx(min(1.0, 2.0 * tail))

    down_strike = closes[-1] * math.exp(-0.04)  # downside touch barrier: P(min S_t <= K)
    touch_dn = ExpiryPricer.price(
        "BTCUSDT", closes, down_strike, horizon_hours=720, contract_type="touch", bar_hours=24, highs=highs, lows=lows
    )
    assert {e.name: e for e in touch_dn.estimates}["empirical"].p_yes == 1.0


def test_price_crypto_expiry_long_horizon_uses_1d_klines_and_reports_deribit_skew() -> None:
    seen: list[httpx.Request] = []
    now_ms = 1_800_000_000_000
    day_ms = 86_400_000
    closes = closes_from([0.015, -0.015] * 25, start=100.0)
    klines = [
        [now_ms - (len(closes) - 1 - i) * day_ms, f"{c:.4f}", f"{c * 1.03:.4f}", f"{c * 0.97:.4f}", f"{c:.4f}", "1", now_ms - (len(closes) - 1 - i) * day_ms, "0", 1, "0", "0", "0"]
        for i, c in enumerate(closes)
    ]
    exp_ts = now_ms + 31 * day_ms  # +1.0d after 720h (30d) horizon
    ivs = {"BTC-EXP-100-C": 50.0, "BTC-EXP-120-C": 55.0, "BTC-EXP-140-C": 60.0}
    instruments = [
        {"instrument_name": name, "strike": float(name.split("-")[2]), "expiration_timestamp": exp_ts, "option_type": "call"}
        for name in ivs
    ]
    routes: dict[str, Route] = {
        "/api/v3/klines": (200, klines),
        "/api/v2/public/get_instruments": (200, {"result": instruments}),
        "/api/v2/public/get_order_book": lambda r: (200, {"result": {"mark_iv": ivs[r.url.params["instrument_name"]]}}),
    }

    lines = call(
        make_server(routes, seen),
        "price_crypto_expiry",
        symbol="BTCUSDT",
        strike=120.0,
        horizon_hours=720,
        contract_type="touch",
    ).splitlines()

    assert dict(seen[0].url.params) == {"symbol": "BTCUSDT", "interval": "1d", "limit": "1000"}
    assert lines[0].endswith("| horizon 720h | touch (1d)")
    assert lines[1] == "model | P(YES: high >= strike) | P(NO) | sigma_h | n"
    assert [ln.split(" | ")[0] for ln in lines[2:6]] == ["empirical", "normal_30d", "normal_full", "implied"]
    # d_iv/dK = (0.60 - 0.50) / (140 - 100) = 0.0025 -> +3.00 iv_pts per 10% strike; positive call skew lowers digital YES.
    assert lines[-1].startswith(
        "implied vol: Deribit BTC-EXP-120-C mark_iv=55.0% (expiry +1.0d vs target; skew +3.00 iv_pts/10%K, BL digital adj -"
    )


# --- SEC EDGAR Corporate Catalysts, Merger Math & Deep CourtListener --------


def test_search_sec_filings_formats_8k_items_archive_url_and_403_guidance() -> None:
    seen: list[httpx.Request] = []
    efts_body = {
        "hits": {
            "total": {"value": 1},
            "hits": [
                {
                    "_id": "0000320193-26-000010:form8k.htm",
                    "_source": {
                        "file_date": "2026-10-15",
                        "form": "8-K",
                        "ciks": ["0000320193"],
                        "display_names": ["Apple Inc. (AAPL)"],
                        "adsh": "0000320193-26-000010",
                        "items": ["1.01", "8.01"],
                        "file_description": "Merger Definitive Agreement",
                    },
                }
            ],
        }
    }

    text = call(
        make_server({"/LATEST/search-index": (200, efts_body)}, seen),
        "search_sec_filings",
        query="second request",
        forms=["8-K", "S-4"],
        start_date="2026-01-01",
        limit=5,
    )

    assert seen[0].headers["user-agent"] == "QuantTest research@example.com"
    assert (seen[0].url.params["forms"], seen[0].url.params["startdt"]) == ("8-K,S-4", "2026-01-01")
    assert "1.01 (Entry into Material Definitive Agreement), 8.01 (Other Material Events)" in text
    assert "https://www.sec.gov/Archives/edgar/data/320193/000032019326000010/form8k.htm" in text

    with pytest.raises(ToolError, match="SEC EFTS error 403 .* set SEC_USER_AGENT"):
        call(make_server({"/LATEST/search-index": (403, {})}), "search_sec_filings", query="wells notice")


def test_get_company_submissions_resolves_ticker_to_cik_and_filters_forms() -> None:
    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    subs = {
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {
            "recent": {
                "form": ["8-K/A", "4", "S-4"],
                "filingDate": ["2026-11-01", "2026-10-30", "2026-10-15"],
                "accessionNumber": ["0000320193-26-000020", "0000320193-26-000019", "0000320193-26-000018"],
                "primaryDocument": ["d8ka.htm", "form4.xml", "s4.htm"],
                "primaryDocDescription": ["Amended 8-K", "Insider Form 4", "Merger Registration"],
                "items": [["5.02"], [], []],
            }
        },
    }
    routes: dict[str, Route] = {
        "/files/company_tickers.json": (200, tickers),
        "/submissions/CIK0000320193.json": (200, subs),
    }
    default_ua_sec = lambda: SecSettings.from_env({})
    srv = make_server(routes, sec=lambda: SecSettings(min_interval_seconds=0.0, user_agent=default_ua_sec().user_agent))

    lines = call(srv, "get_company_submissions", ticker_or_cik="aapl", forms=["8-K", "S-4"], limit=5).splitlines()

    assert lines[0] == "SEC Submissions Apple Inc. (CIK 0000320193, tickers: AAPL) — 2 filings"
    assert "8-K/A | items: 5.02 (Officer/Director Departure or Appointment)" in lines[1]
    assert lines[2].startswith("2026-10-15 | S-4 | items: - | Merger Registration |")
    assert lines[-1].startswith("NOTE: SEC_USER_AGENT not set;")
    with pytest.raises(ToolError, match="Ticker 'ZZZZZ' not found in SEC company_tickers.json"):
        call(srv, "get_company_submissions", ticker_or_cik="ZZZZZ")


def test_calculate_merger_spread_prob_math_regimes_and_validation() -> None:
    srv = make_server({})
    # (90 - 60) / (100 - 60) = 0.75; 365 / 182.5 = 2.0 -> IRR = (100/90)^2 - 1 = 23.4568%
    lines = call(
        srv,
        "calculate_merger_spread_prob",
        target_price=90.0,
        deal_offer_value=100.0,
        downside_break_price=60.0,
        days_to_close=182.5,
    ).splitlines()
    assert lines == [
        "target $90.00 | offer $100.00 | break $60.00 | implied_close_prob 0.7500 (raw +0.7500)",
        "gross_spread +10.00 (+11.11%) | break_downside -33.33% | horizon 182.5d | ann_simple +22.22% | deal_IRR +23.46%",
    ]

    above = call(srv, "calculate_merger_spread_prob", target_price=105.0, deal_offer_value=100.0, downside_break_price=60.0)
    assert "implied_close_prob 1.0000 (raw +1.1250)" in above and "TRADING_ABOVE_OFFER" in above

    with pytest.raises(ToolError, match=r"deal_offer_value \(50\.0\) must exceed downside_break_price \(60\.0\)"):
        call(srv, "calculate_merger_spread_prob", target_price=55.0, deal_offer_value=50.0, downside_break_price=60.0)


def test_search_court_opinions_and_get_docket_entries_format_holdings_and_recap_urls() -> None:
    seen: list[httpx.Request] = []
    opinions = {
        "count": 1,
        "results": [
            {
                "dateFiled": "2026-09-12",
                "court": "Court of International Trade",
                "docketNumber": "25-00066",
                "citation": ["612 F. Supp. 3d 10"],
                "caseName": "V.O.S. Selections v. Trump",
                "absolute_url": "/opinion/99/vos-selections/",
                "opinions": [{"snippet": "Preliminary <mark>injunction</mark> GRANTED."}],
            }
        ],
    }
    entries = {
        "count": 1,
        "results": [
            {
                "entry_number": 48,
                "date_filed": "2026-09-10",
                "description": "ORDER granting motion to stay enforcement.",
                "recap_documents": [
                    {
                        "document_number": 48,
                        "short_description": "Order on Stay",
                        "filepath_local": "gov.uscourts.cit.1/doc48.pdf",
                        "is_available": True,
                    }
                ],
            }
        ],
    }
    routes: dict[str, Route] = {
        "/api/rest/v4/search/": (200, opinions),
        "/api/rest/v4/docket-entries/": (200, entries),
    }
    srv = make_server(routes, seen)

    op_lines = call(srv, "search_court_opinions", query="injunction", court="cit", limit=2).splitlines()
    assert (seen[0].url.params["type"], seen[0].url.params["court"]) == ("o", "cit")
    assert op_lines[1].endswith("612 F. Supp. 3d 10 | V.O.S. Selections v. Trump | https://www.courtlistener.com/opinion/99/vos-selections/")
    assert op_lines[2] == "  snippet: Preliminary injunction GRANTED."

    de_lines = call(srv, "get_docket_entries", docket_id=4201, limit=5).splitlines()
    assert (seen[1].url.params["docket"], seen[1].url.params["order_by"]) == ("4201", "-date_filed")
    assert de_lines[1] == "#48 | 2026-09-10 | ORDER granting motion to stay enforcement."
    assert de_lines[2] == "  doc 48: Order on Stay | https://storage.courtlistener.com/gov.uscourts.cit.1/doc48.pdf"


# --- Epistemic SQLite Ledger & Brier Calibration ----------------------------


def test_epistemic_ledger_lifecycle_brier_skill_and_uncertainty_scaling(tmp_path: Path) -> None:
    db_file = tmp_path / "test_ledger.sqlite3"
    srv = make_server({}, ledger=lambda: LedgerSettings(db_path=str(db_file)))

    up = call(
        srv,
        "record_forecast",
        market_id="m1",
        market_slug="fed-cut-dec",
        family="macro",
        horizon_days=30,
        market_prob=0.50,
        fair_prob=0.80,
        fair_prob_sd=0.05,
        side="YES",
        recommended_usd=40.0,
        thesis_summary="Core PCE decelerating below Fed projection.",
        key_sources=["FRED:PCEPILFE"],
        invalidation_triggers=["Core CPI MoM > 0.4%"],
    )
    assert "LEDGER UPSERT m1 (fed-cut-dec)" in up
    call(
        srv,
        "record_forecast",
        market_id="m1",
        market_slug="fed-cut-dec",
        family="macro",
        horizon_days=29,
        market_prob=0.50,
        fair_prob=0.80,
        fair_prob_sd=0.05,
        side="YES",
        recommended_usd=40.0,
        order_id="0xmaker1",
        thesis_summary="Core PCE decelerating below Fed projection.",
        invalidation_triggers=["Core CPI MoM > 0.4%"],
    )
    active = call(srv, "get_active_theses", family="macro")
    assert "0xmaker1" in active and "Core CPI MoM > 0.4%" in active

    call(srv, "record_forecast", market_id="m2", market_slug="cpi-cool", family="macro", horizon_days=20, market_prob=0.50, fair_prob=0.90)
    call(srv, "record_forecast", market_id="m3", market_slug="gdp-neg", family="macro", horizon_days=45, market_prob=0.50, fair_prob=0.20)
    call(srv, "record_forecast", market_id="c1", market_slug="acme-mna", family="corporate", horizon_days=60, market_prob=0.40, fair_prob=0.70)

    call(srv, "resolve_and_score_calibration", market_id="m1", outcome=1)
    call(srv, "resolve_and_score_calibration", market_id="m2", outcome=1)
    cal = call(srv, "resolve_and_score_calibration", market_id="m3", outcome=0)

    # Agent Brier = ((0.8-1)^2 + (0.9-1)^2 + (0.2-0)^2) / 3 = 0.0300; Market Brier = 0.2500; Skill = 1 - 0.03/0.25 = +0.8800
    assert "Calibration (3 resolved) | agent_brier 0.0300 | market_brier 0.2500 | brier_skill_score +0.8800" in cal
    assert "macro | 3 | 0.0300 | 0.2500 | +0.8800 | 0.50x" in cal
    assert "0.8-0.9 | 1 | 0.8000 | 1.0000 | -0.2000" in cal

    # Resolved theses leave active list; c1 remains open; settled markets cannot be overwritten.
    remaining = call(srv, "get_active_theses")
    assert "c1 | acme-mna" in remaining and "m1 |" not in remaining
    with pytest.raises(ToolError, match="already resolved in the ledger"):
        call(srv, "record_forecast", market_id="m1", market_slug="fed-cut-dec", family="macro", horizon_days=10, market_prob=0.5, fair_prob=0.6)
    with pytest.raises(ToolError, match="not found in ledger"):
        call(srv, "resolve_and_score_calibration", market_id="missing", outcome=1)

    # Pure CalibrationMath check: badly overconfident family (fair=0.90, sd=0.05, outcome=0) inflates suggested_sd_multiplier > 1.25x.
    overconfident = [
        ForecastRecord(f"b{i}", "s", "politics", 10, 0.50, 0.90, 0.05, "YES", 10, None, "", (), (), "resolved", 0, "t", "t", "t")
        for i in range(3)
    ]
    bad_report = CalibrationMath.score(overconfident)
    assert bad_report.brier_skill_score == pytest.approx(1.0 - 0.81 / 0.25)
    assert bad_report.families[0].suggested_sd_multiplier == 3.0


def test_server_instructions_and_tool_routing_metadata() -> None:
    srv = make_server({})
    assert srv.instructions is not None
    for expected in ("get_active_theses", "get_market_resolution_rules", "Falcon MCP", "Firecrawl MCP"):
        assert expected in srv.instructions

    async def inspect_tools() -> dict[str, str]:
        async with Client(srv) as client:
            return {t.name: (t.description or "") for t in await client.list_tools()}

    descriptions = asyncio.run(inspect_tools())
    assert len(descriptions) == 33
    assert "Wallet 360" in descriptions["get_market_holders"]
    assert "firecrawl_scrape" in descriptions["search_sec_filings"]
    assert "Kalshi Markets" in descriptions["search_predictit"]
