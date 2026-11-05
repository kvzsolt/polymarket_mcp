"""Polymarket Quant MCP server: macro, markets, crypto pricing, politics/legal intel, cross-venue odds.

Run over stdio with ``python mcp_server.py``, or point the FastMCP CLI at this
file (``fastmcp list mcp_server.py``); the CLI discovers the module-level ``mcp``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Generic, TypeVar

import httpx2 as httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from crypto_client import CryptoClient, CryptoError, CryptoSettings, ExpiryPricing
from fred_client import SEARCH_LIMIT_MAX, FredClient, FredError, FredSettings, Units
from gov_client import BillType, FederalRegisterResult, GovClient, GovError, GovSettings
from politics_intel_client import (
    DEFAULT_ODDS_SPORT,
    GDELT_MAX_RECORDS,
    EventOdds,
    FedRelease,
    PoliticsIntelClient,
    PoliticsIntelError,
    PoliticsIntelSettings,
    Poll,
    PollDropoutAnalysis,
    PollDropoutCalculator,
)
from poly_client import BundleScan, EdgeReport, Fill, OrderCheck, OrderSender, PolyClient, PolyError, PolySettings, Side
from polynode_client import (
    ONLINE_SEARCH_MAX,
    POSITIONS_MAX,
    X_SEARCH_MAX,
    HolderReport,
    HolderStatus,
    PolynodeClient,
    PolynodeError,
    PolynodeSettings,
    TimeRange,
)

SERVER_NAME = "Polymarket Quant MCP"
ENV_FILE = Path(__file__).resolve().parent / ".env"
EARLIEST_DATE = date(1776, 7, 4)  # FRED's documented defaults: full history
LATEST_DATE = date(9999, 12, 31)

S = TypeVar("S")
C = TypeVar("C")
T = TypeVar("T")


def load_env() -> Mapping[str, str]:
    """``os.environ`` after loading the ``.env`` beside this file (exported vars win).

    Anchored to this file because MCP hosts start stdio servers from arbitrary
    directories with a minimal environment.
    """
    load_dotenv(ENV_FILE)
    return os.environ


def load_fred_settings() -> FredSettings:
    """FRED settings from the environment."""
    return FredSettings.from_env(load_env())


def load_poly_settings() -> PolySettings:
    """Polymarket settings from the environment."""
    return PolySettings.from_env(load_env())


def load_crypto_settings() -> CryptoSettings:
    """Binance/Deribit settings from the environment."""
    return CryptoSettings.from_env(load_env())


def load_polynode_settings() -> PolynodeSettings:
    """PolyNode settings from the environment."""
    return PolynodeSettings.from_env(load_env())


def load_gov_settings() -> GovSettings:
    """Federal Register / Congress.gov / FEC / CourtListener settings from the environment."""
    return GovSettings.from_env(load_env())


def load_politics_settings() -> PoliticsIntelSettings:
    """GDELT / Fed RSS / PredictIt / The Odds API settings from the environment."""
    return PoliticsIntelSettings.from_env(load_env())


class Gateway(Generic[S, C]):
    """Builds a fresh client per tool call and turns its domain error into ``ToolError``.

    Settings load per call, so a missing key is a readable tool error while the
    server still boots and lists its tools. FastMCP forwards a ToolError's
    message verbatim; any other exception reaches the client as
    "Error calling tool '<name>': <str(exc)>" (unmasked by default in fastmcp
    4.1.6), so clients must raise their domain error with a URL-free message.
    """

    def __init__(
        self,
        settings_loader: Callable[[], S],
        client_factory: Callable[[S, httpx.AsyncClient], C],
        error: type[Exception],
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        """Create the gateway.

        Args:
            settings_loader: Produces settings; raises ``error`` when unusable.
            client_factory: Builds the API client from settings and an open HTTP client.
            error: The client's base exception, translated into ``ToolError``.
            transport: Optional HTTP transport; tests pass ``httpx2.MockTransport``.
        """
        self._settings_loader = settings_loader
        self._client_factory = client_factory
        self._error = error
        self._transport = transport

    async def call(self, operation: Callable[[C], Awaitable[T]]) -> T:
        """Run ``operation`` against a fresh client; domain errors become ``ToolError``."""
        try:
            settings = self._settings_loader()
            async with httpx.AsyncClient(transport=self._transport) as http:
                return await operation(self._client_factory(settings, http))
        except self._error as exc:
            raise ToolError(str(exc)) from exc


def _num(value: float | None, fmt: str = ".4f") -> str:
    return "n/a" if value is None else format(value, fmt)


def _clip(text: Any, limit: int = 300) -> str:
    """One line, at most ``limit`` characters."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _ordinal(value: Any) -> str:
    """``119`` -> ``119th``, ``121`` -> ``121st``; non-numbers pass through."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return str(value or "")
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _signed_usd(value: float) -> str:
    return f"{'-' if value < 0 else ''}${abs(value):,.2f}"


def _usd(value: Any) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


class TextFormatter:
    """Renders API data as compact text lines for an LLM client."""

    @staticmethod
    def series(series_id: str, data: dict[str, Any]) -> str:
        """``date: value`` lines, skipping missing (".") values."""
        observations = data.get("observations", [])
        rows = "\n".join(f"{o['date']}: {o['value']}" for o in observations if o["value"] != ".")
        return f"FRED series {series_id} ({len(observations)} observations)\n{rows}"

    @staticmethod
    def search(search_text: str, data: dict[str, Any]) -> str:
        """``ID: title`` lines."""
        matches = data.get("seriess") or []
        if not matches:
            return f"No FRED series found for '{search_text}'."
        return "\n".join(f"{s['id']}: {s['title']}" for s in matches)

    @staticmethod
    def markets(markets: list[dict[str, Any]]) -> str:
        """One row per market; NO quotes are the complement of the YES book."""
        if not markets:
            return "No liquid markets match."
        lines = ["market_id | question | end_date | yes_bid/ask | no_bid/ask | volume | yes_token no_token"]
        for m in markets:
            bid, ask = float(m["bestBid"]), float(m["bestAsk"])
            tokens = " ".join(json.loads(m.get("clobTokenIds") or "[]"))
            lines.append(
                f"{m['id']} | {m['question']} | {str(m.get('endDate', ''))[:10]} | {bid:.3f}/{ask:.3f} | "
                f"{1 - ask:.3f}/{1 - bid:.3f} | ${float(m.get('volumeNum') or 0):,.0f} | {tokens}"
            )
        return "\n".join(lines)

    @staticmethod
    def fill(f: Fill) -> str:
        """Fill summary shared by the order tools."""
        status = "full" if f.complete else f"PARTIAL ${f.filled_usd:.2f} of ${f.requested_usd:.2f}"
        return (
            f"{f.side} ${f.filled_usd:.2f} -> {f.shares:.2f} sh | vwap {_num(f.vwap)} | best {_num(f.best)} | "
            f"slippage {_num(f.slippage_bps, '.1f')} bps | post-trade best {_num(f.post_trade_best)} | {status}"
        )

    @staticmethod
    def edge(r: EdgeReport) -> str:
        """Fill line plus edge and Kelly line."""
        return (
            f"{TextFormatter.fill(r.fill)}\n"
            f"fair {r.fair_prob:.4f} | edge {r.edge:+.4f} | kelly {r.kelly_full:.4f} | kelly_shrunk {r.kelly_shrunk:.4f}"
        )

    @staticmethod
    def bundle(s: BundleScan) -> str:
        """Per-leg VWAPs plus bundle sums and flagged edges."""
        lines = [f"event {s.event_slug} | {len(s.legs)} outcomes | negRisk={s.neg_risk}"]
        if not s.neg_risk:
            lines.append("WARNING: not a neg-risk event; outcomes may not be mutually exclusive, bundle math invalid.")
        lines += [
            f"{leg.title} | YES {_num(leg.yes.vwap)}{'' if leg.yes.complete else '*'} | "
            f"NO {_num(leg.no.vwap)}{'' if leg.no.complete else '*'}"
            for leg in s.legs
        ]
        for name, total, edge in (("YES", s.sum_yes, s.edge_yes), ("NO", s.sum_no, s.edge_no)):
            if total is None:
                lines.append(f"sum({name}) n/a: a leg (*) could not fully fill")
            else:
                flag = "  <-- ARB: buy all" if edge is not None and edge > 0 else ""
                lines.append(f"sum({name}) {total:.4f} | bundle edge {edge:+.4f}{flag}")
        return "\n".join(lines)

    @staticmethod
    def order(c: OrderCheck) -> str:
        """Order validation and, when live, the exchange result."""
        head = (
            f"{c.side} {c.shares:.2f} sh @ {c.limit_price} (${c.notional_usd:.2f}) token {c.token_id} market {c.market_id}\n"
            f"crosses now: {TextFormatter.fill(c.marketable)}\nrests on book: ${c.resting_usd:.2f}"
        )
        if c.result is None:
            tail = "DRY RUN: validated, not sent."
        elif c.result.ok:
            tail = f"SENT: order {c.result.order_id} status {c.result.status}" + (
                f" | deposit wallet {c.result.wallet} (sig type 3)" if c.result.wallet else ""
            )
        else:
            tail = f"REJECTED by exchange: {c.result.status} {c.result.message}"
        return f"{head}\n{tail}" + (f"\nreasoning: {c.reasoning}" if c.reasoning else "")

    @staticmethod
    def expiry(p: ExpiryPricing, fill_price_no: float | None) -> str:
        """One line per model; NO edge shown when a NO fill price is given."""
        lines = [
            f"{p.symbol} spot {p.spot:,.2f} | strike {p.strike:,.2f} | r* {p.r_star:+.5f} | horizon {p.horizon_hours}h",
            "model | P(YES: close > strike) | P(NO) | sigma_h | n"
            + (" | edge_NO" if fill_price_no is not None else ""),
        ]
        for e in p.estimates:
            edge = f" | {e.p_no - fill_price_no:+.4f}" if fill_price_no is not None else ""
            lines.append(f"{e.name} | {e.p_yes:.4f} | {e.p_no:.4f} | {_num(e.sigma_h, '.5f')} | {e.samples}{edge}")
        return "\n".join([*lines, f"implied vol: {p.iv_note}"])

    @staticmethod
    def x_search(query: str, data: dict[str, Any]) -> str:
        """``[time] @user (likes, retweets): text`` lines, newest first."""
        tweets = data.get("tweets") or []
        if not tweets:
            return f"No X posts found for '{query}'."
        rows = []
        for t in tweets:
            m = t.get("public_metrics") or {}
            user = (t.get("author") or {}).get("username", "unknown")
            rows.append(
                f"[{t.get('created_at', '')}] @{user} (likes={m.get('like_count', 0)}, rt={m.get('retweet_count', 0)}): "
                f"{_clip(t.get('text'), 400)}"
            )
        return f"PolyNode X Search '{data.get('query', query)}' ({len(tweets)} posts)\n" + "\n".join(rows)

    @staticmethod
    def online_search(query: str, data: dict[str, Any]) -> str:
        """``[date] domain | title | url`` plus a snippet line per result."""
        results = data.get("results") or []
        if not results:
            return f"No web results for '{query}' (status {data.get('status', 'n/a')})."
        head = f"PolyNode Online Search '{query}' ({len(results)} results, status {data.get('status', 'ok')})"
        if data.get("unresponsive_engines"):
            head += f" | unresponsive engines: {', '.join(map(str, data['unresponsive_engines']))}"
        rows = [
            f"[{r.get('published_at') or '-'}] {r.get('domain', '')} | {_clip(r.get('title'), 150)} | {r.get('url', '')}\n"
            f"  {_clip(r.get('snippet'), 250)}"
            for r in results
        ]
        return "\n".join([head, *rows])

    @staticmethod
    def holders(r: HolderReport, top_n: int) -> str:
        """Per-token concentration lines, then the largest holder rows."""
        if not r.holders:
            return f"No positions found for market slug '{r.slug}'."
        lines = [
            f"market {r.slug} | {len(r.holders)} rows"
            + (" (more exist; stats cover fetched rows)" if r.has_more else "")
        ]
        lines += [
            f"token {t.token_id[-8:]} | holders {t.holders} | shares {t.total_shares:,.0f} | "
            f"top{t.top_n} {t.top_share:.1%} | HHI {t.hhi:.3f} | avg entry {_num(t.avg_entry)}"
            for t in r.tokens
        ]
        lines.append("wallet | token | shares | avg_price | realized_pnl")
        lines += [
            f"{h.wallet} | {h.token_id[-8:]} | {h.shares:,.0f} | {h.avg_price:.4f} | {_signed_usd(h.realized_pnl_usd)}"
            for h in r.holders[:top_n]
        ]
        return "\n".join(lines)

    @staticmethod
    def federal_register(term: str, r: FederalRegisterResult) -> str:
        """Pending (public inspection) documents first, then published ones."""
        lines = [
            f"Federal Register '{term}': {len(r.published)} published, {len(r.public_inspection)} on public inspection"
        ]
        if r.public_inspection_error:
            lines.append(f"public inspection unavailable: {r.public_inspection_error}")
        lines += [
            f"[PUBLIC INSPECTION] {d.get('filed_at') or d.get('publication_date')}: {_clip(d.get('title'), 200)} "
            f"(doc={d.get('document_number')}, pub {d.get('publication_date')})"
            for d in r.public_inspection
        ]
        lines += [
            f"[PUBLISHED] {d.get('publication_date')}: {_clip(d.get('title'), 200)} "
            f"(type={d.get('type')}, doc={d.get('document_number')})"
            for d in r.published
        ]
        return "\n".join(lines)

    @staticmethod
    def bill(b: dict[str, Any]) -> str:
        """Title, intro date, sponsor and latest action of one bill."""
        latest = b.get("latestAction") or {}
        sponsor = ((b.get("sponsors") or [{}])[0]).get("fullName", "n/a")
        policy = (b.get("policyArea") or {}).get("name", "n/a")
        return (
            f"{b.get('type', '')}.{b.get('number', '')} ({_ordinal(b.get('congress'))}) {_clip(b.get('title'), 200)}\n"
            f"introduced {b.get('introducedDate', 'n/a')} in {b.get('originChamber', 'n/a')} | sponsor {sponsor} | "
            f"policy area {policy}\n"
            f"latest action {latest.get('actionDate', 'n/a')}: {_clip(latest.get('text'), 300)}"
        )

    @staticmethod
    def nomination(n: dict[str, Any]) -> str:
        """Citation, description, received date and latest action of one nomination."""
        latest = n.get("latestAction") or {}
        return (
            f"{n.get('citation', '')} ({_ordinal(n.get('congress'))}) {_clip(n.get('description'), 300)}\n"
            f"received {n.get('receivedDate', 'n/a')} | organization {n.get('organization', 'n/a')}\n"
            f"latest action {latest.get('actionDate', 'n/a')}: {_clip(latest.get('text'), 300)}"
        )

    @staticmethod
    def candidates(query: str, cycle: int, rows: list[dict[str, Any]]) -> str:
        """One row per candidate, richest cash on hand first; burn = disbursements / receipts."""
        if not rows:
            return f"No FEC candidate totals for '{query}' in {cycle}."
        lines = [
            f"FEC {cycle} '{query}': name | id | party office-state | receipts | disbursed | burn | cash | debt | through"
        ]
        for c in rows:
            receipts, spent = c.get("receipts") or 0, c.get("disbursements") or 0
            try:
                burn = f"{float(spent) / float(receipts):.0%}"
            except (TypeError, ValueError, ZeroDivisionError):
                burn = "n/a"
            lines.append(
                f"{c.get('name')} | {c.get('candidate_id')} | {c.get('party')} {c.get('office')}-{c.get('state')} | "
                f"{_usd(receipts)} | {_usd(spent)} | {burn} | {_usd(c.get('cash_on_hand_end_period'))} | "
                f"{_usd(c.get('debts_owed_by_committee'))} | {c.get('coverage_end_date') or 'n/a'}"
            )
        return "\n".join(lines)

    @staticmethod
    def dockets(query: str, data: dict[str, Any]) -> str:
        """``date | court | docket number | case name | url`` per docket."""
        results = data.get("results") or []
        if not results:
            return f"No CourtListener dockets for '{query}'."
        lines = [f"CourtListener dockets '{query}' ({data.get('count')} total, showing {len(results)})"]
        lines += [
            f"{d.get('dateFiled') or 'n/a'} | {d.get('court', '')} | {d.get('docketNumber', '')} | "
            f"{_clip(d.get('caseName'), 150)} | https://www.courtlistener.com"
            f"{d.get('docket_absolute_url') or d.get('absolute_url') or ''}"
            for d in results
        ]
        return "\n".join(lines)

    @staticmethod
    def gdelt_articles(query: str, articles: list[dict[str, Any]]) -> str:
        """``seendate | country domain | title | url`` per article."""
        if not articles:
            return f"No GDELT articles for '{query}'."
        lines = [f"GDELT '{query}' ({len(articles)} articles, newest first)"]
        lines += [
            f"{a.get('seendate', '')} | {a.get('sourcecountry', '')} {a.get('domain', '')} | "
            f"{_clip(a.get('title'), 200)} | {a.get('url', '')}"
            for a in articles
        ]
        return "\n".join(lines)

    @staticmethod
    def gdelt_volume(query: str, points: list[tuple[str, float]], tail: int = 12) -> str:
        """Summary (latest, mean, peak) plus the last ``tail`` points of the coverage share series."""
        if not points:
            return f"No GDELT coverage timeline for '{query}'."
        values = [v for _, v in points]
        peak = max(range(len(values)), key=values.__getitem__)
        lines = [
            (
                f"GDELT volume '{query}' ({len(points)} points, % of monitored coverage) | latest {values[-1]:.4f} | "
                f"mean {sum(values) / len(values):.4f} | peak {values[peak]:.4f} at {points[peak][0]}"
            )
        ]
        lines += [f"{d}: {v:.4f}" for d, v in points[-tail:]]
        return "\n".join(lines)

    @staticmethod
    def fed_releases(items: tuple[FedRelease, ...]) -> str:
        """``pubDate | title | link`` per release."""
        if not items:
            return "No Federal Reserve press releases in the feed."
        return "\n".join(f"{i.published} | {_clip(i.title, 200)} | {i.link}" for i in items)

    @staticmethod
    def predictit(keyword: str, markets: list[dict[str, Any]], max_contracts: int = 8) -> str:
        """Market header lines, each followed by its contracts' last/buy-yes/buy-no prices."""
        if not markets:
            return f"No PredictIt markets found for '{keyword}'."
        lines = [f"PredictIt matches for '{keyword}' ({len(markets)} markets):"]
        for m in markets:
            lines.append(f"Market {m.get('id')}: {m.get('name')} [{m.get('status', '')}]")
            for c in (m.get("contracts") or [])[:max_contracts]:
                lines.append(
                    f"  - {c.get('name')}: last={_num(c.get('lastTradePrice'), '.2f')} "
                    f"buyYes={_num(c.get('bestBuyYesCost'), '.2f')} buyNo={_num(c.get('bestBuyNoCost'), '.2f')}"
                )
        return "\n".join(lines)

    @staticmethod
    def bookmaker_odds(sport_key: str, events: tuple[EventOdds, ...], max_outcomes: int = 15) -> str:
        """No-vig consensus probability, best decimal price and book count per outcome."""
        if not events:
            return f"No bookmaker odds for '{sport_key}' (market closed or out of season)."
        lines = [f"Bookmaker outrights '{sport_key}' (no-vig consensus)"]
        for e in events:
            lines.append(
                f"{' '.join(filter(None, (e.title, e.commence_time)))} | no-vig from {e.usable_books} book(s), "
                f"{e.skipped_books} skipped (quotes sum < 100%)"
            )
            lines += [
                f"  {o.name}: fair {_num(o.fair_prob)} | best {o.best_price:.2f} | books {o.books}"
                for o in e.outcomes[:max_outcomes]
            ]
        return "\n".join(lines)

    @staticmethod
    def poll_dropout(a: PollDropoutAnalysis) -> str:
        """Current average, what drops out, and the resulting shift."""
        o = a.oldest
        lines = [
            (
                f"polls {a.polls} | current avg {a.current_avg:.3f} | without oldest {a.avg_without_oldest:.3f} | "
                f"shift if dropped {a.shift_if_dropped:+.3f}"
            ),
            f"oldest: {o.pollster} {o.end_date.isoformat()} value {o.value:g}",
        ]
        if a.oldest_ties:
            lines.append(
                f"WARNING: {a.oldest_ties} other poll(s) share the oldest end date; which one drops is ambiguous."
            )
        return "\n".join(lines)


def create_server(
    fred_settings_loader: Callable[[], FredSettings] = load_fred_settings,
    poly_settings_loader: Callable[[], PolySettings] = load_poly_settings,
    crypto_settings_loader: Callable[[], CryptoSettings] = load_crypto_settings,
    polynode_settings_loader: Callable[[], PolynodeSettings] = load_polynode_settings,
    gov_settings_loader: Callable[[], GovSettings] = load_gov_settings,
    politics_settings_loader: Callable[[], PoliticsIntelSettings] = load_politics_settings,
    transport: httpx.AsyncBaseTransport | None = None,
    order_sender: OrderSender | None = None,
) -> FastMCP:
    """Build the MCP server and register its tools.

    Args:
        fred_settings_loader: FRED settings source, called per tool call.
        poly_settings_loader: Polymarket settings source, called per tool call.
        crypto_settings_loader: Binance/Deribit settings source, called per tool call.
        polynode_settings_loader: PolyNode settings source, called per tool call.
        gov_settings_loader: Federal Register/Congress/FEC/CourtListener settings source, called per tool call.
        politics_settings_loader: GDELT/Fed RSS/PredictIt/Odds API settings source, called per tool call.
        transport: Optional HTTP transport for every API, injected by tests.
        order_sender: Live order sender; defaults to the official Polymarket SDK.
    """
    server = FastMCP(SERVER_NAME)
    fred = Gateway(fred_settings_loader, FredClient, FredError, transport)
    poly = Gateway(poly_settings_loader, lambda s, h: PolyClient(s, h, order_sender), PolyError, transport)
    crypto = Gateway(crypto_settings_loader, CryptoClient, CryptoError, transport)
    polynode = Gateway(polynode_settings_loader, PolynodeClient, PolynodeError, transport)
    gov = Gateway(gov_settings_loader, GovClient, GovError, transport)
    intel = Gateway(politics_settings_loader, PoliticsIntelClient, PoliticsIntelError, transport)

    @server.tool
    async def get_series(
        series_id: Annotated[str, Field(min_length=1, description='FRED series id, e.g. "GDP", "UNRATE".')],
        observation_start: Annotated[date, Field(description="First observation date (YYYY-MM-DD).")] = EARLIEST_DATE,
        observation_end: Annotated[date, Field(description="Last observation date (YYYY-MM-DD).")] = LATEST_DATE,
        units: Annotated[Units, Field(description="lin, chg, ch1, pch, pc1, pca, cch, cca, cap, log.")] = "lin",
    ) -> str:
        """Get a FRED economic time series (metadata + observations)."""
        data = await fred.call(lambda c: c.series_observations(series_id, observation_start, observation_end, units))
        return TextFormatter.series(series_id, data)

    @server.tool
    async def search_series(
        search_text: Annotated[str, Field(min_length=1, description='Text to search for, e.g. "unemployment".')],
        limit: Annotated[int, Field(ge=1, le=SEARCH_LIMIT_MAX, description="Max results (1-1000).")] = 10,
    ) -> str:
        """Search for FRED series by keyword."""
        return TextFormatter.search(search_text, await fred.call(lambda c: c.search_series(search_text, limit)))

    @server.tool
    async def search_markets(
        query: Annotated[str | None, Field(description="Words that must all appear in the question.")] = None,
        tag: Annotated[str | None, Field(description='Gamma tag slug, e.g. "politics", "crypto".')] = None,
        min_volume_usd: Annotated[float, Field(ge=0, description="Minimum lifetime volume in USD.")] = 10_000,
        max_days_to_expiry: Annotated[
            float | None, Field(gt=0, description="Only markets ending within N days.")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=100, description="Max rows.")] = 20,
    ) -> str:
        """Find active, liquid Polymarket markets (two-sided book, YES spread <= 10c)."""
        now = datetime.now(UTC)
        end_max = now + timedelta(days=max_days_to_expiry) if max_days_to_expiry else None
        markets = await poly.call(lambda c: c.search_markets(query, tag, min_volume_usd, now, end_max, limit))
        return TextFormatter.markets(markets)

    @server.tool
    async def simulate_clob_fill(
        token_id: Annotated[str, Field(min_length=1, description="CLOB outcome token id.")],
        side: Annotated[Side, Field(description="BUY walks the asks, SELL walks the bids.")],
        amount_usd: Annotated[float, Field(gt=0, description="USD to spend (BUY) or raise (SELL).")],
        fair_prob: Annotated[float, Field(gt=0, lt=1, description="Your fair probability for this token.")],
        fair_prob_sd: Annotated[float, Field(ge=0, lt=1, description="Std dev of fair_prob; shrinks Kelly.")] = 0.05,
    ) -> str:
        """Walk the live order book: VWAP, slippage, post-trade price, edge and shrunk Kelly."""
        report = await poly.call(lambda c: c.simulate_fill(token_id, side, amount_usd, fair_prob, fair_prob_sd))
        return TextFormatter.edge(report)

    @server.tool
    async def scan_neg_risk_event(
        event_slug: Annotated[str, Field(min_length=1, description="Gamma event slug.")],
        size_usd: Annotated[float, Field(gt=0, description="USD bought per outcome leg when pricing.")] = 50.0,
    ) -> str:
        """Price buying every YES (pays 1) and every NO (pays n-1) at real depth; flag bundle mispricings."""
        return TextFormatter.bundle(await poly.call(lambda c: c.scan_neg_risk(event_slug, size_usd)))

    @server.tool
    async def place_order(
        market_id: Annotated[str, Field(min_length=1, description="Gamma market id (from search_markets).")],
        token_id: Annotated[str, Field(min_length=1, description="Outcome token id belonging to that market.")],
        side: Annotated[Side, Field(description="BUY or SELL.")],
        amount_usd: Annotated[float, Field(gt=0, description="Order notional; capped by POLY_MAX_ORDER_USD.")],
        limit_price: Annotated[float, Field(gt=0, lt=1, description="Limit price, on the market's tick.")],
        dry_run: Annotated[bool, Field(description="True validates only; False signs and sends.")] = True,
        reasoning: Annotated[str, Field(description="Why this trade; logged and echoed.")] = "",
    ) -> str:
        """Validate (and, with dry_run=False, send) a GTC limit order from the Deposit Wallet."""
        check = await poly.call(
            lambda c: c.place_order(market_id, token_id, side, amount_usd, limit_price, dry_run, reasoning)
        )
        return TextFormatter.order(check)

    @server.tool
    async def price_crypto_expiry(
        symbol: Annotated[str, Field(pattern=r"^[A-Z0-9]{5,20}$", description='Binance pair, e.g. "BTCUSDT".')],
        strike: Annotated[float, Field(gt=0, description="Strike from the market's resolution rules.")],
        horizon_hours: Annotated[int, Field(ge=1, le=720, description="Hours until the resolving candle closes.")],
        fill_price_no: Annotated[
            float | None, Field(gt=0, lt=1, description="Your NO fill price; adds edge_NO = P(NO) - price.")
        ] = None,
    ) -> str:
        """P(close > strike) and P(NO) from empirical, 7-day normal, full-sample normal and implied-vol models."""
        pricing = await crypto.call(lambda c: c.price_expiry(symbol, strike, horizon_hours))
        return TextFormatter.expiry(pricing, fill_price_no)

    # --- PolyNode: X, web context, holder concentration ---------------------

    @server.tool
    async def search_x_context(
        query: Annotated[str, Field(min_length=1, description='X query; operators work, e.g. "from:AtlasIntel".')],
        max_results: Annotated[int, Field(ge=1, le=X_SEARCH_MAX, description="Posts to return (1-50).")] = 15,
        since: Annotated[date | None, Field(description="Only posts on/after this date (YYYY-MM-DD).")] = None,
        until: Annotated[date | None, Field(description="Only posts before this date (YYYY-MM-DD).")] = None,
    ) -> str:
        """Latest X posts (pollsters, politicians, primary sources) via PolyNode. Uses monthly quota; 1 call/s."""
        data = await polynode.call(lambda c: c.x_search(query, max_results, since, until))
        return TextFormatter.x_search(query, data)

    @server.tool
    async def search_online_context(
        query: Annotated[str, Field(min_length=1, description="Web search query.")],
        max_results: Annotated[int, Field(ge=1, le=ONLINE_SEARCH_MAX, description="Results (1-20).")] = 10,
        time_range: Annotated[TimeRange | None, Field(description="Freshness: day, week, month, year.")] = None,
    ) -> str:
        """Corroborating web/news results via PolyNode Online Search."""
        data = await polynode.call(lambda c: c.online_search(query, max_results, time_range))
        return TextFormatter.online_search(query, data)

    @server.tool
    async def get_market_holders(
        market_slug: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", description="Polymarket market slug.")],
        limit: Annotated[int, Field(ge=1, le=POSITIONS_MAX, description="Largest positions to fetch (1-300).")] = 100,
        top_n: Annotated[int, Field(ge=1, le=50, description="Top-N for the concentration share and rows shown.")] = 10,
        status: Annotated[
            HolderStatus | None, Field(description="open = current holders, closed = exited, null = both.")
        ] = "open",
    ) -> str:
        """Whale positioning: largest holders, avg entry, top-N share and HHI per outcome token."""
        report = await polynode.call(lambda c: c.market_positions(market_slug, limit, top_n, status))
        return TextFormatter.holders(report, top_n)

    # --- Government and legal oracles ---------------------------------------

    @server.tool
    async def search_federal_register(
        term: Annotated[str, Field(min_length=1, description='Full-text term, e.g. "tariff", "executive order".')],
        include_public_inspection: Annotated[
            bool, Field(description="Also list filed-but-unpublished (public inspection) documents.")
        ] = True,
        limit: Annotated[int, Field(ge=1, le=100, description="Max documents per section.")] = 10,
    ) -> str:
        """Executive orders, tariffs and agency rules: published and pending public-inspection documents."""
        result = await gov.call(lambda c: c.federal_register_search(term, include_public_inspection, limit))
        return TextFormatter.federal_register(term, result)

    @server.tool
    async def get_bill_status(
        congress: Annotated[int, Field(ge=93, le=200, description="Congress number, e.g. 119.")],
        bill_type: Annotated[BillType, Field(description="hr, s, hjres, sjres, hconres, sconres, hres, sres.")],
        bill_number: Annotated[int, Field(ge=1, description="Bill number, e.g. 1 for H.R. 1.")],
    ) -> str:
        """Legislative status and latest action of a US bill (Congress.gov; needs CONGRESS_API_KEY)."""
        return TextFormatter.bill(await gov.call(lambda c: c.congress_bill_status(congress, bill_type, bill_number)))

    @server.tool
    async def get_nomination_status(
        congress: Annotated[int, Field(ge=93, le=200, description="Congress number, e.g. 119.")],
        nomination_number: Annotated[int, Field(ge=1, description="PN number, e.g. 12 for PN12.")],
    ) -> str:
        """Status and latest Senate action of a presidential nomination (Congress.gov; needs CONGRESS_API_KEY)."""
        data = await gov.call(lambda c: c.congress_nomination(congress, nomination_number))
        return TextFormatter.nomination(data)

    @server.tool
    async def get_candidate_financials(
        query: Annotated[str, Field(min_length=2, description='Candidate name, e.g. "ossoff".')],
        cycle: Annotated[int, Field(ge=1980, le=2100, description="Election year, e.g. 2026.")] = 2026,
        limit: Annotated[int, Field(ge=1, le=100, description="Max candidates.")] = 10,
    ) -> str:
        """FEC receipts, spending, burn, cash on hand and debt: drop-out risk signals."""
        rows = await gov.call(lambda c: c.fec_candidate_financials(query, cycle, limit))
        return TextFormatter.candidates(query, cycle, rows)

    @server.tool
    async def search_court_dockets(
        query: Annotated[str, Field(min_length=1, description='CourtListener query, e.g. "tariff injunction".')],
        limit: Annotated[int, Field(ge=1, le=20, description="Max dockets.")] = 10,
    ) -> str:
        """Federal court and SCOTUS dockets, newest filing first (CourtListener; needs COURTLISTENER_API_TOKEN)."""
        return TextFormatter.dockets(query, await gov.call(lambda c: c.courtlistener_dockets(query, limit)))

    # --- News volume, Fed, cross-venue odds, polling ------------------------

    @server.tool
    async def search_gdelt_news(
        query: Annotated[str, Field(min_length=1, description='GDELT query, e.g. "tariff sourcecountry:US".')],
        timespan: Annotated[str, Field(pattern=r"^\d{1,3}(min|h|d|w|m)$", description="e.g. 15min, 24h, 7d.")] = "24h",
        limit: Annotated[int, Field(ge=1, le=GDELT_MAX_RECORDS, description="Max articles.")] = 15,
    ) -> str:
        """Newest global and local news articles from GDELT (no key; 1 request per 5 s)."""
        return TextFormatter.gdelt_articles(query, await intel.call(lambda c: c.gdelt_articles(query, timespan, limit)))

    @server.tool
    async def get_gdelt_volume(
        query: Annotated[str, Field(min_length=1, description="GDELT query.")],
        timespan: Annotated[str, Field(pattern=r"^\d{1,3}(min|h|d|w|m)$", description="e.g. 24h, 3d.")] = "24h",
    ) -> str:
        """Media-attention time series (share of monitored coverage, 15-min steps on short spans)."""
        return TextFormatter.gdelt_volume(query, await intel.call(lambda c: c.gdelt_timeline(query, timespan)))

    @server.tool
    async def get_fed_releases(
        limit: Annotated[int, Field(ge=1, le=50, description="Max releases.")] = 10,
    ) -> str:
        """Latest Federal Reserve press releases (rate decisions, statements) from the official RSS feed."""
        return TextFormatter.fed_releases(await intel.call(lambda c: c.fed_press_releases(limit)))

    @server.tool
    async def search_predictit(
        keyword: Annotated[str, Field(min_length=1, description="Words that must all appear in the market name.")],
    ) -> str:
        """PredictIt markets and contract prices, for cross-venue comparison with Polymarket."""
        return TextFormatter.predictit(keyword, await intel.call(lambda c: c.predictit_markets(keyword)))

    @server.tool
    async def get_bookmaker_odds(
        sport_key: Annotated[
            str, Field(pattern=r"^[a-z0-9_]+$", description="The Odds API politics sport key.")
        ] = DEFAULT_ODDS_SPORT,
        regions: Annotated[
            str, Field(pattern=r"^(us|us2|uk|eu|au)(,(us|us2|uk|eu|au))*$", description="Bookmaker regions.")
        ] = "us,uk,eu",
    ) -> str:
        """No-vig bookmaker consensus for politics outrights (The Odds API; needs THE_ODDS_API_KEY)."""
        return TextFormatter.bookmaker_odds(
            sport_key, await intel.call(lambda c: c.bookmaker_politics_odds(sport_key, regions))
        )

    @server.tool
    async def analyze_poll_dropout(
        polls: Annotated[
            list[Poll],
            Field(min_length=2, description="Every poll in the average: pollster, end_date (YYYY-MM-DD), value."),
        ],
    ) -> str:
        """How a rolling polling average (e.g. RCP) shifts when its oldest poll drops out.

        This tool does not download polls. Fetch the RealClearPolitics (or other
        aggregator) table first, e.g. with the Firecrawl MCP ``firecrawl_scrape``
        and a JSON schema, then pass every row currently in the average here.
        """
        try:  # pure math: no settings or HTTP, so no gateway
            analysis = PollDropoutCalculator.analyze(polls)
        except PoliticsIntelError as exc:
            raise ToolError(str(exc)) from exc
        return TextFormatter.poll_dropout(analysis)

    return server


mcp = create_server()


def main() -> None:
    """Run the server over stdio. Logs go to stderr; stdout carries MCP frames."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
