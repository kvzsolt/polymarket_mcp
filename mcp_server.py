"""Polymarket Quant MCP server: slow-horizon research, calibration ledger, pricing, and maker execution.

Run over stdio with ``python mcp_server.py``, or point the FastMCP CLI at this
file (``fastmcp list mcp_server.py``); the CLI discovers the module-level ``mcp``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Generic, TypeVar

import httpx2 as httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from crypto_client import MAX_HORIZON_HOURS, ContractType, CryptoClient, CryptoError, CryptoSettings, ExpiryPricing
from fred_client import SEARCH_LIMIT_MAX, FredClient, FredError, FredSettings, Units
from gov_client import BillType, FederalRegisterResult, GovClient, GovError, GovSettings
from ledger_client import (
    CalibrationReport,
    ForecastRecord,
    ForecastSide,
    LedgerClient,
    LedgerError,
    LedgerSettings,
    MarketFamily,
)
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
from poly_client import (
    BundleScan,
    CancelResult,
    EdgeReport,
    EventSummary,
    Fill,
    MarketOrder,
    OrderCheck,
    OrderManager,
    OrderSender,
    PolyClient,
    PolyError,
    PolySettings,
    ResolutionRules,
    RestingOrder,
    Side,
)
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
from sec_client import (
    DEFAULT_SEARCH_FORMS,
    DEFAULT_SUBMISSION_FORMS,
    SEC_SEARCH_MAX,
    SEC_SUBMISSIONS_MAX,
    CompanySubmissionsReport,
    MergerMath,
    MergerSpreadAnalysis,
    SecClient,
    SecError,
    SecSearchResult,
    SecSettings,
)

SERVER_NAME = "Polymarket Quant MCP"
SERVER_INSTRUCTIONS = """\
Polymarket Quant MCP is a slow-horizon (days-to-months) quantitative research, settlement-rule auditing, \
Brier-calibrated forecasting, and guarded maker-execution server for Polymarket.

Capabilities:
1. Discovery & Settlement Audit: Paginated market/event search (`search_markets`, `search_events`), full UMA \
resolution-rule extraction & trap flagging (`get_market_resolution_rules`), and neg-risk bundle arb (`scan_neg_risk_event`).
2. Deterministic Quant Pricing: 365d crypto expiry + one-touch barrier reflection/pathwise pricing + Deribit IV skew \
(`price_crypto_expiry`), M&A deal closing probability & IRR (`calculate_merger_spread_prob`), rolling poll-dropout math \
(`analyze_poll_dropout`), de-vigged bookmaker consensus (`get_bookmaker_odds`), and horizon-aware shrunk Kelly + CLOB \
VWAP walk (`simulate_clob_fill`).
3. Primary US Oracles: FRED macro (`get_series`, `search_series`), Fed RSS (`get_fed_releases`), SEC EDGAR EFTS & \
submissions (`search_sec_filings`, `get_company_submissions`), Federal Register (`search_federal_register`), Congress.gov \
(`get_bill_status`, `get_nomination_status`), FEC (`get_candidate_financials`), CourtListener RECAP dockets/entries/opinions \
(`search_court_dockets`, `get_docket_entries`, `search_court_opinions`), GDELT (`search_gdelt_news`, `get_gdelt_volume`), \
PredictIt (`search_predictit`), and PolyNode (`search_x_context`, `search_online_context`, `get_market_holders`).
4. Epistemic Brier Ledger & Execution: SQLite thesis tracking & Brier calibration (`get_active_theses`, `record_forecast`, \
`resolve_and_score_calibration`) and `DEPOSIT_WALLET` (sig type 3) GTC limit order lifecycle (`place_order`, \
`list_open_orders`, `cancel_order`).

Mandatory Research-to-Execution Sequence:
1. `get_active_theses` (check existing exposure, prior thesis, and invalidation triggers; run `resolve_and_score_calibration` \
to check empirical family `fair_prob_sd` multiplier).
2. `get_market_resolution_rules` (audit exact UMA rules, timezone cutoffs, touch-vs-close, and `p3` 50-50 traps BEFORE modeling).
3. Primary-source oracle research + deterministic pricing (`price_crypto_expiry`, `calculate_merger_spread_prob`, \
`analyze_poll_dropout`, `get_bookmaker_odds`).
4. `simulate_clob_fill` with `days_to_resolution` (verify shrunk Kelly > 0, slippage, daily carry bps/d, and annualized IRR).
5. `record_forecast` (commit YES `fair_prob`, `fair_prob_sd`, `key_sources`, and `invalidation_triggers` to the ledger).
6. `place_order(dry_run=True)` first to validate cap/tick/min-size/crossing, then `place_order(dry_run=False)` only when explicitly authorized.

Multi-MCP Orchestration Boundaries:
- **Falcon MCP**: Pair with Falcon for regulated Kalshi cross-venue odds (`Kalshi Markets`), Polymarket historical \
`Candlesticks` & `Market 360` analytics, `Wallet 360` & `F-Score` smart-money profiling of wallets returned by \
`get_market_holders`, and `Social Pulse`.
- **Firecrawl MCP**: Pass primary URLs from `search_sec_filings`, `get_company_submissions`, `search_court_dockets`, \
`get_docket_entries`, `search_court_opinions`, and `search_online_context` to `firecrawl_scrape` (`formats=['markdown']`) \
to read full merger clauses, judicial stays, FDA letters, or scrape RCP polling tables for `analyze_poll_dropout`.\
"""

ENV_FILE = Path(__file__).resolve().parent / ".env"
EARLIEST_DATE = date(1776, 7, 4)  # FRED's documented defaults: full history
LATEST_DATE = date(9999, 12, 31)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

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


def load_sec_settings() -> SecSettings:
    """SEC EDGAR settings from the environment."""
    return SecSettings.from_env(load_env())


def load_ledger_settings() -> LedgerSettings:
    """Epistemic SQLite ledger settings from the environment."""
    return LedgerSettings.from_env(load_env())


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


def _strip_html(text: Any, limit: int = 260) -> str:
    """Remove HTML highlight tags (e.g. CourtListener ``<mark>``) and clip to one line."""
    return _clip(_HTML_TAG_RE.sub("", str(text or "")), limit)


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
    """Renders API and model data as compact pipe-delimited text lines for an LLM client."""

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
    def events(events: tuple[EventSummary, ...]) -> str:
        """One row per Gamma event with open child market count and sample quotes."""
        if not events:
            return "No active Polymarket events match."
        lines = ["event_id | event_slug | title | end_date | negRisk | volume | open_mkts | sample_markets"]
        for e in events:
            sample = "; ".join(e.top_markets) if e.top_markets else "none"
            lines.append(
                f"{e.event_id} | {e.slug} | {_clip(e.title, 120)} | {e.end_date or 'n/a'} | "
                f"{e.neg_risk} | ${e.volume_usd:,.0f} | {e.open_markets} | {sample}"
            )
        return "\n".join(lines)

    @staticmethod
    def resolution_rules(r: ResolutionRules) -> str:
        """Full market + parent event settlement metadata and ambiguity audit flags."""
        neg_extra = f" (negRiskMarketID={r.neg_risk_market_id})" if r.neg_risk_market_id else ""
        lines = [
            f"market {r.market_id} | slug {r.slug} | end {r.end_date or 'n/a'} | negRisk={r.neg_risk}{neg_extra}",
            f"question: {r.question}",
            (
                f"conditionId: {r.condition_id or 'n/a'} | questionID: {r.question_id or 'n/a'} | "
                f"tokens: {' '.join(r.clob_token_ids) or 'n/a'}"
            ),
            (
                f"tick_size: {_num(r.tick_size, 'g')} | min_order_size: {_num(r.min_order_size, 'g')} | "
                f"uma_statuses: {', '.join(r.uma_resolution_statuses) or 'none'} | "
                f"source: {r.resolution_source or 'UNSPECIFIED'}"
            ),
        ]
        if r.event_slug or r.event_title:
            lines.append(f"parent_event: {r.event_slug or 'n/a'} — {_clip(r.event_title, 180)}")
        lines.append(f"rules: {_clip(r.description, 4000) or 'NONE'}")
        if r.event_description:
            lines.append(f"event_rules: {_clip(r.event_description, 2000)}")
        if r.ambiguity_flags:
            lines.append(f"AMBIGUITY FLAGS ({len(r.ambiguity_flags)}):")
            lines.extend(f"- {flag}" for flag in r.ambiguity_flags)
        else:
            lines.append("AMBIGUITY FLAGS: none detected")
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
        """Fill line, edge/Kelly line, and optional horizon capital-efficiency carry line."""
        lines = [
            TextFormatter.fill(r.fill),
            f"fair {r.fair_prob:.4f} | edge {r.edge:+.4f} | kelly {r.kelly_full:.4f} | kelly_shrunk {r.kelly_shrunk:.4f}",
        ]
        if r.days_to_resolution is not None:
            lines.append(
                f"horizon {r.days_to_resolution:g}d | carry {_num(r.edge_per_day_bps, '+.2f')} bps/d | "
                f"ann_ret {_num(r.annualized_return, '+.2%')} | ann_log_growth {_num(r.annualized_log_growth, '+.2%')}"
            )
        return "\n".join(lines)

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
    def open_orders(orders: tuple[RestingOrder, ...]) -> str:
        """Resting open GTC orders on the CLOB."""
        if not orders:
            return "No open orders."
        lines = [
            f"Open orders ({len(orders)}): order_id | side | outcome | price | orig_sh | matched_sh | rem_usd | status | token_id | condition_id"
        ]
        for o in orders:
            lines.append(
                f"{o.order_id} | {o.side} | {o.outcome} | {o.price:.4f} | {o.original_size:.2f} | "
                f"{o.size_matched:.2f} | ${o.remaining_usd:.2f} | {o.status} | {o.token_id} | {o.condition_id}"
            )
        return "\n".join(lines)

    @staticmethod
    def cancel_order(c: CancelResult) -> str:
        """Order cancellation dry-run validation or live CLOB response."""
        if c.dry_run:
            return f"DRY RUN cancel order {c.order_id}: {c.reason}"
        wallet_note = f" | deposit wallet {c.wallet} (sig type 3)" if c.wallet else ""
        if c.canceled:
            return f"CANCELED: order {c.order_id} | status {c.status}{wallet_note}"
        return f"NOT CANCELED: order {c.order_id} | status {c.status} | reason: {c.reason}{wallet_note}"

    @staticmethod
    def expiry(p: ExpiryPricing, fill_price_no: float | None) -> str:
        """One line per model; NO edge shown when a NO fill price is given."""
        mode_tag = (
            "" if (p.contract_type == "expiry" and p.interval == "1h") else f" | {p.contract_type} ({p.interval})"
        )
        if p.contract_type == "expiry":
            cond = "close > strike" if p.upside else "close < strike"
        else:
            cond = "high >= strike" if p.upside else "low <= strike"
        lines = [
            f"{p.symbol} spot {p.spot:,.2f} | strike {p.strike:,.2f} | r* {p.r_star:+.5f} | horizon {p.horizon_hours}h{mode_tag}",
            f"model | P(YES: {cond}) | P(NO) | sigma_h | n" + (" | edge_NO" if fill_price_no is not None else ""),
        ]
        for e in p.estimates:
            edge = f" | {e.p_no - fill_price_no:+.4f}" if fill_price_no is not None else ""
            lines.append(f"{e.name} | {e.p_yes:.4f} | {e.p_no:.4f} | {_num(e.sigma_h, '.5f')} | {e.samples}{edge}")
        return "\n".join([*lines, f"implied vol: {p.iv_note}"])

    @staticmethod
    def sec_filings(r: SecSearchResult) -> str:
        """Compact SEC EDGAR EFTS full-text search rows with 8-K item tags and archive URLs."""
        ua_note = (
            ["NOTE: SEC_USER_AGENT not set; export SEC_USER_AGENT='YourOrg email@domain.com' in .env (<=10 req/s)."]
            if r.using_default_user_agent
            else []
        )
        if not r.hits:
            return "\n".join([f"No SEC EDGAR filings found for '{r.query}'.", *ua_note])
        lines = [f"SEC EDGAR EFTS '{r.query}' ({r.total_hits} total, showing {len(r.hits)})"]
        for h in r.hits:
            items = ", ".join(h.items) if h.items else "-"
            lines.append(
                f"{h.filed_date} | {h.form} | {_clip(h.entity, 90)} | items: {items} | "
                f"{_clip(h.description, 90)} | {h.document_url}"
            )
        return "\n".join([*lines, *ua_note])

    @staticmethod
    def company_submissions(r: CompanySubmissionsReport) -> str:
        """Recent SEC submissions for a ticker/CIK with 8-K item labels and direct EDGAR URLs."""
        ua_note = (
            ["NOTE: SEC_USER_AGENT not set; export SEC_USER_AGENT='YourOrg email@domain.com' in .env (<=10 req/s)."]
            if r.using_default_user_agent
            else []
        )
        tickers = ", ".join(r.tickers) if r.tickers else "n/a"
        if not r.submissions:
            return "\n".join([f"No matching SEC submissions for {r.entity_name} (CIK {r.cik}).", *ua_note])
        lines = [f"SEC Submissions {r.entity_name} (CIK {r.cik}, tickers: {tickers}) — {len(r.submissions)} filings"]
        for s in r.submissions:
            items = ", ".join(s.items) if s.items else "-"
            lines.append(f"{s.filing_date} | {s.form} | items: {items} | {_clip(s.title, 110)} | {s.document_url}")
        return "\n".join([*lines, *ua_note])

    @staticmethod
    def merger_spread(m: MergerSpreadAnalysis) -> str:
        """Implied M&A deal closing probability, gross spread, break downside, and annualized IRR."""
        lines = [
            (
                f"target ${m.target_price:,.2f} | offer ${m.deal_offer_value:,.2f} | "
                f"break ${m.downside_break_price:,.2f} | implied_close_prob {m.implied_prob:.4f} "
                f"(raw {m.raw_implied_prob:+.4f})"
            ),
            f"gross_spread {m.gross_spread_usd:+,.2f} ({m.gross_spread_pct:+.2%}) | break_downside {m.downside_pct:+.2%}"
            + (
                f" | horizon {m.days_to_close:g}d | ann_simple {_num(m.annualized_simple, '+.2%')} | "
                f"deal_IRR {_num(m.annualized_irr, '+.2%')}"
                if m.days_to_close is not None
                else ""
            ),
        ]
        if m.regime_note:
            lines.append(f"WARNING: {m.regime_note}")
        return "\n".join(lines)

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
    def opinions(query: str, data: dict[str, Any]) -> str:
        """``date | court | docket | citation | case | url`` plus snippet per judicial opinion."""
        results = data.get("results") or []
        if not results:
            return f"No CourtListener opinions for '{query}'."
        lines = [f"CourtListener opinions '{query}' ({data.get('count')} total, showing {len(results)})"]
        for o in results:
            cites = o.get("citation") or []
            cite_str = cites[0] if isinstance(cites, list) and cites else str(cites or "-")
            nested = o.get("opinions") or []
            raw_snip = (
                nested[0].get("snippet")
                if isinstance(nested, list) and nested and isinstance(nested[0], dict)
                else None
            ) or o.get("snippet")
            url = f"https://www.courtlistener.com{o.get('absolute_url') or ''}"
            lines.append(
                f"{o.get('dateFiled') or 'n/a'} | {o.get('court', '')} | {o.get('docketNumber', '')} | "
                f"{cite_str} | {_clip(o.get('caseName'), 140)} | {url}"
            )
            if raw_snip:
                lines.append(f"  snippet: {_strip_html(raw_snip, 260)}")
        return "\n".join(lines)

    @staticmethod
    def docket_entries(data: dict[str, Any]) -> str:
        """Recent orders, stays, injunctions, and RECAP PDF links on a CourtListener docket."""
        docket_id = data.get("docket_id")
        results = data.get("results") or []
        if not results:
            return f"No docket entries found for docket {docket_id}."
        lines = [f"CourtListener docket {docket_id} entries ({data.get('count')} total, showing {len(results)})"]
        for e in results:
            num = e.get("entry_number") if e.get("entry_number") is not None else "?"
            lines.append(f"#{num} | {e.get('date_filed') or 'n/a'} | {_clip(e.get('description'), 240)}")
            for doc in (e.get("recap_documents") or [])[:4]:
                if not isinstance(doc, dict):
                    continue
                dnum = doc.get("document_number") or "?"
                att = f"-{doc['attachment_number']}" if doc.get("attachment_number") else ""
                desc = _clip(doc.get("short_description") or doc.get("description") or "document", 120)
                if doc.get("filepath_local") and doc.get("is_available", True):
                    url = f"https://storage.courtlistener.com/{str(doc['filepath_local']).lstrip('/')}"
                elif doc.get("absolute_url"):
                    url = f"https://www.courtlistener.com{doc['absolute_url']}"
                else:
                    url = "unavailable"
                lines.append(f"  doc {dnum}{att}: {desc} | {url}")
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

    @staticmethod
    def forecast_recorded(r: ForecastRecord) -> str:
        """Confirmation line after upserting an open forecast in the ledger."""
        return (
            f"LEDGER UPSERT {r.market_id} ({r.market_slug}) | family={r.family} | side={r.side} | "
            f"mkt={r.market_prob:.4f} -> fair={r.fair_prob:.4f} (±{r.fair_prob_sd:.4f}) | "
            f"rec=${r.recommended_usd:,.2f} | order={r.order_id or 'none'} | status={r.status}"
        )

    @staticmethod
    def active_theses(rows: tuple[ForecastRecord, ...]) -> str:
        """Open theses with probabilities, exposure, order IDs, and invalidation triggers."""
        if not rows:
            return "No active open theses in ledger."
        lines = [
            f"Active theses ({len(rows)} open): market_id | slug | family | side | mkt->fair (±sd) | rec_usd | order_id | horizon | invalidation_triggers"
        ]
        for r in rows:
            triggers = "; ".join(r.invalidation_triggers) if r.invalidation_triggers else "-"
            lines.append(
                f"{r.market_id} | {r.market_slug} | {r.family} | {r.side} | "
                f"{r.market_prob:.3f}->{r.fair_prob:.3f} (±{r.fair_prob_sd:.3f}) | "
                f"${r.recommended_usd:,.2f} | {r.order_id or '-'} | {r.horizon_days:g}d | {triggers}"
            )
            if r.thesis_summary:
                lines.append(f"  thesis: {_clip(r.thesis_summary, 220)}")
        return "\n".join(lines)

    @staticmethod
    def calibration(c: CalibrationReport) -> str:
        """Brier score, Brier skill score, per-family uncertainty multipliers, and decile table."""
        lines: list[str] = []
        if c.resolved_just_now is not None:
            r = c.resolved_just_now
            lines.append(
                f"RESOLVED {r.market_id} ({r.market_slug}) -> outcome={r.outcome} "
                f"(fair={r.fair_prob:.4f}, mkt={r.market_prob:.4f})"
            )
        if c.total_resolved == 0:
            lines.append("No resolved forecasts in ledger yet.")
            return "\n".join(lines)

        lines.append(
            f"Calibration ({c.total_resolved} resolved) | agent_brier {_num(c.agent_brier)} | "
            f"market_brier {_num(c.market_brier)} | brier_skill_score {_num(c.brier_skill_score, '+.4f')}"
        )
        lines.append("family | n | agent_brier | market_brier | skill | suggested_sd_mult")
        for f in c.families:
            lines.append(
                f"{f.family} | {f.count} | {f.agent_brier:.4f} | {f.market_brier:.4f} | "
                f"{_num(f.brier_skill_score, '+.4f')} | {f.suggested_sd_multiplier:.2f}x"
            )
        lines.append("decile | n | mean_fair | actual_win | cal_gap")
        for d in c.deciles:
            lines.append(
                f"{d.label} | {d.count} | {d.mean_fair_prob:.4f} | {d.actual_win_rate:.4f} | {d.calibration_gap:+.4f}"
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
    *,
    sec_settings_loader: Callable[[], SecSettings] = load_sec_settings,
    ledger_settings_loader: Callable[[], LedgerSettings] = load_ledger_settings,
    order_manager: OrderManager | None = None,
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
        sec_settings_loader: SEC EDGAR settings source, called per tool call.
        ledger_settings_loader: SQLite epistemic ledger settings source, called per tool call.
        order_manager: Live CLOB order list/cancel manager; defaults to the official Polymarket SDK.
    """
    server = FastMCP(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    fred = Gateway(fred_settings_loader, FredClient, FredError, transport)
    poly = Gateway(
        poly_settings_loader, lambda s, h: PolyClient(s, h, order_sender, order_manager), PolyError, transport
    )
    crypto = Gateway(crypto_settings_loader, CryptoClient, CryptoError, transport)
    polynode = Gateway(polynode_settings_loader, PolynodeClient, PolynodeError, transport)
    gov = Gateway(gov_settings_loader, GovClient, GovError, transport)
    intel = Gateway(politics_settings_loader, PoliticsIntelClient, PoliticsIntelError, transport)
    sec = Gateway(sec_settings_loader, SecClient, SecError, transport)
    ledger = Gateway(ledger_settings_loader, lambda s, _h: LedgerClient(s), LedgerError, transport)

    # --- FRED Macroeconomics ------------------------------------------------

    @server.tool
    async def get_series(
        series_id: Annotated[str, Field(min_length=1, description='Exact FRED series ID, e.g. "CPIAUCSL", "UNRATE", "DFF".')],
        observation_start: Annotated[date, Field(description="First observation date (YYYY-MM-DD).")] = EARLIEST_DATE,
        observation_end: Annotated[date, Field(description="Last observation date (YYYY-MM-DD).")] = LATEST_DATE,
        units: Annotated[
            Units,
            Field(description="Units transform: lin (levels), chg, ch1, pch, pc1 (YoY %), pca (annualized %), cch, cca, cap, log."),
        ] = "lin",
    ) -> str:
        """Fetch official FRED economic time-series observations (needs FRED_API_KEY; "." missing values skipped).

        Use ``pc1`` for YoY inflation/GDP growth or ``lin`` for unemployment/rate levels when benchmarking
        Polymarket macro thresholds; cross-check regulated US contract pricing on Falcon MCP ``Kalshi Markets``.
        """
        data = await fred.call(lambda c: c.series_observations(series_id, observation_start, observation_end, units))
        return TextFormatter.series(series_id, data)

    @server.tool
    async def search_series(
        search_text: Annotated[str, Field(min_length=1, description='Keyword query, e.g. "core pce", "nonfarm payrolls".')],
        limit: Annotated[int, Field(ge=1, le=SEARCH_LIMIT_MAX, description="Max series returned (1-1000).")] = 10,
    ) -> str:
        """Search FRED macroeconomic series IDs by keyword (needs FRED_API_KEY), then pass the exact ID to ``get_series``."""
        return TextFormatter.search(search_text, await fred.call(lambda c: c.search_series(search_text, limit)))

    # --- Polymarket Discovery, Rules, Sizing & Order Lifecycle --------------

    @server.tool
    async def search_markets(
        query: Annotated[str | None, Field(description="Words that must all appear in the market question or slug.")] = None,
        tag: Annotated[str | None, Field(description='Gamma tag slug, e.g. "politics", "economy", "crypto", "tech".')] = None,
        min_volume_usd: Annotated[float, Field(ge=0, description="Minimum lifetime market volume in USD.")] = 10_000,
        min_days_to_expiry: Annotated[
            float | None, Field(ge=0, description="Exclude short-dated noise ending sooner than N days from now.")
        ] = None,
        max_days_to_expiry: Annotated[
            float | None, Field(gt=0, description="Only markets ending within N days from now.")
        ] = None,
        order: Annotated[
            MarketOrder,
            Field(description="Gamma sort column: volume24hr, volumeNum (lifetime), liquidityNum (depth), or endDate."),
        ] = "volume24hr",
        limit: Annotated[int, Field(ge=1, le=100, description="Max liquid markets to return (1-100).")] = 20,
    ) -> str:
        """Find active, liquid Polymarket binary markets (two-sided book, YES spread <= 10c) with multi-page Gamma scan.

        Quotes are 0..1 probabilities. For niche corporate/regulatory/legal markets outside top-600 volume rankings,
        also call ``search_events(query=...)`` (which hits Gamma ``/public-search``). Always call
        ``get_market_resolution_rules`` before modeling or trading any candidate market.
        """
        now = datetime.now(UTC)
        end_min = now + timedelta(days=min_days_to_expiry) if min_days_to_expiry else now
        end_max = now + timedelta(days=max_days_to_expiry) if max_days_to_expiry else None
        markets = await poly.call(lambda c: c.search_markets(query, tag, min_volume_usd, end_min, end_max, limit, order))
        return TextFormatter.markets(markets)

    @server.tool
    async def search_events(
        query: Annotated[
            str | None,
            Field(description='Full-catalog keyword query for Gamma /public-search, e.g. "antitrust", "acquisition", "FDA".'),
        ] = None,
        tag: Annotated[str | None, Field(description='Optional Gamma tag slug, e.g. "business", "politics", "science".')] = None,
        min_volume_usd: Annotated[float, Field(ge=0, description="Minimum event lifetime volume in USD.")] = 5_000,
        min_days_to_expiry: Annotated[
            float | None, Field(ge=0, description="Minimum days until event endDate.")
        ] = None,
        max_days_to_expiry: Annotated[
            float | None, Field(gt=0, description="Maximum days until event endDate.")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=50, description="Max events returned (1-50).")] = 15,
    ) -> str:
        """Discover Polymarket events across the full Gamma catalog (``/public-search`` when ``query`` is set, else ``/events``).

        Surfaces slow-horizon corporate, regulatory, legal, and multi-outcome neg-risk events hidden by 24h volume
        truncation. Pass returned ``event_slug`` to ``scan_neg_risk_event`` or child ``market_id`` to
        ``get_market_resolution_rules``; pair with Falcon MCP ``Market 360`` / ``Candlesticks`` for historical price paths.
        """
        now = datetime.now(UTC)
        end_min = (now + timedelta(days=min_days_to_expiry)) if min_days_to_expiry is not None else now
        end_max = (now + timedelta(days=max_days_to_expiry)) if max_days_to_expiry is not None else None
        events = await poly.call(lambda c: c.search_events(query, tag, min_volume_usd, end_min, end_max, limit))
        return TextFormatter.events(events)

    @server.tool
    async def get_market_resolution_rules(
        market_id_or_slug: Annotated[
            str,
            Field(min_length=1, description="Numeric Gamma market_id (from search_markets) or market slug."),
        ],
    ) -> str:
        """Fetch exact UMA settlement rules, resolutionSource, conditionId, tick/min size, and ambiguity warning flags.

        MANDATORY gate before pricing or trading any market. Automatically flags timezone cutoffs (UTC vs ET),
        announced-vs-effective policy triggers, intraday touch vs candle close, unnamed consensus sources,
        50-50 ``p3`` fallbacks, and neg-risk 'Other' traps. If ``source`` contains a URL, scrape it via Firecrawl MCP
        ``firecrawl_scrape`` (`formats=['markdown']`) to verify exact oracle definitions.
        """
        rules = await poly.call(lambda c: c.get_market_resolution_rules(market_id_or_slug))
        return TextFormatter.resolution_rules(rules)

    @server.tool
    async def simulate_clob_fill(
        token_id: Annotated[str, Field(min_length=1, description="CLOB outcome token ID (YES or NO token from search_markets).")],
        side: Annotated[Side, Field(description="BUY walks asks (spend USD); SELL walks bids (raise USD).")],
        amount_usd: Annotated[float, Field(gt=0, description="Order notional in USD to simulate against live depth.")],
        fair_prob: Annotated[
            float, Field(gt=0, lt=1, description="Your calibrated probability (0..1) that THIS token pays $1.")
        ],
        fair_prob_sd: Annotated[
            float,
            Field(ge=0, lt=1, description="Epistemic 1-sigma uncertainty on fair_prob (0..1); shrinks Kelly."),
        ] = 0.05,
        days_to_resolution: Annotated[
            float | None,
            Field(gt=0, description="Days until settlement; computes daily carry (bps/d), annualized return & log-growth."),
        ] = None,
    ) -> str:
        """Walk the live CLOB book for VWAP, slippage (bps), shrunk Kelly, and horizon-adjusted annualized carry/IRR.

        Always pass ``days_to_resolution`` on multi-week markets to penalize long capital lockups on small edges,
        and scale ``fair_prob_sd`` by the family's empirical multiplier from ``resolve_and_score_calibration``.
        """
        report = await poly.call(
            lambda c: c.simulate_fill(token_id, side, amount_usd, fair_prob, fair_prob_sd, days_to_resolution)
        )
        return TextFormatter.edge(report)

    @server.tool
    async def scan_neg_risk_event(
        event_slug: Annotated[str, Field(min_length=1, description="Gamma event slug (from search_events).")],
        size_usd: Annotated[float, Field(gt=0, description="USD notional walked per outcome leg on live asks.")] = 50.0,
    ) -> str:
        """Price buying every YES (pays $1) and every NO (pays $(n-1)) at real CLOB depth; flag structural bundle arbs.

        Before acting on a neg-risk bundle edge, inspect ``get_market_resolution_rules`` on child legs for unnamed
        'Other' / placeholder outcomes that break mutually-exclusive completeness.
        """
        return TextFormatter.bundle(await poly.call(lambda c: c.scan_neg_risk(event_slug, size_usd)))

    @server.tool
    async def place_order(
        market_id: Annotated[str, Field(min_length=1, description="Numeric Gamma market_id (from search_markets).")],
        token_id: Annotated[str, Field(min_length=1, description="Outcome token ID belonging to market_id.")],
        side: Annotated[Side, Field(description="BUY or SELL.")],
        amount_usd: Annotated[float, Field(gt=0, description="Order notional in USD; hard-capped by POLY_MAX_ORDER_USD.")],
        limit_price: Annotated[float, Field(gt=0, lt=1, description="GTC limit price (0..1) aligned to market tick_size.")],
        dry_run: Annotated[
            bool, Field(description="Defaults to True (validate only). Set False only after dry_run=True succeeds.")
        ] = True,
        reasoning: Annotated[str, Field(description="Concise thesis & edge summary; echoed in audit logs.")] = "",
    ) -> str:
        """Validate (``dry_run=True``) or sign and post (``dry_run=False``) a GTC limit order via Deposit Wallet (sig type 3).

        Enforces ``POLY_MAX_ORDER_USD``, tick size, min order size, and market-token ownership before touching keys.
        Record the thesis via ``record_forecast`` BEFORE sending live orders, and prefer passive maker prices near
        ``best_bid`` / ``best_ask`` on slow-horizon markets rather crossing wide spreads.
        """
        check = await poly.call(
            lambda c: c.place_order(market_id, token_id, side, amount_usd, limit_price, dry_run, reasoning)
        )
        return TextFormatter.order(check)

    @server.tool
    async def list_open_orders(
        market_id: Annotated[
            str | None,
            Field(description="Optional Gamma market_id, market slug, or 0x conditionId to filter resting orders."),
        ] = None,
    ) -> str:
        """List open resting GTC orders on the Deposit Wallet (needs POLY_PRIVATE_KEY; enforces sig type 3).

        Pair with ``get_active_theses`` during scheduled sweeps to reconcile live resting maker exposure against
        invalidated theses before calling ``cancel_order``.
        """
        orders = await poly.call(lambda c: c.list_open_orders(market_id))
        return TextFormatter.open_orders(orders)

    @server.tool
    async def cancel_order(
        order_id: Annotated[str, Field(min_length=1, description="CLOB order_id from place_order or list_open_orders.")],
        dry_run: Annotated[
            bool, Field(description="Defaults to True (validate only); set False to cancel on the CLOB.")
        ] = True,
    ) -> str:
        """Validate (``dry_run=True``) or cancel (``dry_run=False``) a resting GTC order on the Deposit Wallet (sig type 3).

        Use immediately when an ``invalidation_trigger`` from ``get_active_theses`` fires so stale maker quotes are
        not picked off by faster news-reacting takers.
        """
        result = await poly.call(lambda c: c.cancel_order(order_id, dry_run))
        return TextFormatter.cancel_order(result)

    # --- Long-Dated Crypto Expiry & One-Touch Barrier Pricing ---------------

    @server.tool
    async def price_crypto_expiry(
        symbol: Annotated[
            str, Field(pattern=r"^[A-Z0-9]{5,20}$", description='Binance spot pair, e.g. "BTCUSDT", "ETHUSDT", "SOLUSDT".')
        ],
        strike: Annotated[float, Field(gt=0, description="Target/barrier price from get_market_resolution_rules.")],
        horizon_hours: Annotated[
            int,
            Field(
                ge=1,
                le=MAX_HORIZON_HOURS,
                description="Hours until settlement (1..8760). <=168h uses 1h klines; >168h uses 1000d 1d klines.",
            ),
        ],
        contract_type: Annotated[
            ContractType,
            Field(
                description="'expiry' for terminal close P(S_T > K) / P(S_T < K); 'touch' for one-touch barrier "
                "P(max S_t >= K) / P(min S_t <= K) ('Will BTC hit/dip to $K before date?')."
            ),
        ] = "expiry",
        fill_price_no: Annotated[
            float | None, Field(gt=0, lt=1, description="Optional NO fill price (0..1); adds edge_NO = P(NO) - price.")
        ] = None,
    ) -> str:
        """Price crypto terminal-expiry or one-touch barrier contracts up to 365 days (Binance OHLCV + Deribit IV/skew).

        CRITICAL: Long-dated Polymarket crypto markets ('Will BTC hit $150k by Dec 31?') are almost always ``touch``
        barriers (~2x higher probability than ``expiry`` under zero-drift reflection!). Always verify ``touch`` vs
        ``expiry`` and reference pair via ``get_market_resolution_rules`` first.
        """
        pricing = await crypto.call(lambda c: c.price_expiry(symbol, strike, horizon_hours, contract_type))
        return TextFormatter.expiry(pricing, fill_price_no)

    # --- SEC EDGAR Corporate Filings & M&A Math -----------------------------

    @server.tool
    async def search_sec_filings(
        query: Annotated[
            str,
            Field(
                min_length=1,
                description='EDGAR full-text query, e.g. \'"second request" merger\', \'"Wells notice"\', \'"termination fee"\'.',
            ),
        ],
        forms: Annotated[
            tuple[str, ...],
            Field(description='SEC form filter, e.g. ["8-K", "S-4", "DEFM14A", "13D", "10-Q", "10-K"].'),
        ] = DEFAULT_SEARCH_FORMS,
        start_date: Annotated[
            date | None, Field(description="Optional earliest filing date (YYYY-MM-DD).")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=SEC_SEARCH_MAX, description="Max filings returned (1-50).")] = 10,
    ) -> str:
        """Search SEC EDGAR EFTS full-text index for corporate/M&A/regulatory catalysts (no key; <=10 req/s; SEC_USER_AGENT).

        Flags 8-K item codes (1.01 Definitive Agreement, 1.02 Termination, 5.02 Officer Change, 8.01 Other Events)
        and returns direct ``sec.gov/Archives`` URLs—pass those URLs to Firecrawl MCP ``firecrawl_scrape``
        (`formats=['markdown']`) to read full merger outside-dates, reverse termination fees, or DOJ/FTC disclosures.
        """
        result = await sec.call(lambda c: c.search_filings(query, forms, start_date, limit))
        return TextFormatter.sec_filings(result)

    @server.tool
    async def get_company_submissions(
        ticker_or_cik: Annotated[
            str, Field(min_length=1, description='Public equity ticker (e.g. "AAPL", "NVDA") or 10-digit SEC CIK.')
        ],
        forms: Annotated[
            tuple[str, ...],
            Field(description='SEC form filter (includes /A amendments), e.g. ["8-K", "10-Q", "10-K", "S-4"].'),
        ] = DEFAULT_SUBMISSION_FORMS,
        limit: Annotated[int, Field(ge=1, le=SEC_SUBMISSIONS_MAX, description="Max recent filings (1-50).")] = 15,
    ) -> str:
        """List recent SEC submissions for a company ticker or CIK with 8-K item descriptions and direct EDGAR URLs.

        No API key needed (uses ``SEC_USER_AGENT``, <=10 req/s). Pass returned primary document URLs to Firecrawl MCP
        ``firecrawl_scrape`` (`formats=['markdown']`) when inspecting earnings guidance, S-4 risk factors, or 8-K exhibits.
        """
        report = await sec.call(lambda c: c.company_submissions(ticker_or_cik, forms, limit))
        return TextFormatter.company_submissions(report)

    @server.tool
    async def calculate_merger_spread_prob(
        target_price: Annotated[float, Field(gt=0, description="Current live share price of the M&A target in USD.")],
        deal_offer_value: Annotated[float, Field(gt=0, description="Deal consideration value per target share in USD.")],
        downside_break_price: Annotated[
            float, Field(ge=0, description="Estimated standalone target price if the deal breaks (pre-rumor / peer multiple).")
        ],
        days_to_close: Annotated[
            float | None, Field(gt=0, description="Expected days to deal completion/outside-date for annualized IRR.")
        ] = None,
    ) -> str:
        """Compute equity-market-implied M&A closing probability ``(P - Break) / (Offer - Break)``, gross spread, and deal IRR.

        Pure deterministic math (no HTTP). Benchmark ``implied_close_prob`` directly against Polymarket M&A YES prices
        after checking S-4 / 8-K outside-dates via ``get_company_submissions`` + Firecrawl ``firecrawl_scrape``.
        """
        try:
            analysis = MergerMath.analyze(target_price, deal_offer_value, downside_break_price, days_to_close)
        except SecError as exc:
            raise ToolError(str(exc)) from exc
        return TextFormatter.merger_spread(analysis)

    # --- PolyNode: X, web context, holder concentration ---------------------

    @server.tool
    async def search_x_context(
        query: Annotated[str, Field(min_length=1, description='X search query; operators work, e.g. "from:AtlasIntel".')],
        max_results: Annotated[int, Field(ge=1, le=X_SEARCH_MAX, description="Posts to return (1-50).")] = 15,
        since: Annotated[date | None, Field(description="Only posts on/after this date (YYYY-MM-DD).")] = None,
        until: Annotated[date | None, Field(description="Only posts before this date (YYYY-MM-DD).")] = None,
    ) -> str:
        """Latest X posts from primary sources/journalists/pollsters via PolyNode (paid monthly quota; max 1 req/s).

        Conserve quota by checking primary gov/SEC/FRED tools first; also pair with Falcon MCP ``Social Pulse`` when
        monitoring broad market sentiment shifts.
        """
        data = await polynode.call(lambda c: c.x_search(query, max_results, since, until))
        return TextFormatter.x_search(query, data)

    @server.tool
    async def search_online_context(
        query: Annotated[str, Field(min_length=1, description="Web/news search query.")],
        max_results: Annotated[int, Field(ge=1, le=ONLINE_SEARCH_MAX, description="Results (1-20).")] = 10,
        time_range: Annotated[TimeRange | None, Field(description="Freshness filter: day, week, month, year.")] = None,
    ) -> str:
        """Corroborating web/news search via PolyNode Online Search (needs POLYNODE_API_KEY).

        For full-page tables, PDFs, or deep document extraction, pass discovered URLs to Firecrawl MCP
        ``firecrawl_scrape`` (`formats=['markdown']`) or use ``firecrawl_search``.
        """
        data = await polynode.call(lambda c: c.online_search(query, max_results, time_range))
        return TextFormatter.online_search(query, data)

    @server.tool
    async def get_market_holders(
        market_slug: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$", description="Polymarket market slug.")],
        limit: Annotated[int, Field(ge=1, le=POSITIONS_MAX, description="Largest positions to fetch (1-300).")] = 100,
        top_n: Annotated[int, Field(ge=1, le=50, description="Top-N wallets for concentration share and rows shown.")] = 10,
        status: Annotated[
            HolderStatus | None, Field(description="open = current holders, closed = exited, null = both.")
        ] = "open",
    ) -> str:
        """Whale positioning: largest holders, share-weighted avg entry (0..1), top-N share, and HHI per token (needs POLYNODE_API_KEY).

        Routing hint: Pair returned top ``wallet`` addresses with Falcon MCP ``Wallet 360`` / ``F-Score`` leaderboards
        to verify whether concentrated holders are historically profitable sharps or uninformed whales.
        """
        report = await polynode.call(lambda c: c.market_positions(market_slug, limit, top_n, status))
        return TextFormatter.holders(report, top_n)

    # --- Government and legal oracles ---------------------------------------

    @server.tool
    async def search_federal_register(
        term: Annotated[str, Field(min_length=1, description='Full-text term, e.g. "section 301 tariff", "semiconductor".')],
        include_public_inspection: Annotated[
            bool, Field(description="Include filed-but-unpublished Public Inspection desk documents (early signal).")
        ] = True,
        limit: Annotated[int, Field(ge=1, le=100, description="Max documents per section (1-100).")] = 10,
    ) -> str:
        """Search published and pending Public-Inspection executive orders, tariffs, and agency rules (keyless).

        Public Inspection filings often appear hours/days before formal publication—critical for 'enacted/published by'
        vs 'announced by' resolution rules audited via ``get_market_resolution_rules``.
        """
        result = await gov.call(lambda c: c.federal_register_search(term, include_public_inspection, limit))
        return TextFormatter.federal_register(term, result)

    @server.tool
    async def get_bill_status(
        congress: Annotated[int, Field(ge=93, le=200, description="Congress number, e.g. 119.")],
        bill_type: Annotated[BillType, Field(description="Chamber/type: hr, s, hjres, sjres, hconres, sconres, hres, sres.")],
        bill_number: Annotated[int, Field(ge=1, description="Bill number, e.g. 1 for H.R. 1.")],
    ) -> str:
        """Official legislative status, sponsor, and latest action of a US bill (Congress.gov; needs CONGRESS_API_KEY)."""
        return TextFormatter.bill(await gov.call(lambda c: c.congress_bill_status(congress, bill_type, bill_number)))

    @server.tool
    async def get_nomination_status(
        congress: Annotated[int, Field(ge=93, le=200, description="Congress number, e.g. 119.")],
        nomination_number: Annotated[int, Field(ge=1, description="PN number, e.g. 12 for PN12.")],
    ) -> str:
        """Official Senate committee/floor status of a presidential nomination (Congress.gov; needs CONGRESS_API_KEY)."""
        data = await gov.call(lambda c: c.congress_nomination(congress, nomination_number))
        return TextFormatter.nomination(data)

    @server.tool
    async def get_candidate_financials(
        query: Annotated[str, Field(min_length=2, description='Candidate surname or state query, e.g. "ossoff".')],
        cycle: Annotated[int, Field(ge=1980, le=2100, description="Two-year FEC election cycle, e.g. 2026 or 2028.")] = 2026,
        limit: Annotated[int, Field(ge=1, le=100, description="Max candidates returned (1-100).")] = 10,
    ) -> str:
        """FEC receipts, disbursements, burn rate, cash-on-hand, and campaign debt (uses FEC_API_KEY or DEMO_KEY).

        High burn (>90%) + low cash-on-hand + rising committee debt is a leading primary drop-out indicator weeks before
        formal suspension announcements.
        """
        rows = await gov.call(lambda c: c.fec_candidate_financials(query, cycle, limit))
        return TextFormatter.candidates(query, cycle, rows)

    @server.tool
    async def search_court_dockets(
        query: Annotated[str, Field(min_length=1, description='RECAP docket query, e.g. "tariff injunction", "FTC v.". ')],
        limit: Annotated[int, Field(ge=1, le=20, description="Max dockets returned (1-20).")] = 10,
    ) -> str:
        """Search federal RECAP court dockets, newest filing first (CourtListener v4; needs COURTLISTENER_API_TOKEN).

        Routing hint: Extract the numeric ``docket_id`` from ``/docket/<docket_id>/...`` and call ``get_docket_entries``
        to inspect actual orders, stays, and injunctions, or pass URLs to Firecrawl MCP ``firecrawl_scrape``.
        """
        return TextFormatter.dockets(query, await gov.call(lambda c: c.courtlistener_dockets(query, limit)))

    @server.tool
    async def search_court_opinions(
        query: Annotated[
            str, Field(min_length=1, description='Judicial opinion query, e.g. "preliminary injunction tariff", "Chevron".')
        ],
        court: Annotated[
            str | None,
            Field(description='Optional CourtListener court slug, e.g. "scotus", "cadc", "ca5", "cit", "dcd".'),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=20, description="Max opinions returned (1-20).")] = 10,
    ) -> str:
        """Search federal appellate, district, and SCOTUS judicial opinions (CourtListener v4 ``type=o``; needs COURTLISTENER_API_TOKEN).

        Routing hint: Pass returned ``courtlistener.com/opinion/...`` URLs to Firecrawl MCP ``firecrawl_scrape``
        (`formats=['markdown']`) to read full holdings, remedies, and stay conditions.
        """
        return TextFormatter.opinions(query, await gov.call(lambda c: c.courtlistener_opinions(query, court, limit)))

    @server.tool
    async def get_docket_entries(
        docket_id: Annotated[
            int, Field(ge=1, description="Numeric CourtListener docket_id (from /docket/<id>/ in search_court_dockets).")
        ],
        limit: Annotated[int, Field(ge=1, le=50, description="Max recent docket entries returned (1-50).")] = 15,
    ) -> str:
        """Fetch recent filings, orders, briefing schedules, and RECAP PDF URLs on a federal docket (needs COURTLISTENER_API_TOKEN).

        Routing hint: Pass returned ``storage.courtlistener.com`` PDF or document URLs to Firecrawl MCP
        ``firecrawl_scrape`` (`formats=['markdown']`) to read exact judicial order text rather than guessing from titles.
        """
        return TextFormatter.docket_entries(await gov.call(lambda c: c.courtlistener_docket_entries(docket_id, limit)))

    # --- News volume, Fed, cross-venue odds, polling ------------------------

    @server.tool
    async def search_gdelt_news(
        query: Annotated[str, Field(min_length=1, description='GDELT query, e.g. "tariff sourcecountry:US".')],
        timespan: Annotated[str, Field(pattern=r"^\d{1,3}(min|h|d|w|m)$", description="Window: e.g. 15min, 24h, 7d, 1m.")] = "24h",
        limit: Annotated[int, Field(ge=1, le=GDELT_MAX_RECORDS, description="Max articles (1-250).")] = 15,
    ) -> str:
        """Newest global and regional news articles from GDELT Doc 2.0 (keyless; strict rate limit: max 1 req / 5s).

        Space GDELT calls >=5 seconds apart; pass high-signal article URLs to Firecrawl MCP ``firecrawl_scrape`` for full text.
        """
        return TextFormatter.gdelt_articles(query, await intel.call(lambda c: c.gdelt_articles(query, timespan, limit)))

    @server.tool
    async def get_gdelt_volume(
        query: Annotated[str, Field(min_length=1, description="GDELT attention query.")],
        timespan: Annotated[str, Field(pattern=r"^\d{1,3}(min|h|d|w|m)$", description="Window: e.g. 24h, 3d, 2w.")] = "24h",
    ) -> str:
        """Media-attention share timeline (% of monitored global coverage; keyless; max 1 req / 5s across GDELT tools)."""
        return TextFormatter.gdelt_volume(query, await intel.call(lambda c: c.gdelt_timeline(query, timespan)))

    @server.tool
    async def get_fed_releases(
        limit: Annotated[int, Field(ge=1, le=50, description="Max releases (1-50).")] = 10,
    ) -> str:
        """Latest official Federal Reserve press releases (FOMC statements, rate decisions, enforcement) from Fed RSS (keyless)."""
        return TextFormatter.fed_releases(await intel.call(lambda c: c.fed_press_releases(limit)))

    @server.tool
    async def search_predictit(
        keyword: Annotated[str, Field(min_length=1, description="Words that must all appear in the PredictIt market name.")],
    ) -> str:
        """PredictIt contract prices (0..1 USD) for cross-venue political benchmarks (keyless; note $850 cap & 10% profit fee).

        Routing hint: Also check Falcon MCP ``Kalshi Markets`` when available for deeper regulated US cross-venue liquidity.
        """
        return TextFormatter.predictit(keyword, await intel.call(lambda c: c.predictit_markets(keyword)))

    @server.tool
    async def get_bookmaker_odds(
        sport_key: Annotated[
            str, Field(pattern=r"^[a-z0-9_]+$", description="The Odds API politics sport key.")
        ] = DEFAULT_ODDS_SPORT,
        regions: Annotated[
            str, Field(pattern=r"^(us|us2|uk|eu|au)(,(us|us2|uk|eu|au))*$", description="Comma-separated book regions.")
        ] = "us,uk,eu",
    ) -> str:
        """De-vigged bookmaker consensus probability (0..1) for politics outrights (needs THE_ODDS_API_KEY; 500 credits/mo free tier).

        Routing hint: Combine no-vig sportsbook consensus with Falcon MCP ``Kalshi Markets`` and ``search_predictit``
        to triangulate fair cross-venue political odds.
        """
        return TextFormatter.bookmaker_odds(
            sport_key, await intel.call(lambda c: c.bookmaker_politics_odds(sport_key, regions))
        )

    @server.tool
    async def analyze_poll_dropout(
        polls: Annotated[
            list[Poll],
            Field(min_length=2, description="Every poll currently in the rolling average: pollster, end_date (YYYY-MM-DD), value."),
        ],
    ) -> str:
        """Deterministic rolling-average dropout calculator (e.g., RealClearPolitics approval/matchup brackets).

        Pure math (no HTTP). Scrape the live RCP/538 polling table first via Firecrawl MCP ``firecrawl_scrape``
        (`formats=['markdown']` or structured JSON), then pass every included row here to see the exact mechanical
        shift when the oldest poll ages out.
        """
        try:  # pure math: no settings or HTTP, so no gateway
            analysis = PollDropoutCalculator.analyze(polls)
        except PoliticsIntelError as exc:
            raise ToolError(str(exc)) from exc
        return TextFormatter.poll_dropout(analysis)

    # --- Self-Improving Epistemic Brier Ledger ------------------------------

    @server.tool
    async def record_forecast(
        market_id: Annotated[str, Field(min_length=1, description="Polymarket market_id (primary key for open thesis upsert).")],
        market_slug: Annotated[str, Field(min_length=1, description="Polymarket market slug.")],
        family: Annotated[
            MarketFamily,
            Field(description="Market domain family: macro, crypto_long, corporate, regulatory_legal, politics, or tech."),
        ],
        horizon_days: Annotated[float, Field(gt=0, description="Expected days until market settlement.")],
        market_prob: Annotated[
            float, Field(gt=0, lt=1, description="Current Polymarket YES implied probability (0..1) at forecast time.")
        ],
        fair_prob: Annotated[
            float, Field(gt=0, lt=1, description="Agent's calibrated YES probability (0..1), regardless of trade side.")
        ],
        fair_prob_sd: Annotated[
            float, Field(ge=0, lt=1, description="1-sigma epistemic uncertainty on fair_prob (0..1).")
        ] = 0.05,
        side: Annotated[
            ForecastSide, Field(description="Recommended action: YES, NO, BUY, SELL, or PASS (logged for calibration).")
        ] = "PASS",
        recommended_usd: Annotated[float, Field(ge=0, description="Kelly-sized notional USD (0 if PASS).")] = 0.0,
        order_id: Annotated[str | None, Field(description="CLOB order_id once a live GTC order is placed.")] = None,
        thesis_summary: Annotated[str, Field(description="Concise causal thesis and base-rate anchor.")] = "",
        key_sources: Annotated[
            list[str] | str, Field(description="Primary oracle URLs/IDs supporting the forecast (list or ';'-delimited).")
        ] = "",
        invalidation_triggers: Annotated[
            list[str] | str,
            Field(description="Concrete observable events that invalidate this thesis and require canceling open orders."),
        ] = "",
    ) -> str:
        """Upsert a structured forecast and invalidation triggers into the SQLite epistemic ledger (QUANT_LEDGER_PATH).

        Always store ``market_prob`` and ``fair_prob`` as **YES probabilities (0..1)** so ``resolve_and_score_calibration``
        computes exact Brier scores against ``outcome`` (1=YES, 0=NO). Settled markets cannot be overwritten.
        """
        async def op(c: LedgerClient) -> ForecastRecord:
            return c.record_forecast(
                market_id=market_id,
                market_slug=market_slug,
                family=family,
                horizon_days=horizon_days,
                market_prob=market_prob,
                fair_prob=fair_prob,
                fair_prob_sd=fair_prob_sd,
                side=side,
                recommended_usd=recommended_usd,
                order_id=order_id,
                thesis_summary=thesis_summary,
                key_sources=key_sources,
                invalidation_triggers=invalidation_triggers,
            )

        return TextFormatter.forecast_recorded(await ledger.call(op))

    @server.tool
    async def get_active_theses(
        market_id: Annotated[str | None, Field(description="Optional market_id filter.")] = None,
        family: Annotated[MarketFamily | None, Field(description="Optional market family filter.")] = None,
    ) -> str:
        """List open/unresolved theses, exposure, resting order IDs, and invalidation triggers from the epistemic ledger.

        Call FIRST before researching a market (to avoid duplicate orders) and at the start of every scheduled sweep
        (pairing with ``list_open_orders`` and ``cancel_order`` if an invalidation trigger has fired).
        """
        async def op(c: LedgerClient) -> tuple[ForecastRecord, ...]:
            return c.get_active_theses(market_id=market_id, family=family)

        return TextFormatter.active_theses(await ledger.call(op))

    @server.tool
    async def resolve_and_score_calibration(
        market_id: Annotated[
            str | None, Field(description="Market ID to mark settled (omit together with outcome to read current scorecard).")
        ] = None,
        outcome: Annotated[
            int | None, Field(ge=0, le=1, description="Settled YES outcome: 1 if YES resolved true, 0 if NO.")
        ] = None,
        family: Annotated[
            MarketFamily | None, Field(description="Optional family filter when reading calibration scores.")
        ] = None,
    ) -> str:
        """Mark a settled market (``outcome`` 0 or 1) and compute Brier Score, Brier Skill Score, deciles, and ``fair_prob_sd`` multipliers.

        Omit ``market_id`` and ``outcome`` to inspect your historical calibration table before sizing a new trade, and
        multiply your baseline ``fair_prob_sd`` in ``simulate_clob_fill`` by the family's ``suggested_sd_mult``.
        """
        async def op(c: LedgerClient) -> CalibrationReport:
            return c.resolve_and_score(market_id=market_id, outcome=outcome, family=family)

        return TextFormatter.calibration(await ledger.call(op))

    return server


mcp = create_server()


def main() -> None:
    """Run the server over stdio. Logs go to stderr; stdout carries MCP frames."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
