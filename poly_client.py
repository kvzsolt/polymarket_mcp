"""Polymarket client: Gamma market discovery, resolution rules, CLOB books, fill math, orders.

No MCP code here. ``OrderBookMath`` and ``ResolutionRuleAuditor`` are pure and
deterministic; ``PolyClient`` does the HTTP for market data. Live orders and
authenticated order management go through the official SDK (``polymarket-client``,
``AsyncSecureClient``), strictly enforcing the signer's Deposit Wallet (signature
type 3), behind injectable ``OrderSender`` and ``OrderManager`` protocols so
offline tests never touch the SDK.

References:
    https://docs.polymarket.com/api-reference/market-data/get-order-book
    https://github.com/Polymarket/py-sdk (polymarket-client 0.16.0)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx2 as httpx

if TYPE_CHECKING:
    from polymarket import AsyncSecureClient

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
MAX_SPREAD = 0.10  # wider YES spreads are treated as illiquid noise
GAMMA_FETCH = 200  # candidates fetched per Gamma page before client-side filtering
MAX_GAMMA_PAGES = 3  # up to 600 markets scanned before stopping

Side = Literal["BUY", "SELL"]
MarketOrder = Literal["volume24hr", "volumeNum", "liquidityNum", "endDate"]

_CONDITION_ID_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")
_TZ_RE = re.compile(r"\b(?:UTC|GMT|EST|EDT|ET|PST|PDT|PT|CST|CDT|CT|Eastern Time|Pacific Time)\b", re.IGNORECASE)
_DEADLINE_RE = re.compile(r"\b(?:by|before|on|prior to|through|until|end of)\b", re.IGNORECASE)
_POLICY_SUBJECT_RE = re.compile(
    r"\b(?:tariff|bill|law|act|rule|regulation|merger|acqui(?:re|sition)|deal|resign|fire|appoint|nominat|sanction|executive order)\b",
    re.IGNORECASE,
)
_ANNOUNCE_RE = re.compile(r"\b(?:announc\w*|agree\w*|propos\w*|sign\w*)\b", re.IGNORECASE)
_EFFECTIVE_RE = re.compile(r"\b(?:enact\w*|effective|implement\w*|complet\w*|clos(?:e|ed|ing)|confirm\w*|take effect)\b", re.IGNORECASE)
_PRICE_BARRIER_RE = re.compile(r"(?:\$\s*\d|\b(?:hit|reach|touch|dip|above|below|exceed|trade at)\b)", re.IGNORECASE)
_TOUCH_CUE_RE = re.compile(r"\b(?:1\s*-?\s*m(?:in(?:ute)?)?|high|low|intraday|any\s+point|touch\w*|wick)\b", re.IGNORECASE)
_CLOSE_CUE_RE = re.compile(r"\b(?:daily close|closing price|candle close|closes above|closes below|settlement price)\b", re.IGNORECASE)
_CONSENSUS_RE = re.compile(r"\b(?:consensus of credible|credible reporting|credible media|major news)\b", re.IGNORECASE)
_FIFTY_FIFTY_RE = re.compile(r"(?:\b50\s*[-/]\s*50\b|\b0\.50\b|\bp3\b|\bresolve\s+to\s+50%\b)", re.IGNORECASE)
_OTHER_OUTCOME_RE = re.compile(r"\b(?:other|none of the above|another candidate|anyone else)\b", re.IGNORECASE)


class PolyError(Exception):
    """Any Polymarket failure: config, HTTP, bad book, or a refused order."""


@dataclass(frozen=True)
class PolySettings:
    """Polymarket settings. Secrets are optional (market data needs none) and hidden from ``repr``.

    Live orders and order management need ``private_key``. ``wallet`` defaults to
    the signer's Deposit Wallet, which the SDK deploys via the relayer on first
    live use. L2 credentials are derived from the key unless all three are supplied.
    """

    max_order_usd: float = 100.0
    private_key: str | None = field(default=None, repr=False)
    wallet: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api_secret: str | None = field(default=None, repr=False)
    api_passphrase: str | None = field(default=None, repr=False)
    gamma_url: str = GAMMA_URL
    clob_url: str = CLOB_URL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> PolySettings:
        """Read ``POLY_MAX_ORDER_USD``, ``POLY_PRIVATE_KEY``, ``POLY_DEPOSIT_WALLET``,
        ``POLY_API_KEY``/``POLY_API_SECRET``/``POLY_API_PASSPHRASE``, ``POLY_GAMMA_URL``, ``POLY_CLOB_URL``.

        Raises:
            PolyError: If the cap is not a positive number, or L2 credentials are partial.
        """
        raw_cap = env.get("POLY_MAX_ORDER_USD", "100")
        try:
            cap = float(raw_cap)
        except ValueError:
            cap = -1.0
        if not math.isfinite(cap) or cap <= 0:
            raise PolyError(f"POLY_MAX_ORDER_USD must be a positive number, got {raw_cap!r}.")
        creds = [env.get(k) or None for k in ("POLY_API_KEY", "POLY_API_SECRET", "POLY_API_PASSPHRASE")]
        if any(creds) and not all(creds):
            raise PolyError("Set all of POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE, or none (derive).")
        return cls(
            max_order_usd=cap,
            private_key=env.get("POLY_PRIVATE_KEY") or None,
            wallet=env.get("POLY_DEPOSIT_WALLET") or None,
            api_key=creds[0],
            api_secret=creds[1],
            api_passphrase=creds[2],
            gamma_url=env.get("POLY_GAMMA_URL", GAMMA_URL),
            clob_url=env.get("POLY_CLOB_URL", CLOB_URL),
        )


@dataclass(frozen=True)
class Level:
    """One price level: ``size`` shares at ``price`` (0..1)."""

    price: float
    size: float


@dataclass(frozen=True)
class Book:
    """Order book with both sides sorted best-first."""

    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    tick_size: float | None = None
    min_order_size: float | None = None

    def side(self, side: Side) -> tuple[Level, ...]:
        """Levels a taker on ``side`` would hit: asks to BUY, bids to SELL."""
        return self.asks if side == "BUY" else self.bids


@dataclass(frozen=True)
class Fill:
    """Result of walking the book. ``vwap`` is ``None`` when nothing filled."""

    side: Side
    requested_usd: float
    filled_usd: float
    shares: float
    vwap: float | None
    best: float | None
    slippage_bps: float | None
    post_trade_best: float | None  # best price left on that side afterwards

    @property
    def complete(self) -> bool:
        """True if the whole requested notional filled."""
        return self.filled_usd >= self.requested_usd - 1e-9


@dataclass(frozen=True)
class EdgeReport:
    """A fill plus edge, shrunk Kelly, and optional horizon capital-efficiency metrics."""

    fill: Fill
    fair_prob: float
    edge: float
    kelly_full: float
    kelly_shrunk: float
    days_to_resolution: float | None = None
    edge_per_day_bps: float | None = None
    annualized_return: float | None = None
    annualized_log_growth: float | None = None


@dataclass(frozen=True)
class EventSummary:
    """Summary of one Polymarket Gamma event and its open child markets."""

    event_id: str
    slug: str
    title: str
    end_date: str
    neg_risk: bool
    volume_usd: float
    open_markets: int
    top_markets: tuple[str, ...]


@dataclass(frozen=True)
class ResolutionRules:
    """Full resolution metadata and settlement-ambiguity flags for one market."""

    market_id: str
    slug: str
    question: str
    description: str
    event_slug: str | None
    event_title: str | None
    event_description: str | None
    resolution_source: str | None
    end_date: str | None
    neg_risk: bool
    neg_risk_market_id: str | None
    condition_id: str | None
    question_id: str | None
    uma_resolution_statuses: tuple[str, ...]
    tick_size: float | None
    min_order_size: float | None
    clob_token_ids: tuple[str, ...]
    ambiguity_flags: tuple[str, ...]


@dataclass(frozen=True)
class BundleLeg:
    """One outcome of a neg-risk event priced at ``size_usd`` per side."""

    title: str
    yes: Fill
    no: Fill


@dataclass(frozen=True)
class BundleScan:
    """Slippage-adjusted cost of buying every YES, or every NO, in an event."""

    event_slug: str
    neg_risk: bool
    legs: tuple[BundleLeg, ...]
    sum_yes: float | None  # None if any leg could not fully fill
    sum_no: float | None

    @property
    def edge_yes(self) -> float | None:
        """Profit per $1 payout from buying all YES (exactly one pays 1)."""
        return None if self.sum_yes is None else 1.0 - self.sum_yes

    @property
    def edge_no(self) -> float | None:
        """Profit from buying all NO (exactly n-1 pay 1)."""
        return None if self.sum_no is None else len(self.legs) - 1 - self.sum_no


@dataclass(frozen=True)
class OrderResult:
    """Exchange answer to a live order."""

    ok: bool
    order_id: str | None
    status: str  # "live" | "matched" | "delayed", or the rejection code
    message: str = ""
    wallet: str = ""  # wallet the order was signed for (signature type 3)


@dataclass(frozen=True)
class OrderCheck:
    """A validated order, plus the exchange result when it was sent live."""

    market_id: str
    token_id: str
    side: Side
    limit_price: float
    shares: float  # size actually sent: amount_usd / limit_price, floored to 0.01
    marketable: Fill  # the part that crosses at or better than the limit
    reasoning: str
    result: OrderResult | None = None  # None means dry run

    @property
    def notional_usd(self) -> float:
        """USD value of ``shares`` at the limit price."""
        return self.shares * self.limit_price

    @property
    def resting_usd(self) -> float:
        """Notional that would rest on the book as a maker order."""
        return max(0.0, self.notional_usd - self.marketable.filled_usd)


@dataclass(frozen=True)
class RestingOrder:
    """One open resting order on the CLOB."""

    order_id: str
    condition_id: str
    token_id: str
    outcome: str
    side: Side
    price: float
    original_size: float
    size_matched: float
    status: str
    created_at: str

    @property
    def remaining_size(self) -> float:
        """Unfilled share count still resting on the book."""
        return max(0.0, self.original_size - self.size_matched)

    @property
    def remaining_usd(self) -> float:
        """Unfilled notional USD value at the order's limit price."""
        return self.remaining_size * self.price


@dataclass(frozen=True)
class CancelResult:
    """Result of validating (dry_run=True) or executing (dry_run=False) an order cancellation."""

    order_id: str
    dry_run: bool
    canceled: bool
    status: str
    reason: str = ""
    wallet: str = ""


class OrderSender(Protocol):
    """Signs and posts one limit order; the seam that keeps the SDK out of tests."""

    async def __call__(
        self, settings: PolySettings, token_id: str, side: Side, price: float, size: float
    ) -> OrderResult:
        """Post a GTC limit order of ``size`` shares at ``price``."""
        ...


class OrderManager(Protocol):
    """Lists and cancels open orders on the CLOB; injectable seam for offline tests."""

    async def list_open_orders(
        self, settings: PolySettings, condition_id: str | None = None
    ) -> tuple[RestingOrder, ...]:
        """Return open orders for the Deposit Wallet, optionally filtered by ``condition_id``."""
        ...

    async def cancel_order(self, settings: PolySettings, order_id: str) -> CancelResult:
        """Cancel ``order_id`` on the CLOB and return the exchange confirmation."""
        ...


@asynccontextmanager
async def _deposit_wallet_client(settings: PolySettings) -> AsyncIterator[AsyncSecureClient]:
    """Open an authenticated ``AsyncSecureClient`` strictly pinned to ``DEPOSIT_WALLET`` (sig type 3)."""
    if not settings.private_key:
        raise PolyError("POLY_PRIVATE_KEY is required for live CLOB account operations.")
    # Imported lazily: eth-account is heavy and only live account paths need it.
    from polymarket import ApiKeyCreds, AsyncSecureClient, PolymarketError

    creds = None
    if settings.api_key:
        creds = ApiKeyCreds.model_validate(
            {"apiKey": settings.api_key, "secret": settings.api_secret, "passphrase": settings.api_passphrase}
        )
    try:
        client = await AsyncSecureClient.create(
            private_key=settings.private_key, wallet=settings.wallet, credentials=creds
        )
        async with client:
            if client.wallet_type != "DEPOSIT_WALLET":
                raise PolyError(
                    f"wallet {client.wallet} classifies as {client.wallet_type}, not DEPOSIT_WALLET "
                    "(signature type 3); refusing to sign. Fix POLY_DEPOSIT_WALLET or unset it."
                )
            yield client
    except PolymarketError as exc:
        raise PolyError(f"Polymarket SDK {type(exc).__name__}: {exc}") from exc


class SdkOrderSender:
    """``OrderSender`` backed by the official SDK's ``AsyncSecureClient``, Deposit Wallet only."""

    async def __call__(
        self, settings: PolySettings, token_id: str, side: Side, price: float, size: float
    ) -> OrderResult:
        """Sign for the configured (or default) Deposit Wallet and post the order."""
        from polymarket import AcceptedOrder

        async with _deposit_wallet_client(settings) as client:
            wallet = str(client.wallet)
            response = await client.place_limit_order(
                token_id=token_id, price=str(price), size=str(size), side=side
            )
        if isinstance(response, AcceptedOrder):
            return OrderResult(True, response.order_id, response.status, wallet=wallet)
        return OrderResult(False, None, response.code, response.message, wallet=wallet)


class SdkOrderManager:
    """``OrderManager`` backed by ``AsyncSecureClient``, Deposit Wallet (sig type 3) only."""

    async def list_open_orders(
        self, settings: PolySettings, condition_id: str | None = None
    ) -> tuple[RestingOrder, ...]:
        """List open resting orders for the Deposit Wallet."""
        async with _deposit_wallet_client(settings) as client:
            paginator = client.list_open_orders(market=condition_id)
            orders = [
                RestingOrder(
                    order_id=o.id,
                    condition_id=str(o.condition_id),
                    token_id=str(o.asset_id),
                    outcome=o.outcome,
                    side=o.side,
                    price=float(o.price),
                    original_size=float(o.original_size),
                    size_matched=float(o.size_matched),
                    status=o.status,
                    created_at=o.created_at.isoformat(),
                )
                for o in [item async for item in paginator.iter_items()]
            ]
        return tuple(orders)

    async def cancel_order(self, settings: PolySettings, order_id: str) -> CancelResult:
        """Cancel a single open order on the CLOB."""
        async with _deposit_wallet_client(settings) as client:
            wallet = str(client.wallet)
            resp = await client.cancel_order(order_id=order_id)
        canceled = order_id in {str(cid) for cid in resp.canceled}
        reasons = {str(k): str(v) for k, v in resp.not_canceled.items()}
        reason = "" if canceled else (reasons.get(order_id) or next(iter(reasons.values()), "not canceled"))
        return CancelResult(
            order_id=order_id,
            dry_run=False,
            canceled=canceled,
            status="canceled" if canceled else "not_canceled",
            reason=reason,
            wallet=wallet,
        )


class ResolutionRuleAuditor:
    """Pure deterministic heuristics flagging common Polymarket UMA settlement traps."""

    @staticmethod
    def audit(market: Mapping[str, Any], event: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        """Inspect market + parent event metadata and return compact ambiguity warning flags."""
        question = str(market.get("question") or "")
        desc = str(market.get("description") or "")
        ev_desc = str((event or {}).get("description") or "")
        combined = f"{question}\n{desc}\n{ev_desc}".strip()
        source = str(market.get("resolutionSource") or (event or {}).get("resolutionSource") or "").strip()
        end_date = str(market.get("endDate") or (event or {}).get("endDate") or "")
        neg_risk = bool(market.get("negRisk") or (event or {}).get("negRisk"))

        flags: list[str] = []

        has_tz = bool(_TZ_RE.search(combined))
        if (end_date or _DEADLINE_RE.search(combined)) and not has_tz:
            flags.append("TIMEZONE_CUTOFF: deadline stated without explicit timezone (UTC endDate vs ET prose risk)")
        elif end_date.endswith("Z") and re.search(r"\b(?:ET|EST|EDT|Eastern Time)\b", combined, re.IGNORECASE):
            flags.append("TIMEZONE_CUTOFF: Gamma endDate is UTC ('Z') while rules cite Eastern Time; verify exact cutoff")

        if _POLICY_SUBJECT_RE.search(combined):
            has_ann = bool(_ANNOUNCE_RE.search(combined))
            has_eff = bool(_EFFECTIVE_RE.search(combined))
            if has_ann and has_eff:
                flags.append("ANNOUNCED_VS_EFFECTIVE: rules mention both announcement/signing and enactment/closing")
            elif not has_ann and not has_eff:
                flags.append("ANNOUNCED_VS_EFFECTIVE: policy/corporate market lacks explicit announced vs enacted trigger")

        if _PRICE_BARRIER_RE.search(question):
            has_touch = bool(_TOUCH_CUE_RE.search(combined))
            has_close = bool(_CLOSE_CUE_RE.search(combined))
            if has_touch and has_close:
                flags.append("TOUCH_VS_CLOSE: rules reference both intraday high/low touch and candle close")
            elif not has_touch and not has_close:
                flags.append("TOUCH_VS_CLOSE: price target does not specify intraday candle touch vs period close")

        if not source or _CONSENSUS_RE.search(combined) or _CONSENSUS_RE.search(source):
            flags.append("UNNAMED_CONSENSUS_SOURCE: no single primary oracle URL or relies on 'credible reporting'")

        if _FIFTY_FIFTY_RE.search(combined):
            flags.append("P3_50_50_FALLBACK: explicit 50-50 / p3 fallback or tie-split clause present")

        if neg_risk and _OTHER_OUTCOME_RE.search(combined):
            flags.append("NEG_RISK_OTHER_TRAP: negRisk event mentions 'Other'/catch-all outcome; verify complement sum")

        return tuple(flags)


class OrderBookMath:
    """Pure order-book arithmetic: VWAP walk, edge, shrunk Kelly, and horizon carry."""

    @staticmethod
    def walk(levels: tuple[Level, ...], side: Side, amount_usd: float, limit: float | None = None) -> Fill:
        """Spend (BUY) or raise (SELL) ``amount_usd`` against best-first ``levels``.

        Levels worse than ``limit`` are not touched. A thin book yields a
        partial fill (``complete`` is False) rather than an error.
        """
        remaining, spent, shares, post = amount_usd, 0.0, 0.0, None
        for level in levels:
            if limit is not None and (level.price > limit if side == "BUY" else level.price < limit):
                post = level.price
                break
            if remaining <= 1e-12:
                post = level.price
                break
            take = min(remaining, level.price * level.size)
            spent, shares, remaining = spent + take, shares + take / level.price, remaining - take
            if take < level.price * level.size - 1e-12:
                post = level.price  # level only partly consumed
                break
        best = levels[0].price if levels else None
        vwap = spent / shares if shares else None
        slippage = None
        if vwap is not None and best:
            slippage = (vwap - best if side == "BUY" else best - vwap) / best * 10_000
        return Fill(side, amount_usd, spent, shares, vwap, best, slippage, post)

    @staticmethod
    def edge(
        fill: Fill,
        fair_prob: float,
        fair_prob_sd: float,
        days_to_resolution: float | None = None,
    ) -> EdgeReport:
        """Edge of the fill vs ``fair_prob`` (token win probability) and Kelly/carry sizing.

        Kelly for a binary bought at ``p`` with win probability ``q`` is
        ``(q - p) / (1 - p)``; selling at ``p`` is buying the complement at ``1 - p``,
        giving ``(p - q) / p``. The shrunk fraction multiplies by
        ``edge^2 / (edge^2 + sd^2)``, discounting edges that are small relative
        to the uncertainty in ``fair_prob``. When ``days_to_resolution`` is supplied,
        also computes daily edge carry (bps/day), annualized simple return on locked
        collateral, and annualized expected log bankroll growth at ``kelly_shrunk``.
        """
        if fill.vwap is None:
            raise PolyError("Nothing filled, so there is no edge to measure.")
        if days_to_resolution is not None and (not math.isfinite(days_to_resolution) or days_to_resolution <= 0):
            raise PolyError("days_to_resolution must be a positive finite number.")
        p = fill.vwap
        if not (0.0 < p < 1.0):
            raise PolyError(f"fill vwap {p} must lie strictly in (0, 1) for binary Kelly sizing.")
        edge = fair_prob - p if fill.side == "BUY" else p - fair_prob
        full = max(0.0, edge / (1.0 - p) if fill.side == "BUY" else edge / p)
        shrink = edge * edge / (edge * edge + fair_prob_sd * fair_prob_sd) if fair_prob_sd > 0 else 1.0
        shrunk = full * shrink

        if days_to_resolution is None:
            return EdgeReport(fill, fair_prob, edge, full, shrunk)

        cost = p if fill.side == "BUY" else 1.0 - p
        win_prob = fair_prob if fill.side == "BUY" else 1.0 - fair_prob
        carry_bps = (edge * 10_000.0) / days_to_resolution
        ann_ret = (edge / cost) * (365.0 / days_to_resolution)
        if 0.0 < shrunk < 1.0:
            log_g = win_prob * math.log(1.0 + shrunk * (1.0 - cost) / cost) + (1.0 - win_prob) * math.log(1.0 - shrunk)
            ann_log_g = log_g * (365.0 / days_to_resolution)
        else:
            ann_log_g = 0.0
        return EdgeReport(fill, fair_prob, edge, full, shrunk, days_to_resolution, carry_bps, ann_ret, ann_log_g)


def _parse_json_strings(raw: Any) -> tuple[str, ...]:
    """Decode a Gamma field that arrives as either a JSON-encoded list string or a Python list."""
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        try:
            parsed = json.loads(raw)
        except ValueError:
            return (raw,)
        raw = parsed
    if isinstance(raw, list | tuple):
        return tuple(str(x) for x in raw if x is not None and str(x) != "")
    return ()


def _opt_float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_dt(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


class PolyClient:
    """Async Polymarket API client; the injected HTTP client is never closed here."""

    def __init__(
        self,
        settings: PolySettings,
        http: httpx.AsyncClient,
        sender: OrderSender | None = None,
        manager: OrderManager | None = None,
    ) -> None:
        """Create the client.

        Args:
            settings: Polymarket settings.
            http: Open async HTTP client for Gamma/CLOB reads; never closed here.
            sender: Live order sender; defaults to the official SDK.
            manager: Live order manager (list/cancel); defaults to the official SDK.
        """
        self._settings = settings
        self._http = http
        self._sender = sender or SdkOrderSender()
        self._manager = manager or SdkOrderManager()

    async def search_markets(
        self,
        query: str | None,
        tag: str | None,
        min_volume_usd: float,
        end_date_min: datetime,
        end_date_max: datetime | None,
        limit: int,
        order: MarketOrder = "volume24hr",
        max_pages: int = MAX_GAMMA_PAGES,
    ) -> list[dict[str, Any]]:
        """Active, liquid markets matching every word of ``query`` (case-insensitive).

        Paginates up to ``max_pages`` pages of ``GAMMA_FETCH`` rows so slow-horizon
        markets outside today's top-200 24h volume remain discoverable.
        """
        if end_date_max is not None and end_date_min >= end_date_max:
            raise PolyError("min_days_to_expiry must be strictly less than max_days_to_expiry.")
        params: dict[str, str | int | float] = {
            "active": "true",
            "closed": "false",
            "order": order,
            "ascending": "true" if order == "endDate" else "false",
            "limit": GAMMA_FETCH,
            "volume_num_min": min_volume_usd,
            "end_date_min": end_date_min.isoformat(),
        }
        if end_date_max is not None:
            params["end_date_max"] = end_date_max.isoformat()
        if tag:
            params["tag_id"] = (await self._get(f"{self._settings.gamma_url}/tags/slug/{tag}"))["id"]
        words = (query or "").lower().split()
        hits: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for page in range(max(1, max_pages)):
            page_params = {**params, "offset": page * GAMMA_FETCH} if page > 0 else params
            markets = await self._get(f"{self._settings.gamma_url}/markets", page_params)
            if not isinstance(markets, list):
                raise PolyError("Polymarket returned malformed market list.")
            for m in markets:
                mid = str(m.get("id", ""))
                haystack = f"{m.get('question', '')} {m.get('slug', '')}".lower()
                if mid not in seen_ids and self._is_liquid(m) and all(w in haystack for w in words):
                    seen_ids.add(mid)
                    hits.append(m)
                    if len(hits) >= limit:
                        return hits
            if len(markets) < GAMMA_FETCH:
                break
        return hits

    async def search_events(
        self,
        query: str | None,
        tag: str | None,
        min_volume_usd: float,
        end_date_min: datetime | None,
        end_date_max: datetime | None,
        limit: int,
    ) -> tuple[EventSummary, ...]:
        """Discover Polymarket events via ``/public-search`` (when ``query`` is set) or ``/events``.

        Unlike top-200 ``volume24hr`` market scans, ``/public-search`` searches the full
        Gamma catalog server-side, surfacing slow-horizon corporate, legal, and macro events.
        """
        if end_date_min and end_date_max and end_date_min >= end_date_max:
            raise PolyError("min_days_to_expiry must be strictly less than max_days_to_expiry.")
        if query and query.strip():
            params: dict[str, str | int | float] = {
                "q": query.strip(),
                "events_status": "active",
                "keep_closed_markets": 0,
            }
            if tag:
                params["events_tag"] = tag
            body = await self._get(f"{self._settings.gamma_url}/public-search", params)
            raw_events = body.get("events") if isinstance(body, dict) else None
        else:
            params = {
                "active": "true",
                "closed": "false",
                "order": "volume",
                "ascending": "false",
                "limit": max(limit * 2, 50),
                "volume_min": min_volume_usd,
            }
            if end_date_min is not None:
                params["end_date_min"] = end_date_min.isoformat()
            if end_date_max is not None:
                params["end_date_max"] = end_date_max.isoformat()
            if tag:
                params["tag_slug"] = tag
            raw_events = await self._get(f"{self._settings.gamma_url}/events", params)

        if not isinstance(raw_events, list):
            raise PolyError("Polymarket returned malformed events payload.")

        results: list[EventSummary] = []
        for ev in raw_events:
            if not isinstance(ev, dict) or ev.get("closed"):
                continue
            vol = _opt_float(ev.get("volume")) or _opt_float(ev.get("volumeNum")) or 0.0
            if vol < min_volume_usd:
                continue
            ev_end = _parse_iso_dt(ev.get("endDate"))
            if end_date_min is not None and ev_end is not None and ev_end < end_date_min:
                continue
            if end_date_max is not None and ev_end is not None and ev_end > end_date_max:
                continue
            open_mkts = [
                m for m in (ev.get("markets") or []) if isinstance(m, dict) and m.get("active") and not m.get("closed")
            ]
            snippets: list[str] = []
            for m in open_mkts[:3]:
                label = str(m.get("groupItemTitle") or m.get("question") or "?")
                bid, ask = _opt_float(m.get("bestBid")), _opt_float(m.get("bestAsk"))
                quote = f"{bid:.2f}/{ask:.2f}" if bid is not None and ask is not None else "n/a"
                snippets.append(f"{m.get('id', '?')}:{label} ({quote})")
            results.append(
                EventSummary(
                    event_id=str(ev.get("id", "")),
                    slug=str(ev.get("slug", "")),
                    title=str(ev.get("title", "")),
                    end_date=str(ev.get("endDate", ""))[:10],
                    neg_risk=bool(ev.get("negRisk")),
                    volume_usd=vol,
                    open_markets=len(open_mkts),
                    top_markets=tuple(snippets),
                )
            )
            if len(results) >= limit:
                break
        return tuple(results)

    async def get_market_resolution_rules(self, market_id_or_slug: str) -> ResolutionRules:
        """Fetch full market + parent event resolution metadata and run ``ResolutionRuleAuditor``."""
        target = market_id_or_slug.strip()
        if not target:
            raise PolyError("market_id_or_slug cannot be blank.")
        path = f"/markets/{target}" if target.isdigit() else f"/markets/slug/{target}"
        market = await self._get(f"{self._settings.gamma_url}{path}")
        if not isinstance(market, dict):
            raise PolyError(f"Unexpected market payload for {target!r}.")

        event: dict[str, Any] | None = None
        events = market.get("events")
        if isinstance(events, list) and events and isinstance(events[0], dict):
            ev_ref = events[0]
            if ev_ref.get("slug"):
                fetched = await self._get(f"{self._settings.gamma_url}/events/slug/{ev_ref['slug']}")
                event = fetched if isinstance(fetched, dict) else ev_ref
            elif ev_ref.get("id"):
                fetched = await self._get(f"{self._settings.gamma_url}/events/{ev_ref['id']}")
                event = fetched if isinstance(fetched, dict) else ev_ref
            else:
                event = ev_ref

        statuses = list(_parse_json_strings(market.get("umaResolutionStatuses")))
        single_status = str(market.get("umaResolutionStatus") or "").strip()
        if single_status and single_status not in statuses:
            statuses.append(single_status)

        desc = str(market.get("description") or "")
        ev_desc_raw = str((event or {}).get("description") or "")
        ev_desc = ev_desc_raw if ev_desc_raw and ev_desc_raw != desc else None

        return ResolutionRules(
            market_id=str(market.get("id") or ""),
            slug=str(market.get("slug") or ""),
            question=str(market.get("question") or ""),
            description=desc,
            event_slug=str(event.get("slug")) if event and event.get("slug") else None,
            event_title=str(event.get("title")) if event and event.get("title") else None,
            event_description=ev_desc,
            resolution_source=str(market.get("resolutionSource") or (event or {}).get("resolutionSource") or "") or None,
            end_date=str(market.get("endDate") or (event or {}).get("endDate") or "") or None,
            neg_risk=bool(market.get("negRisk") or (event or {}).get("negRisk")),
            neg_risk_market_id=(
                str(market.get("negRiskMarketID") or (event or {}).get("negRiskMarketID") or "") or None
            ),
            condition_id=str(market.get("conditionId") or "") or None,
            question_id=str(market.get("questionID") or "") or None,
            uma_resolution_statuses=tuple(statuses),
            tick_size=_opt_float(market.get("orderPriceMinTickSize")),
            min_order_size=_opt_float(market.get("orderMinSize")),
            clob_token_ids=_parse_json_strings(market.get("clobTokenIds")),
            ambiguity_flags=ResolutionRuleAuditor.audit(market, event),
        )

    async def book(self, token_id: str) -> Book:
        """Order book for one outcome token, both sides sorted best-first."""
        body = await self._get(f"{self._settings.clob_url}/book", {"token_id": token_id})

        def levels(key: str) -> list[Level]:
            return [Level(float(x["price"]), float(x["size"])) for x in body.get(key) or []]

        return Book(
            bids=tuple(sorted(levels("bids"), key=lambda lv: -lv.price)),
            asks=tuple(sorted(levels("asks"), key=lambda lv: lv.price)),
            tick_size=float(body["tick_size"]) if body.get("tick_size") else None,
            min_order_size=float(body["min_order_size"]) if body.get("min_order_size") else None,
        )

    async def simulate_fill(
        self,
        token_id: str,
        side: Side,
        amount_usd: float,
        fair_prob: float,
        fair_prob_sd: float,
        days_to_resolution: float | None = None,
    ) -> EdgeReport:
        """Walk the live book for ``amount_usd`` and score it against ``fair_prob``."""
        fill = OrderBookMath.walk((await self.book(token_id)).side(side), side, amount_usd)
        if fill.vwap is None:
            raise PolyError(f"No {'asks' if side == 'BUY' else 'bids'} on the book for token {token_id}.")
        return OrderBookMath.edge(fill, fair_prob, fair_prob_sd, days_to_resolution)

    async def scan_neg_risk(self, event_slug: str, size_usd: float) -> BundleScan:
        """Buy-all-YES and buy-all-NO bundle costs for every open outcome of an event."""
        event = await self._get(f"{self._settings.gamma_url}/events/slug/{event_slug}")
        markets = [m for m in event.get("markets", []) if m.get("active") and not m.get("closed")]
        if not markets:
            raise PolyError(f"Event {event_slug!r} has no open outcome markets.")
        tokens = [json.loads(m["clobTokenIds"]) for m in markets]
        books = await asyncio.gather(*(self.book(t) for pair in tokens for t in pair))
        legs = tuple(
            BundleLeg(
                title=m.get("groupItemTitle") or m.get("question", "?"),
                yes=OrderBookMath.walk(books[2 * i].asks, "BUY", size_usd),
                no=OrderBookMath.walk(books[2 * i + 1].asks, "BUY", size_usd),
            )
            for i, m in enumerate(markets)
        )

        def total(fills: list[Fill]) -> float | None:
            return sum(f.vwap for f in fills if f.vwap is not None) if all(f.complete for f in fills) else None

        return BundleScan(
            event_slug,
            bool(event.get("negRisk")),
            legs,
            total([leg.yes for leg in legs]),
            total([leg.no for leg in legs]),
        )

    async def place_order(
        self,
        market_id: str,
        token_id: str,
        side: Side,
        amount_usd: float,
        limit_price: float,
        dry_run: bool = True,
        reasoning: str = "",
    ) -> OrderCheck:
        """Validate an order against the cap, market and live book; send it only if ``dry_run`` is False.

        The same checks run in both modes, so a dry run proves the live order
        would pass every local guard.

        Raises:
            PolyError: Over the cap, token not in market, market closed, price
                off-tick, size below the market minimum, or a send failure.
        """
        if amount_usd > self._settings.max_order_usd:
            raise PolyError(f"amount_usd {amount_usd} exceeds POLY_MAX_ORDER_USD {self._settings.max_order_usd}.")
        shares = math.floor(amount_usd / limit_price * 100) / 100  # never round size up past the cap
        market = await self._get(f"{self._settings.gamma_url}/markets/{market_id}")
        if token_id not in json.loads(market.get("clobTokenIds") or "[]"):
            raise PolyError(f"token {token_id} does not belong to market {market_id}.")
        if market.get("closed") or not market.get("acceptingOrders"):
            raise PolyError(f"market {market_id} is not accepting orders.")
        book = await self.book(token_id)
        if book.tick_size and abs(round(limit_price / book.tick_size) * book.tick_size - limit_price) > 1e-9:
            raise PolyError(f"limit_price {limit_price} is not a multiple of tick size {book.tick_size}.")
        if shares <= 0 or (book.min_order_size and shares < book.min_order_size):
            raise PolyError(f"order is {shares:.2f} shares; market minimum is {book.min_order_size}.")
        fill = OrderBookMath.walk(book.side(side), side, shares * limit_price, limit=limit_price)
        logger.info(
            "order %s market=%s side=%s shares=%s px=%s reasoning=%r",
            "DRY" if dry_run else "LIVE",
            market_id,
            side,
            shares,
            limit_price,
            reasoning,
        )
        result = None if dry_run else await self._sender(self._settings, token_id, side, limit_price, shares)
        return OrderCheck(market_id, token_id, side, limit_price, shares, fill, reasoning, result)

    async def list_open_orders(self, market_id: str | None = None) -> tuple[RestingOrder, ...]:
        """List resting GTC orders for the Deposit Wallet, resolving Gamma ``market_id``/slug to ``conditionId``."""
        condition_id: str | None = None
        if market_id is not None and market_id.strip():
            target = market_id.strip()
            if _CONDITION_ID_RE.fullmatch(target):
                condition_id = target
            else:
                path = f"/markets/{target}" if target.isdigit() else f"/markets/slug/{target}"
                market = await self._get(f"{self._settings.gamma_url}{path}")
                raw_cid = str(market.get("conditionId") or "").strip() if isinstance(market, dict) else ""
                if not raw_cid:
                    raise PolyError(f"Gamma market {target!r} has no conditionId.")
                condition_id = raw_cid
        return await self._manager.list_open_orders(self._settings, condition_id)

    async def cancel_order(self, order_id: str, dry_run: bool = True) -> CancelResult:
        """Validate or cancel one resting CLOB order. Defaults to ``dry_run=True`` for trading safety."""
        clean_id = order_id.strip()
        if not clean_id:
            raise PolyError("order_id cannot be blank.")
        logger.info("cancel_order %s order_id=%s", "DRY" if dry_run else "LIVE", clean_id)
        if dry_run:
            return CancelResult(
                order_id=clean_id,
                dry_run=True,
                canceled=False,
                status="dry_run",
                reason=f"validated (POLY_MAX_ORDER_USD={self._settings.max_order_usd:g}); pass dry_run=False to cancel.",
            )
        return await self._manager.cancel_order(self._settings, clean_id)

    @staticmethod
    def _is_liquid(m: dict[str, Any]) -> bool:
        bid, ask = m.get("bestBid") or 0, m.get("bestAsk") or 0
        return bool(m.get("enableOrderBook") and m.get("acceptingOrders")) and 0 < bid < ask <= bid + MAX_SPREAD

    async def _get(self, url: str, params: Mapping[str, str | int | float] | None = None) -> Any:
        try:
            response = await self._http.get(url, params=params, timeout=self._settings.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise PolyError("Polymarket request timed out, please retry.") from exc
        except httpx.RequestError as exc:
            raise PolyError(f"Could not reach Polymarket ({type(exc).__name__}).") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise PolyError(f"Polymarket returned non-JSON (HTTP {response.status_code}).") from exc
        if response.is_success:
            return body
        reason = body.get("error") or body.get("message") if isinstance(body, dict) else None
        raise PolyError(f"Polymarket error {response.status_code}: {reason or 'unexpected response body'}")
