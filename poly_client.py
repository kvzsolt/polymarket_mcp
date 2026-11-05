"""Polymarket client: Gamma market discovery, CLOB order books, fill math, orders.

No MCP code here. ``OrderBookMath`` is pure and deterministic; ``PolyClient``
does the HTTP for market data. Live orders go through the official SDK
(``polymarket-client``, ``AsyncSecureClient``), which signs EIP-712 orders for
the signer's Deposit Wallet and derives L2 credentials; it sits behind the
injectable ``OrderSender`` so tests never touch it.

References:
    https://docs.polymarket.com/api-reference/market-data/get-order-book
    https://github.com/Polymarket/py-sdk (polymarket-client 0.16.0)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

import httpx2 as httpx

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
MAX_SPREAD = 0.10  # wider YES spreads are treated as illiquid noise
GAMMA_FETCH = 200  # candidates fetched before client-side filtering

Side = Literal["BUY", "SELL"]


class PolyError(Exception):
    """Any Polymarket failure: config, HTTP, bad book, or a refused order."""


@dataclass(frozen=True)
class PolySettings:
    """Polymarket settings. Secrets are optional (market data needs none) and hidden from ``repr``.

    Live orders need ``private_key``. ``wallet`` defaults to the signer's
    Deposit Wallet, which the SDK deploys via the relayer on first live use.
    L2 credentials are derived from the key unless all three are supplied.
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
    """A fill plus edge and Kelly sizing against a fair probability."""

    fill: Fill
    fair_prob: float
    edge: float
    kelly_full: float
    kelly_shrunk: float


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


class OrderSender(Protocol):
    """Signs and posts one limit order; the seam that keeps the SDK out of tests."""

    async def __call__(
        self, settings: PolySettings, token_id: str, side: Side, price: float, size: float
    ) -> OrderResult:
        """Post a GTC limit order of ``size`` shares at ``price``."""
        ...


class SdkOrderSender:
    """``OrderSender`` backed by the official SDK's ``AsyncSecureClient``, Deposit Wallet only.

    The SDK infers the signature type from the signer/wallet pair (EOA=0,
    POLY_PROXY=1, GNOSIS_SAFE=2, DEPOSIT_WALLET=3). This sender refuses to post
    unless the pair classifies as ``DEPOSIT_WALLET``, i.e. signature type 3.
    """

    async def __call__(
        self, settings: PolySettings, token_id: str, side: Side, price: float, size: float
    ) -> OrderResult:
        """Sign for the configured (or default) Deposit Wallet and post the order.

        Raises:
            PolyError: Missing private key, a wallet that is not a Deposit
                Wallet, or any SDK error (signing, allowance, rejection,
                transport), with the SDK's message.
        """
        if not settings.private_key:
            raise PolyError("POLY_PRIVATE_KEY is required for live orders (dry_run=False).")
        # Imported lazily: eth-account is heavy and only the live path needs it.
        from polymarket import AcceptedOrder, ApiKeyCreds, AsyncSecureClient, PolymarketError

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
                wallet = str(client.wallet)
                response = await client.place_limit_order(
                    token_id=token_id, price=str(price), size=str(size), side=side
                )
        except PolymarketError as exc:
            raise PolyError(f"Polymarket SDK {type(exc).__name__}: {exc}") from exc
        if isinstance(response, AcceptedOrder):
            return OrderResult(True, response.order_id, response.status, wallet=wallet)
        return OrderResult(False, None, response.code, response.message, wallet=wallet)


class OrderBookMath:
    """Pure order-book arithmetic: VWAP walk, edge and shrunk Kelly."""

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
    def edge(fill: Fill, fair_prob: float, fair_prob_sd: float) -> EdgeReport:
        """Edge of the fill vs ``fair_prob`` (YES probability) and Kelly sizing.

        Kelly for a binary bought at ``p`` with win probability ``q`` is
        ``(q - p) / (1 - p)``; selling YES at ``p`` is buying NO at ``1 - p``,
        giving ``(p - q) / p``. The shrunk fraction multiplies by
        ``edge^2 / (edge^2 + sd^2)``, discounting edges that are small relative
        to the uncertainty in ``fair_prob``.
        """
        if fill.vwap is None:
            raise PolyError("Nothing filled, so there is no edge to measure.")
        p = fill.vwap
        edge = fair_prob - p if fill.side == "BUY" else p - fair_prob
        full = max(0.0, edge / (1 - p) if fill.side == "BUY" else edge / p)
        shrink = edge * edge / (edge * edge + fair_prob_sd * fair_prob_sd) if fair_prob_sd > 0 else 1.0
        return EdgeReport(fill, fair_prob, edge, full, full * shrink)


class PolyClient:
    """Async Polymarket API client; the injected HTTP client is never closed here."""

    def __init__(self, settings: PolySettings, http: httpx.AsyncClient, sender: OrderSender | None = None) -> None:
        """Create the client.

        Args:
            settings: Polymarket settings.
            http: Open async HTTP client for Gamma/CLOB reads; never closed here.
            sender: Live order sender; defaults to the official SDK.
        """
        self._settings = settings
        self._http = http
        self._sender = sender or SdkOrderSender()

    async def search_markets(
        self,
        query: str | None,
        tag: str | None,
        min_volume_usd: float,
        end_date_min: datetime,
        end_date_max: datetime | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Active, liquid markets matching every word of ``query`` (case-insensitive).

        Volume, expiry window and tag are filtered by Gamma; text match and
        liquidity (two-sided book, YES spread <= ``MAX_SPREAD``) are filtered here.
        """
        params: dict[str, str | int | float] = {
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
            "limit": GAMMA_FETCH,
            "volume_num_min": min_volume_usd,
            "end_date_min": end_date_min.isoformat(),
        }
        if end_date_max is not None:
            params["end_date_max"] = end_date_max.isoformat()
        if tag:
            params["tag_id"] = (await self._get(f"{self._settings.gamma_url}/tags/slug/{tag}"))["id"]
        words = (query or "").lower().split()
        markets = await self._get(f"{self._settings.gamma_url}/markets", params)
        hits = [m for m in markets if self._is_liquid(m) and all(w in m.get("question", "").lower() for w in words)]
        return hits[:limit]

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
        self, token_id: str, side: Side, amount_usd: float, fair_prob: float, fair_prob_sd: float
    ) -> EdgeReport:
        """Walk the live book for ``amount_usd`` and score it against ``fair_prob``."""
        fill = OrderBookMath.walk((await self.book(token_id)).side(side), side, amount_usd)
        if fill.vwap is None:
            raise PolyError(f"No {'asks' if side == 'BUY' else 'bids'} on the book for token {token_id}.")
        return OrderBookMath.edge(fill, fair_prob, fair_prob_sd)

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
