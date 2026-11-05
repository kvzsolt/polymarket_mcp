"""PolyNode API client: X Search, Online Search, and market holder positioning. No MCP code here.

``HolderMath`` is pure and deterministic; ``PolynodeClient`` does the HTTP.
PolyNode reports balances and prices as raw 6-decimal integers in strings
(``"500000"`` is $0.50), so every number is scaled by ``RAW_SCALE``.

References:
    https://docs.polynode.dev/api-reference/x-search.md
    https://docs.polynode.dev/api-reference/search-online.md
    https://docs.polynode.dev/data/markets/by-slug.md
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import httpx2 as httpx

logger = logging.getLogger(__name__)

POLYNODE_URL = "https://api.polynode.dev"
RAW_SCALE = 1_000_000  # 6-decimal fixed point used for shares, prices and USDC
X_SEARCH_MAX = 50
ONLINE_SEARCH_MAX = 20
POSITIONS_MAX = 300

TimeRange = Literal["day", "week", "month", "year"]
HolderStatus = Literal["open", "closed"]

# Used only when the error body carries no message of its own.
_STATUS_HINTS = {
    401: "missing or invalid POLYNODE_API_KEY",
    402: "this endpoint needs a paid PolyNode plan",
    429: "rate limited (X Search: 1 request/s) or monthly quota exhausted",
}


class PolynodeError(Exception):
    """Any PolyNode failure: missing key, HTTP or API error, malformed payload."""


@dataclass(frozen=True)
class PolynodeSettings:
    """PolyNode settings; the API key is hidden from ``repr``."""

    api_key: str = field(repr=False)
    base_url: str = POLYNODE_URL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> PolynodeSettings:
        """Read ``POLYNODE_API_KEY`` and the optional ``POLYNODE_BASE_URL``.

        Raises:
            PolynodeError: If the key is missing or blank.
        """
        api_key = env.get("POLYNODE_API_KEY", "").strip()
        if not api_key:
            raise PolynodeError("POLYNODE_API_KEY is not set. Add it to the .env next to mcp_server.py or export it.")
        return cls(api_key=api_key, base_url=env.get("POLYNODE_BASE_URL", POLYNODE_URL).rstrip("/"))


@dataclass(frozen=True)
class Holder:
    """One wallet's position in one outcome token, in human units."""

    wallet: str
    token_id: str
    shares: float
    avg_price: float
    realized_pnl_usd: float

    @classmethod
    def from_raw(cls, row: Mapping[str, Any]) -> Holder:
        """Scale a PolyNode positions row from raw 6-decimal strings.

        Raises:
            PolynodeError: If a numeric field is missing or not a number.
        """
        try:
            return cls(
                wallet=str(row["user"]),
                token_id=str(row["token_id"]),
                shares=int(row["amount"]) / RAW_SCALE,
                avg_price=int(row.get("avg_price") or 0) / RAW_SCALE,
                realized_pnl_usd=int(row.get("realized_pnl") or 0) / RAW_SCALE,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PolynodeError(f"Malformed PolyNode position row ({type(exc).__name__}: {exc}).") from exc


@dataclass(frozen=True)
class TokenConcentration:
    """Holder concentration for one outcome token, over the fetched rows only."""

    token_id: str
    holders: int
    total_shares: float
    top_n: int
    top_share: float  # fraction of total_shares held by the top_n wallets
    hhi: float  # Herfindahl index of share fractions, 0..1
    avg_entry: float | None  # share-weighted average entry price


@dataclass(frozen=True)
class HolderReport:
    """Positions for one market slug, largest holders first."""

    slug: str
    holders: tuple[Holder, ...]
    tokens: tuple[TokenConcentration, ...]
    has_more: bool


class HolderMath:
    """Pure concentration statistics over a list of holders."""

    @staticmethod
    def concentration(holders: Sequence[Holder], top_n: int) -> tuple[TokenConcentration, ...]:
        """Per-token top-N share, HHI and weighted entry price; tokens with no shares are skipped.

        Outcomes are computed separately because YES and NO shares are not comparable.
        """
        by_token: dict[str, list[Holder]] = {}
        for h in holders:
            if h.shares > 0:
                by_token.setdefault(h.token_id, []).append(h)
        result = []
        for token_id, rows in by_token.items():
            total = sum(h.shares for h in rows)
            sizes = sorted((h.shares for h in rows), reverse=True)
            cost = sum(h.shares * h.avg_price for h in rows if h.avg_price > 0)
            priced = sum(h.shares for h in rows if h.avg_price > 0)
            result.append(
                TokenConcentration(
                    token_id=token_id,
                    holders=len(rows),
                    total_shares=total,
                    top_n=top_n,
                    top_share=sum(sizes[:top_n]) / total,
                    hhi=sum((s / total) ** 2 for s in sizes),
                    avg_entry=cost / priced if priced else None,
                )
            )
        return tuple(sorted(result, key=lambda t: t.total_shares, reverse=True))


class PolynodeClient:
    """Calls PolyNode search and positions endpoints; the injected HTTP client is never closed here."""

    def __init__(self, settings: PolynodeSettings, http: httpx.AsyncClient) -> None:
        """Create the client with settings and an open async HTTP client."""
        self._settings = settings
        self._http = http

    async def x_search(
        self, query: str, max_results: int = 15, since: date | None = None, until: date | None = None
    ) -> dict[str, Any]:
        """Latest X posts for ``query``; dates become X's ``since:``/``until:`` operators.

        Raises:
            PolynodeError: Reversed dates, or PolyNode failed/rejected the request.
        """
        if since and until and since > until:
            raise PolynodeError(f"since {since} is after until {until}.")
        q = " ".join(
            [query.strip()]
            + ([f"since:{since.isoformat()}"] if since else [])
            + ([f"until:{until.isoformat()}"] if until else [])
        )
        return await self._get("/v2/x/search", {"q": q, "max": max_results})

    async def online_search(
        self, query: str, max_results: int = 10, time_range: TimeRange | None = None
    ) -> dict[str, Any]:
        """Normalized web search results for corroborating news."""
        params: dict[str, str | int] = {"q": query.strip(), "max_results": max_results}
        if time_range:
            params["time_range"] = time_range
        return await self._get("/v2/search/online", params)

    async def market_positions(
        self, slug: str, limit: int = 100, top_n: int = 10, status: HolderStatus | None = "open"
    ) -> HolderReport:
        """Largest holders of a market (all outcomes) plus per-token concentration.

        Concentration covers the fetched rows only; with ``has_more`` it overstates the true figure.

        Raises:
            PolynodeError: PolyNode failed, or returned malformed rows.
        """
        # Closed positions all have amount 0, so rank those by realized P&L instead.
        params: dict[str, str | int] = {"limit": limit, "sort": "realized_pnl" if status == "closed" else "amount"}
        if status:
            params["status"] = status
        body = await self._get(f"/v3/markets/slug/{slug}/positions", params)
        rows = body.get("data") or []
        if not isinstance(rows, list):
            raise PolynodeError("PolyNode positions payload has no row list.")
        holders = tuple(Holder.from_raw(row) for row in rows)
        if status != "closed":  # do not rely on the API's sort direction
            holders = tuple(sorted(holders, key=lambda h: h.shares, reverse=True))
        return HolderReport(
            slug=slug,
            holders=holders,
            tokens=HolderMath.concentration(holders, top_n),
            has_more=bool(body.get("has_more")),
        )

    async def _get(self, path: str, params: Mapping[str, str | int]) -> dict[str, Any]:
        logger.info("PolyNode GET %s %s", path, dict(params))  # the key travels in a header
        try:
            response = await self._http.get(
                f"{self._settings.base_url}{path}",
                params=params,
                headers={"x-api-key": self._settings.api_key, "Accept": "application/json"},
                timeout=self._settings.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise PolynodeError("PolyNode request timed out, please retry.") from exc
        except httpx.RequestError as exc:
            raise PolynodeError(f"Could not reach PolyNode ({type(exc).__name__}).") from exc
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.is_success and isinstance(body, dict):
            return body
        if response.is_success:
            raise PolynodeError(f"PolyNode returned an unexpected body (HTTP {response.status_code}).")
        reason = None
        if isinstance(body, dict):
            reason = body.get("error") or body.get("message") or body.get("detail")
        reason = reason or _STATUS_HINTS.get(response.status_code) or response.text[:200] or "no response body"
        raise PolynodeError(f"PolyNode error {response.status_code} on {path}: {reason}")
