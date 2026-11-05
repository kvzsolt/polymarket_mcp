"""Reference-oracle pricing for crypto expiry markets ("will X close above K?").

No MCP code here. ``ExpiryPricer`` is pure stdlib math; ``CryptoClient`` fetches
Binance public hourly klines and, when the coin has listed options, Deribit's
mark implied volatility for the nearest option at or after the horizon.

References:
    https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
    https://docs.deribit.com/#public-get_order_book
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx2 as httpx

BINANCE_URL = "https://api.binance.com"
DERIBIT_URL = "https://www.deribit.com/api/v2"
KLINES = 1000  # Binance's per-request maximum
WEEK_HOURS = 168
YEAR_HOURS = 24 * 365


class CryptoError(Exception):
    """Any Binance/Deribit failure or unusable price series."""


@dataclass(frozen=True)
class CryptoSettings:
    """Public endpoints only; no keys needed."""

    binance_url: str = BINANCE_URL
    deribit_url: str = DERIBIT_URL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> CryptoSettings:
        """Optional overrides: ``BINANCE_URL``, ``DERIBIT_URL``."""
        return cls(env.get("BINANCE_URL", BINANCE_URL), env.get("DERIBIT_URL", DERIBIT_URL))


@dataclass(frozen=True)
class Estimate:
    """One model's probability that the close ends above the strike."""

    name: str
    p_yes: float
    sigma_h: float | None  # horizon log-return sd; None for the empirical CDF
    samples: int

    @property
    def p_no(self) -> float:
        """Probability the market resolves NO."""
        return 1.0 - self.p_yes


@dataclass(frozen=True)
class ExpiryPricing:
    """All estimates for one strike and horizon."""

    symbol: str
    spot: float
    strike: float
    horizon_hours: int
    r_star: float  # ln(strike / spot): log return needed to finish above the strike
    estimates: tuple[Estimate, ...]
    iv_note: str  # Deribit instrument used, or why implied vol is absent


class ExpiryPricer:
    """Zero-drift P(close > strike) from hourly closes, plus optional implied vol."""

    @staticmethod
    def price(
        symbol: str,
        closes: Sequence[float],
        strike: float,
        horizon_hours: int,
        implied_vol: float | None = None,
        iv_note: str = "",
    ) -> ExpiryPricing:
        """Price with the empirical CDF and two normal fits (and implied vol if given).

        * ``empirical``: share of overlapping ``h``-hour log returns, demeaned
          (zero drift), that exceed ``r_star``.
        * ``normal_7d`` / ``normal_full``: ``P(Z > r_star / sigma_h)`` with
          ``sigma_h = sigma_1h * sqrt(h)`` from the last 168 / all hourly returns.
        * ``implied``: same with ``sigma_h = iv * sqrt(h / 8760)``.

        Raises:
            CryptoError: Too few closes for the horizon, or a flat series.
        """
        if len(closes) < horizon_hours + 2:
            raise CryptoError(f"need more than {horizon_hours + 1} hourly closes, got {len(closes)}.")
        spot = closes[-1]
        r_star = math.log(strike / spot)
        hourly = [math.log(b / a) for a, b in itertools.pairwise(closes)]
        window = [math.log(closes[i + horizon_hours] / closes[i]) for i in range(len(closes) - horizon_hours)]
        drift = statistics.fmean(window)
        empirical = sum(r - drift > r_star for r in window) / len(window)
        estimates = [Estimate("empirical", empirical, None, len(window))]
        for name, sample in (("normal_7d", hourly[-WEEK_HOURS:]), ("normal_full", hourly)):
            sigma_1h = statistics.stdev(sample)
            if sigma_1h == 0:
                raise CryptoError(f"{symbol} price series is flat; volatility is zero.")
            estimates.append(ExpiryPricer._normal(name, r_star, sigma_1h * math.sqrt(horizon_hours), len(sample)))
        if implied_vol:
            sigma_h = implied_vol * math.sqrt(horizon_hours / YEAR_HOURS)
            estimates.append(ExpiryPricer._normal("implied", r_star, sigma_h, 1))
        return ExpiryPricing(symbol, spot, strike, horizon_hours, r_star, tuple(estimates), iv_note)

    @staticmethod
    def _normal(name: str, r_star: float, sigma_h: float, samples: int) -> Estimate:
        return Estimate(name, 1.0 - statistics.NormalDist(0.0, sigma_h).cdf(r_star), sigma_h, samples)


class CryptoClient:
    """Binance + Deribit public data; the injected HTTP client is never closed here."""

    def __init__(self, settings: CryptoSettings, http: httpx.AsyncClient) -> None:
        """Create the client with settings and an open async HTTP client."""
        self._settings = settings
        self._http = http

    async def price_expiry(self, symbol: str, strike: float, horizon_hours: int) -> ExpiryPricing:
        """Fetch 1000 hourly closes (and implied vol when available) and price the expiry."""
        klines = await self._get(
            f"{self._settings.binance_url}/api/v3/klines",
            {"symbol": symbol, "interval": "1h", "limit": KLINES},
        )
        closes = [float(k[4]) for k in klines]
        # The last kline's close time is the end of the current hour: a deterministic "now".
        target_ms = int(klines[-1][6]) + horizon_hours * 3_600_000
        iv, note = await self._implied_vol(symbol, strike, target_ms)
        return ExpiryPricer.price(symbol, closes, strike, horizon_hours, iv, note)

    async def _implied_vol(self, symbol: str, strike: float, target_ms: int) -> tuple[float | None, str]:
        """Mark IV of the nearest-strike option expiring at or after ``target_ms``.

        Deribit is supplementary: when it fails or lists no options, the reason
        is returned for display instead of failing the whole pricing.
        """
        currency = symbol.upper().removesuffix("USDT").removesuffix("USDC").removesuffix("USD")
        base = f"{self._settings.deribit_url}/public"
        try:
            options = await self._get(
                f"{base}/get_instruments", {"currency": currency, "kind": "option", "expired": "false"}
            )
            later = [o for o in options["result"] if o["expiration_timestamp"] >= target_ms]
            if not later:
                return None, f"no Deribit {currency} option expires after the horizon"
            expiry = min(o["expiration_timestamp"] for o in later)
            pick = min(
                (o for o in later if o["expiration_timestamp"] == expiry), key=lambda o: abs(o["strike"] - strike)
            )
            book = await self._get(f"{base}/get_order_book", {"instrument_name": pick["instrument_name"]})
        except CryptoError as exc:
            return None, f"implied vol unavailable ({exc})"
        return book["result"]["mark_iv"] / 100, f"Deribit {pick['instrument_name']} mark_iv"

    async def _get(self, url: str, params: Mapping[str, str | int]) -> Any:
        try:
            response = await self._http.get(url, params=params, timeout=self._settings.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise CryptoError(f"request to {url} timed out, please retry.") from exc
        except httpx.RequestError as exc:
            raise CryptoError(f"could not reach {url} ({type(exc).__name__}).") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise CryptoError(f"{url} returned non-JSON (HTTP {response.status_code}).") from exc
        if response.is_success:
            return body
        # Binance: {"code": -1121, "msg": "..."}; Deribit: {"error": {"message": ..., "data": {"reason": ...}}}
        reason = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                error = (error.get("data") or {}).get("reason") or error.get("message")
            reason = body.get("msg") or error
        raise CryptoError(f"HTTP {response.status_code} from {url}: {reason or 'unexpected response body'}")
