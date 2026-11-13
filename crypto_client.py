"""Reference-oracle pricing for crypto expiry and one-touch barrier markets.

No MCP code here. ``ExpiryPricer`` is pure stdlib math supporting both terminal
expiry (``P(S_T > K)`` / ``P(S_T < K)``) and pathwise one-touch barriers
(``P(max S_t >= K)`` / ``P(min S_t <= K)`` via the Brownian reflection principle
and drift-demeaned OHLCV paths). ``CryptoClient`` automatically selects Binance
``1h`` klines for horizons ``<= 168h`` and ``1d`` klines (up to 1000 days) for
multi-week/multi-month horizons up to ``8760h`` (365d), plus Deribit mark IV,
expiry-mismatch diagnostics, and Breeden-Litzenberger call/put IV skew notes.

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
from typing import Any, Literal

import httpx2 as httpx

BINANCE_URL = "https://api.binance.com"
DERIBIT_URL = "https://www.deribit.com/api/v2"
KLINES = 1000  # Binance's per-request maximum
WEEK_HOURS = 168
MONTH_DAYS = 30
YEAR_HOURS = 24 * 365
MAX_HORIZON_HOURS = YEAR_HOURS

ContractType = Literal["expiry", "touch"]


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
    """One model's probability that the YES barrier/expiry condition resolves true."""

    name: str
    p_yes: float
    sigma_h: float | None  # horizon log-return sd; None for the empirical distribution
    samples: int

    @property
    def p_no(self) -> float:
        """Probability the market resolves NO."""
        return 1.0 - self.p_yes


@dataclass(frozen=True)
class ExpiryPricing:
    """All estimates for one strike, horizon, and contract type (expiry or one-touch)."""

    symbol: str
    spot: float
    strike: float
    horizon_hours: int
    r_star: float  # ln(strike / spot): log move from spot to strike
    estimates: tuple[Estimate, ...]
    iv_note: str  # Deribit instrument, expiry mismatch, and skew note (or why IV is absent)
    contract_type: ContractType = "expiry"
    interval: str = "1h"

    @property
    def upside(self) -> bool:
        """True when strike >= spot (upside target/barrier); False for downside dips."""
        return self.strike >= self.spot


class ExpiryPricer:
    """Zero-drift terminal expiry and one-touch reflection-barrier pricer."""

    @staticmethod
    def price(
        symbol: str,
        closes: Sequence[float],
        strike: float,
        horizon_hours: int,
        implied_vol: float | None = None,
        iv_note: str = "",
        *,
        contract_type: ContractType = "expiry",
        bar_hours: int = 1,
        highs: Sequence[float] | None = None,
        lows: Sequence[float] | None = None,
    ) -> ExpiryPricing:
        """Price terminal ``expiry`` or pathwise ``touch`` with empirical, normal, and IV models.

        * Direction: ``strike >= spot`` prices upside (``S_T > K`` or ``max S_t >= K``);
          ``strike < spot`` prices downside (``S_T < K`` or ``min S_t <= K``).
        * ``empirical``:
          - ``expiry``: share of rolling ``steps``-bar log returns (demeaned) crossing ``r_star``.
          - ``touch``: share of rolling ``steps``-bar OHLC paths (demeaned by per-bar drift)
            whose pathwise kline high/low crosses ``r_star``.
        * ``normal_7d`` (for ``1h`` bars) / ``normal_30d`` (for ``1d`` bars) & ``normal_full``:
          - ``expiry``: zero-drift Gaussian tail ``1 - Phi(|r_star| / sigma_h)`` (or ``Phi(r_star / sigma_h)``).
          - ``touch``: Brownian reflection principle ``min(1, 2 * (1 - Phi(|r_star| / sigma_h)))``.
        * ``implied``: same terminal/reflection formula using Deribit ``sigma_h = iv * sqrt(h / 8760)``.

        Raises:
            CryptoError: Invalid inputs, too few bars for the horizon, or zero volatility.
        """
        if horizon_hours < 1 or horizon_hours > MAX_HORIZON_HOURS:
            raise CryptoError(f"horizon_hours must be in 1..{MAX_HORIZON_HOURS}, got {horizon_hours}.")
        if bar_hours <= 0:
            raise CryptoError(f"bar_hours must be positive, got {bar_hours}.")
        steps = max(1, round(horizon_hours / bar_hours))
        unit = "hourly" if bar_hours == 1 else f"{bar_hours}h"
        if bar_hours == 24:
            unit = "daily"
        if len(closes) < steps + 2:
            raise CryptoError(f"need more than {steps + 1} {unit} closes, got {len(closes)}.")

        path_highs = closes if highs is None else highs
        path_lows = closes if lows is None else lows
        if len(path_highs) != len(closes) or len(path_lows) != len(closes):
            raise CryptoError("highs and lows must match closes in length.")

        spot = closes[-1]
        if spot <= 0 or strike <= 0:
            raise CryptoError("spot and strike must be positive.")
        r_star = math.log(strike / spot)
        upside = strike >= spot

        bar_returns = [math.log(b / a) for a, b in itertools.pairwise(closes)]
        window = [math.log(closes[i + steps] / closes[i]) for i in range(len(closes) - steps)]
        drift = statistics.fmean(window)

        if contract_type == "expiry":
            empirical = sum((r - drift > r_star if upside else r - drift < r_star) for r in window) / len(window)
        elif r_star == 0.0:
            empirical = 1.0
        else:
            mu_bar = drift / steps
            x_base = [math.log(c) - idx * mu_bar for idx, c in enumerate(closes)]
            if upside:
                u_high = [math.log(h) - idx * mu_bar for idx, h in enumerate(path_highs)]
                hits = sum(max(u_high[i + 1 : i + steps + 1]) - x_base[i] >= r_star for i in range(len(window)))
            else:
                l_low = [math.log(lv) - idx * mu_bar for idx, lv in enumerate(path_lows)]
                hits = sum(min(l_low[i + 1 : i + steps + 1]) - x_base[i] <= r_star for i in range(len(window)))
            empirical = hits / len(window)

        estimates = [Estimate("empirical", empirical, None, len(window))]
        short_name, short_bars = ("normal_7d", WEEK_HOURS) if bar_hours == 1 else ("normal_30d", MONTH_DAYS)
        for name, sample in ((short_name, bar_returns[-short_bars:]), ("normal_full", bar_returns)):
            if len(sample) < 2:
                raise CryptoError(f"need at least 2 {unit} returns for {name}, got {len(sample)}.")
            sigma_bar = statistics.stdev(sample)
            if sigma_bar == 0:
                raise CryptoError(f"{symbol} price series is flat; volatility is zero.")
            sigma_h = sigma_bar * math.sqrt(horizon_hours / bar_hours)
            estimates.append(ExpiryPricer._parametric(name, r_star, sigma_h, len(sample), contract_type, upside))
        if implied_vol:
            sigma_h = implied_vol * math.sqrt(horizon_hours / YEAR_HOURS)
            estimates.append(ExpiryPricer._parametric("implied", r_star, sigma_h, 1, contract_type, upside))

        interval = "1h" if bar_hours == 1 else ("1d" if bar_hours == 24 else f"{bar_hours}h")
        return ExpiryPricing(
            symbol=symbol,
            spot=spot,
            strike=strike,
            horizon_hours=horizon_hours,
            r_star=r_star,
            estimates=tuple(estimates),
            iv_note=iv_note,
            contract_type=contract_type,
            interval=interval,
        )

    @staticmethod
    def _parametric(
        name: str,
        r_star: float,
        sigma_h: float,
        samples: int,
        contract_type: ContractType,
        upside: bool,
    ) -> Estimate:
        dist = statistics.NormalDist(0.0, sigma_h)
        if contract_type == "touch":
            p_yes = 1.0 if r_star == 0.0 else min(1.0, 2.0 * (1.0 - dist.cdf(abs(r_star))))
        else:
            p_yes = 1.0 - dist.cdf(r_star) if upside else dist.cdf(r_star)
        return Estimate(name, p_yes, sigma_h, samples)


class CryptoClient:
    """Binance + Deribit public data; the injected HTTP client is never closed here."""

    def __init__(self, settings: CryptoSettings, http: httpx.AsyncClient) -> None:
        """Create the client with settings and an open async HTTP client."""
        self._settings = settings
        self._http = http

    async def price_expiry(
        self,
        symbol: str,
        strike: float,
        horizon_hours: int,
        contract_type: ContractType = "expiry",
    ) -> ExpiryPricing:
        """Fetch Binance ``1h`` (<=168h) or ``1d`` (>168h) klines + Deribit IV and price the contract."""
        interval, bar_hours = ("1h", 1) if horizon_hours <= WEEK_HOURS else ("1d", 24)
        klines = await self._get(
            f"{self._settings.binance_url}/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": KLINES},
        )
        if not isinstance(klines, list) or not klines:
            raise CryptoError(f"Binance returned no klines for {symbol}.")
        closes = [float(k[4]) for k in klines]
        # Fall back cleanly to close when synthetic/minimal klines report "0" for high/low.
        highs = [max(float(k[2]), c) if float(k[2]) > 0 else c for k, c in zip(klines, closes, strict=True)]
        lows = [min(float(k[3]), c) if float(k[3]) > 0 else c for k, c in zip(klines, closes, strict=True)]
        target_ms = int(klines[-1][6]) + horizon_hours * 3_600_000
        iv, note = await self._implied_vol(symbol, closes[-1], strike, horizon_hours, target_ms)
        return ExpiryPricer.price(
            symbol,
            closes,
            strike,
            horizon_hours,
            iv,
            note,
            contract_type=contract_type,
            bar_hours=bar_hours,
            highs=highs,
            lows=lows,
        )

    async def _implied_vol(
        self,
        symbol: str,
        spot: float,
        strike: float,
        horizon_hours: int,
        target_ms: int,
    ) -> tuple[float | None, str]:
        """Mark IV of the nearest-strike option expiring >= ``target_ms``, plus mismatch & skew diagnostics.

        Deribit is supplementary: when it fails or lists no options, the reason
        is returned for display instead of failing the whole pricing.
        """
        currency = symbol.upper().removesuffix("USDT").removesuffix("USDC").removesuffix("USD")
        base = f"{self._settings.deribit_url}/public"
        upside = strike >= spot
        want_suffix = "-C" if upside else "-P"
        want_type = "call" if upside else "put"
        try:
            options = await self._get(
                f"{base}/get_instruments", {"currency": currency, "kind": "option", "expired": "false"}
            )
            later = [o for o in options["result"] if o["expiration_timestamp"] >= target_ms]
            if not later:
                return None, f"no Deribit {currency} option expires after the horizon"
            expiry = min(o["expiration_timestamp"] for o in later)
            at_expiry = [o for o in later if o["expiration_timestamp"] == expiry]
            same_side = [
                o
                for o in at_expiry
                if o.get("option_type") == want_type or str(o.get("instrument_name", "")).endswith(want_suffix)
            ]
            pool = same_side or at_expiry
            pick = min(pool, key=lambda o: abs(float(o["strike"]) - strike))
            book = await self._get(f"{base}/get_order_book", {"instrument_name": pick["instrument_name"]})
            raw_iv = book.get("result", {}).get("mark_iv")
            if raw_iv is None or float(raw_iv) <= 0:
                return None, f"Deribit {pick['instrument_name']} has no positive mark_iv"
            iv = float(raw_iv) / 100.0
        except CryptoError as exc:
            return None, f"implied vol unavailable ({exc})"

        mismatch_h = (int(expiry) - target_ms) / 3_600_000.0
        mismatch_str = f"{mismatch_h / 24.0:+.1f}d" if abs(mismatch_h) >= 24.0 else f"{mismatch_h:+.1f}h"
        skew_note = await self._skew_note(base, pool, pick, iv, spot, strike, horizon_hours, upside)
        return iv, f"Deribit {pick['instrument_name']} mark_iv={iv * 100:.1f}% (expiry {mismatch_str} vs target{skew_note})"

    async def _skew_note(
        self,
        base: str,
        pool: list[dict[str, Any]],
        pick: dict[str, Any],
        pick_iv: float,
        spot: float,
        strike: float,
        horizon_hours: int,
        upside: bool,
    ) -> str:
        """Finite-difference IV slope and Breeden-Litzenberger digital correction across neighboring strikes."""
        k_pick = float(pick["strike"])
        lower = max((o for o in pool if float(o["strike"]) < k_pick), key=lambda o: float(o["strike"]), default=None)
        upper = min((o for o in pool if float(o["strike"]) > k_pick), key=lambda o: float(o["strike"]), default=None)
        if lower is None and upper is None:
            return ""

        k_lo, iv_lo = k_pick, pick_iv
        k_hi, iv_hi = k_pick, pick_iv
        for neighbor, is_low in ((lower, True), (upper, False)):
            if neighbor is None:
                continue
            try:
                nb = await self._get(f"{base}/get_order_book", {"instrument_name": neighbor["instrument_name"]})
                nb_iv = nb.get("result", {}).get("mark_iv")
                if nb_iv is not None and float(nb_iv) > 0:
                    if is_low:
                        k_lo, iv_lo = float(neighbor["strike"]), float(nb_iv) / 100.0
                    else:
                        k_hi, iv_hi = float(neighbor["strike"]), float(nb_iv) / 100.0
            except CryptoError:
                continue

        if k_hi <= k_lo:
            return ""
        d_iv_dk = (iv_hi - iv_lo) / (k_hi - k_lo)
        skew_10pct = d_iv_dk * (0.10 * strike) * 100.0
        t_yr = horizon_hours / YEAR_HOURS
        sigma_t = pick_iv * math.sqrt(t_yr)
        d1 = (math.log(spot / strike) + 0.5 * sigma_t * sigma_t) / sigma_t
        phi_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        # Breeden-Litzenberger: d(-dC/dK) = -S0 * phi(d1) * sqrt(T) * d(sigma)/dK; flip sign for P(S_T < K).
        bl_call_adj = -spot * phi_d1 * math.sqrt(t_yr) * d_iv_dk
        bl_yes_adj = bl_call_adj if upside else -bl_call_adj
        if abs(skew_10pct) < 1e-12:
            skew_10pct = 0.0
        if abs(bl_yes_adj) < 1e-12:
            bl_yes_adj = 0.0
        return f"; skew {skew_10pct:+.2f} iv_pts/10%K, BL digital adj {bl_yes_adj:+.4f}"

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
