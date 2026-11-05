"""Async client for the FRED v1 ``series`` endpoints. No MCP code here.

References:
    https://fred.stlouisfed.org/docs/api/fred/series_observations.html
    https://fred.stlouisfed.org/docs/api/fred/series_search.html
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import httpx2 as httpx

logger = logging.getLogger(__name__)

# FRED v1 authenticates with ?api_key=..., and httpx2 logs every request URL at
# INFO. Pin its logger to WARNING so enabling INFO logging cannot leak the key.
logging.getLogger("httpx2").setLevel(logging.WARNING)

FRED_URL = "https://api.stlouisfed.org/fred"
SEARCH_LIMIT_MAX = 1000

Units = Literal["lin", "chg", "ch1", "pch", "pc1", "pca", "cch", "cca", "cap", "log"]


class FredError(Exception):
    """Any FRED failure: missing key, bad arguments, HTTP or API error."""


@dataclass(frozen=True)
class FredSettings:
    """FRED settings; the API key is hidden from ``repr``."""

    api_key: str = field(repr=False)
    base_url: str = FRED_URL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> FredSettings:
        """Read ``FRED_API_KEY``.

        Raises:
            FredError: If the key is missing or blank.
        """
        api_key = env.get("FRED_API_KEY", "").strip()
        if not api_key:
            raise FredError("FRED_API_KEY is not set. Add it to the .env next to mcp_server.py or export it.")
        return cls(api_key=api_key)


class FredClient:
    """Calls FRED's series endpoints; the injected HTTP client is never closed here."""

    def __init__(self, settings: FredSettings, http: httpx.AsyncClient) -> None:
        """Create the client with settings and an open async HTTP client."""
        self._settings = settings
        self._http = http

    async def series_observations(
        self, series_id: str, observation_start: date, observation_end: date, units: Units = "lin"
    ) -> dict[str, Any]:
        """Observations for one series; missing values are reported by FRED as ``"."``.

        Raises:
            FredError: Reversed dates, or FRED failed/rejected the request.
        """
        if observation_start > observation_end:
            raise FredError(f"observation_start {observation_start} is after observation_end {observation_end}.")
        return await self._get(
            "series/observations",
            {
                "series_id": series_id.strip(),
                "observation_start": observation_start.isoformat(),
                "observation_end": observation_end.isoformat(),
                "units": units,
            },
        )

    async def search_series(self, search_text: str, limit: int = 10) -> dict[str, Any]:
        """Full-text series search; matches are under ``"seriess"`` (FRED's spelling)."""
        return await self._get("series/search", {"search_text": search_text.strip(), "limit": limit})

    async def _get(self, path: str, params: dict[str, str | int]) -> dict[str, Any]:
        query = {**params, "api_key": self._settings.api_key, "file_type": "json"}
        logger.info("FRED GET %s %s", path, params)  # params never include the key
        try:
            response = await self._http.get(
                f"{self._settings.base_url}/{path}", params=query, timeout=self._settings.timeout_seconds
            )
        except httpx.TimeoutException as exc:
            raise FredError("FRED request timed out, please retry.") from exc
        except httpx.RequestError as exc:
            # str(exc) can embed the URL, and with it the key; report the type only.
            raise FredError(f"Could not reach FRED ({type(exc).__name__}).") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise FredError(f"FRED returned non-JSON (HTTP {response.status_code}).") from exc
        if response.is_success and isinstance(body, dict):
            return body
        # FRED v1 errors look like {"error_code": 400, "error_message": "..."}; a bad key is a 400 too.
        message = body.get("error_message") if isinstance(body, dict) else None
        raise FredError(f"FRED error {response.status_code}: {message or 'unexpected response body'}")
