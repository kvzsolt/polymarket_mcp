"""Polling math, GDELT news volume, Fed press releases and cross-venue politics odds. No MCP code here.

``PollDropoutCalculator`` and ``OddsMath`` are pure and deterministic. Poll rows
are not fetched here: RealClearPolitics has no API, so the MCP host collects the
table (e.g. with the Firecrawl MCP) and passes the rows in.

References:
    https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    https://www.federalreserve.gov/feeds/feeds.htm
    https://www.predictit.org/api/marketdata/all/
    https://the-odds-api.com/liveapi/guides/v4/
"""

from __future__ import annotations

import logging
import math
import statistics
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import httpx2 as httpx

logger = logging.getLogger(__name__)

# The Odds API key travels in the query string and httpx2 logs request URLs at INFO.
logging.getLogger("httpx2").setLevel(logging.WARNING)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
FED_RSS_URL = "https://www.federalreserve.gov/feeds/press_all.xml"
PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"
ODDS_API_URL = "https://api.the-odds-api.com/v4"
GDELT_MAX_RECORDS = 250
DEFAULT_ODDS_SPORT = "politics_us_presidential_election_winner"


class PoliticsIntelError(Exception):
    """Any polling-math, GDELT, Fed RSS, PredictIt or The Odds API failure."""


@dataclass(frozen=True)
class PoliticsIntelSettings:
    """Only The Odds API needs a key; it is hidden from ``repr``."""

    odds_api_key: str = field(default="", repr=False)
    gdelt_url: str = GDELT_URL
    fed_rss_url: str = FED_RSS_URL
    predictit_url: str = PREDICTIT_URL
    odds_api_url: str = ODDS_API_URL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> PoliticsIntelSettings:
        """Read the optional ``THE_ODDS_API_KEY``."""
        return cls(odds_api_key=env.get("THE_ODDS_API_KEY", "").strip())


@dataclass(frozen=True)
class Poll:
    """One poll in a rolling average: the pollster, its field end date, and the topline value."""

    pollster: str
    end_date: date
    value: float


@dataclass(frozen=True)
class PollDropoutAnalysis:
    """How the average moves when the oldest poll leaves the window and nothing replaces it."""

    polls: int
    current_avg: float
    oldest: Poll
    oldest_ties: int  # other polls sharing the oldest end date: which one drops is ambiguous
    avg_without_oldest: float
    shift_if_dropped: float


class PollDropoutCalculator:
    """Pure rolling-average dropout math."""

    @staticmethod
    def analyze(polls: Sequence[Poll]) -> PollDropoutAnalysis:
        """Average with and without the oldest poll; input order does not matter.

        Raises:
            PoliticsIntelError: Fewer than 2 polls, or a non-finite value.
        """
        if len(polls) < 2:
            raise PoliticsIntelError("At least 2 polls are required to compute a dropout effect.")
        if not all(math.isfinite(p.value) for p in polls):
            raise PoliticsIntelError("Poll values must be finite numbers.")
        ordered = sorted(polls, key=lambda p: p.end_date, reverse=True)  # stable: ties keep input order
        oldest = ordered[-1]
        current = statistics.fmean(p.value for p in ordered)
        without = statistics.fmean(p.value for p in ordered[:-1])
        return PollDropoutAnalysis(
            polls=len(ordered),
            current_avg=current,
            oldest=oldest,
            oldest_ties=sum(1 for p in ordered[:-1] if p.end_date == oldest.end_date),
            avg_without_oldest=without,
            shift_if_dropped=without - current,
        )


@dataclass(frozen=True)
class OutcomeConsensus:
    """Bookmaker consensus for one outcome."""

    name: str
    fair_prob: float | None  # mean no-vig probability over usable books quoting it; None if none do
    best_price: float  # highest decimal price offered by any book
    books: int  # books quoting this outcome


@dataclass(frozen=True)
class EventOdds:
    """Consensus over the bookmakers quoting one event's outrights market."""

    title: str
    commence_time: str
    outcomes: tuple[OutcomeConsensus, ...]
    usable_books: int  # books whose quotes sum to >= 100%, used for fair_prob
    skipped_books: int  # books whose quotes sum to < 100%: normalizing them would inflate every quote


class OddsMath:
    """Pure decimal-odds arithmetic."""

    @staticmethod
    def consensus(event: Mapping[str, Any], market_key: str = "outrights") -> EventOdds:
        """Per usable book, implied = 1/price normalized to sum to 1 (removes the overround), then averaged.

        Books in a large outright field each quote a slightly different set of
        longshots, so requiring identical coverage would discard every real book.
        A book whose quotes sum to >= 100% is treated as its own complete market
        (unquoted names priced near zero). A book summing to < 100% (e.g. one
        quoting a single favourite) cannot be de-vigged: normalizing would scale
        its quotes up toward certainty, so it is skipped for ``fair_prob`` but
        still counts toward ``best_price``.

        Raises:
            KeyError, TypeError, ValueError, AttributeError: On a malformed payload;
                the client turns these into ``PoliticsIntelError``.
        """
        books: list[dict[str, float]] = []
        for book in event.get("bookmakers") or []:
            market = next((m for m in book.get("markets") or [] if m.get("key") == market_key), None)
            quotes = {str(o["name"]): float(o["price"]) for o in (market or {}).get("outcomes") or []}
            quotes = {n: p for n, p in quotes.items() if p > 1.0}
            if quotes:
                books.append(quotes)
        field: set[str] = set().union(*books)
        usable = [b for b in books if sum(1 / p for p in b.values()) >= 1.0]
        outcomes = []
        for name in field:
            fair = [(1 / b[name]) / sum(1 / p for p in b.values()) for b in usable if name in b]
            quoting = [b[name] for b in books if name in b]
            outcomes.append(
                OutcomeConsensus(name, statistics.fmean(fair) if fair else None, max(quoting), len(quoting))
            )
        outcomes.sort(key=lambda o: (o.fair_prob if o.fair_prob is not None else -1.0, -o.best_price), reverse=True)
        title = str(event.get("sport_title") or event.get("home_team") or event.get("id") or "event")
        return EventOdds(
            title, str(event.get("commence_time") or ""), tuple(outcomes), len(usable), len(books) - len(usable)
        )


@dataclass(frozen=True)
class FedRelease:
    """One Federal Reserve press release from the RSS feed."""

    title: str
    published: str
    link: str


class PoliticsIntelClient:
    """Fetches GDELT, Fed RSS, PredictIt and The Odds API data; never closes the HTTP client."""

    def __init__(self, settings: PoliticsIntelSettings, http: httpx.AsyncClient) -> None:
        """Create the client with settings and an open async HTTP client."""
        self._settings = settings
        self._http = http

    async def gdelt_articles(self, query: str, timespan: str = "24h", limit: int = 15) -> list[dict[str, Any]]:
        """Newest global news articles matching a GDELT query (``mode=artlist``)."""
        body = await self._gdelt(query, "artlist", timespan, {"maxrecords": limit, "sort": "DateDesc"})
        return list(body.get("articles") or [])

    async def gdelt_timeline(self, query: str, timespan: str = "24h") -> list[tuple[str, float]]:
        """``(date, share of coverage)`` points (``mode=timelinevol``, 15-minute steps for short spans).

        Raises:
            PoliticsIntelError: HTTP failure or a malformed timeline.
        """
        body = await self._gdelt(query, "timelinevol", timespan, {})
        series = body.get("timeline") or []
        try:
            return [(str(p["date"]), float(p["value"])) for p in (series[0]["data"] if series else [])]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise PoliticsIntelError(f"GDELT returned a malformed timeline ({type(exc).__name__}).") from exc

    async def fed_press_releases(self, limit: int = 10) -> tuple[FedRelease, ...]:
        """Latest Federal Reserve press releases. Parses bytes, so the feed's UTF-8 BOM is harmless.

        Raises:
            PoliticsIntelError: HTTP failure or malformed XML.
        """
        response = await self._request("Fed RSS", self._settings.fed_rss_url, {})
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise PoliticsIntelError(f"Fed RSS returned malformed XML ({exc}).") from exc
        return tuple(
            FedRelease(
                (item.findtext("title") or "").strip(),
                (item.findtext("pubDate") or "").strip(),
                (item.findtext("link") or "").strip(),
            )
            for item in root.iter("item")
        )[:limit]

    async def predictit_markets(self, keyword: str) -> list[dict[str, Any]]:
        """Open PredictIt markets whose name contains every word of ``keyword`` (case-insensitive)."""
        body = self._json_object("PredictIt", await self._request("PredictIt", self._settings.predictit_url, {}))
        markets = body.get("markets") or []
        if not isinstance(markets, list) or not all(isinstance(m, dict) for m in markets):
            raise PoliticsIntelError("PredictIt returned malformed markets.")
        words = keyword.lower().split()
        return [m for m in markets if all(w in f"{m.get('name', '')} {m.get('shortName', '')}".lower() for w in words)]

    async def bookmaker_politics_odds(
        self, sport_key: str = DEFAULT_ODDS_SPORT, regions: str = "us,uk,eu"
    ) -> tuple[EventOdds, ...]:
        """Outright bookmaker odds for a politics sport key, reduced to no-vig consensus per event.

        Raises:
            PoliticsIntelError: No ``THE_ODDS_API_KEY``, or The Odds API failed.
        """
        if not self._settings.odds_api_key:
            raise PoliticsIntelError("THE_ODDS_API_KEY is not set. A free key is at https://the-odds-api.com/.")
        response = await self._request(
            "The Odds API",
            f"{self._settings.odds_api_url}/sports/{sport_key}/odds",
            {
                "apiKey": self._settings.odds_api_key,
                "regions": regions,
                "markets": "outrights",
                "oddsFormat": "decimal",
            },
        )
        try:
            events = response.json()
        except ValueError as exc:
            raise PoliticsIntelError(f"The Odds API returned non-JSON (HTTP {response.status_code}).") from exc
        if not isinstance(events, list):
            raise PoliticsIntelError("The Odds API returned an unexpected body.")
        try:
            return tuple(OddsMath.consensus(e) for e in events)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PoliticsIntelError(f"The Odds API returned malformed odds ({type(exc).__name__}: {exc}).") from exc

    async def _gdelt(self, query: str, mode: str, timespan: str, extra: Mapping[str, str | int]) -> dict[str, Any]:
        params = {"query": query.strip(), "mode": mode, "timespan": timespan, "format": "json", **extra}
        response = await self._request("GDELT", self._settings.gdelt_url, params)
        if not response.content.strip():
            return {}  # GDELT answers an empty body when nothing matches
        return self._json_object("GDELT", response)

    @staticmethod
    def _json_object(source: str, response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            # GDELT reports query problems as 200 + plain text ("Your search contained ...").
            raise PoliticsIntelError(f"{source}: {response.text.strip()[:200] or 'non-JSON response'}") from exc
        if not isinstance(body, dict):
            raise PoliticsIntelError(f"{source} returned an unexpected body.")
        return body

    async def _request(self, source: str, url: str, params: Mapping[str, str | int]) -> httpx.Response:
        """GET ``url``; transport failures and HTTP errors become ``PoliticsIntelError``."""
        logger.info("%s GET %s", source, {k: v for k, v in params.items() if k != "apiKey"})
        try:
            response = await self._http.get(url, params=params, timeout=self._settings.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise PoliticsIntelError(f"{source} request timed out, please retry.") from exc
        except httpx.RequestError as exc:
            # str(exc) can embed the URL, and with it the key; report the type only.
            raise PoliticsIntelError(f"Could not reach {source} ({type(exc).__name__}).") from exc
        if response.is_success:
            return response
        try:
            body = response.json()
            reason = body.get("message") or body.get("error") if isinstance(body, dict) else None
        except ValueError:
            reason = None
        # GDELT's rate limit is plain text: "Please limit requests to one every 5 seconds ...".
        reason = reason or response.text.strip()[:200] or "no response body"
        raise PoliticsIntelError(f"{source} error {response.status_code}: {reason}")
