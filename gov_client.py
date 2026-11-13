"""US government and legal oracle client: Federal Register, Congress.gov, FEC, CourtListener. No MCP code here.

Congress.gov and FEC authenticate with ``?api_key=...`` in the URL, so request
URLs are never logged or echoed in errors. CourtListener v4 search requires a
token (it answers 401 without one); Federal Register needs no key.

References:
    https://www.federalregister.gov/developers/documentation/api/v1
    https://github.com/LibraryOfCongress/api.congress.gov (BillEndpoint.md, NominationEndpoint.md)
    https://api.open.fec.gov/developers/
    https://www.courtlistener.com/help/api/rest/search/
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx2 as httpx

logger = logging.getLogger(__name__)

# Keys travel in query strings and httpx2 logs request URLs at INFO.
logging.getLogger("httpx2").setLevel(logging.WARNING)

FED_REG_URL = "https://www.federalregister.gov/api/v1"
CONGRESS_URL = "https://api.congress.gov/v3"
FEC_URL = "https://api.open.fec.gov/v1"
COURTLISTENER_URL = "https://www.courtlistener.com/api/rest/v4"

BillType = Literal["hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", "sres"]


class GovError(Exception):
    """Any government/legal API failure: missing key, HTTP or API error, unexpected body."""


@dataclass(frozen=True)
class GovSettings:
    """Keys are optional at load time; each tool checks the one it needs. Secrets hidden from ``repr``."""

    congress_api_key: str = field(default="", repr=False)
    fec_api_key: str = field(default="DEMO_KEY", repr=False)
    courtlistener_token: str = field(default="", repr=False)
    federal_register_url: str = FED_REG_URL
    congress_url: str = CONGRESS_URL
    fec_url: str = FEC_URL
    courtlistener_url: str = COURTLISTENER_URL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> GovSettings:
        """Read ``CONGRESS_API_KEY``, ``FEC_API_KEY`` (default ``DEMO_KEY``), ``COURTLISTENER_API_TOKEN``."""
        return cls(
            congress_api_key=env.get("CONGRESS_API_KEY", "").strip(),
            fec_api_key=env.get("FEC_API_KEY", "").strip() or "DEMO_KEY",
            courtlistener_token=env.get("COURTLISTENER_API_TOKEN", "").strip(),
        )


@dataclass(frozen=True)
class FederalRegisterResult:
    """Published documents plus documents on public inspection (filed, not yet published)."""

    published: tuple[dict[str, Any], ...]
    public_inspection: tuple[dict[str, Any], ...]
    public_inspection_error: str | None  # why the pending list is missing, if it is


def _money(value: Any) -> float | None:
    """FEC sends money as numbers or numeric strings; ``None`` when absent or unparsable."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class GovClient:
    """Queries legislative, regulatory, campaign-finance and judicial APIs; never closes the HTTP client."""

    def __init__(self, settings: GovSettings, http: httpx.AsyncClient) -> None:
        """Create the client with settings and an open async HTTP client."""
        self._settings = settings
        self._http = http

    async def federal_register_search(
        self, term: str, include_public_inspection: bool = True, limit: int = 10
    ) -> FederalRegisterResult:
        """Newest published rules, notices and presidential documents, plus pending public-inspection docs.

        A public-inspection failure does not fail the call; it is reported in the result.
        """
        url = self._settings.federal_register_url
        published = await self._get(
            "Federal Register",
            f"{url}/documents.json",
            {"conditions[term]": term.strip(), "per_page": limit, "order": "newest"},
        )
        pending: tuple[dict[str, Any], ...] = ()
        pending_error = None
        if include_public_inspection:
            try:
                body = await self._get(
                    "Federal Register", f"{url}/public-inspection-documents.json", {"conditions[term]": term.strip()}
                )
                pending = tuple(body.get("results") or [])[:limit]
            except GovError as exc:
                pending_error = str(exc)
        return FederalRegisterResult(tuple(published.get("results") or []), pending, pending_error)

    async def congress_bill_status(self, congress: int, bill_type: BillType, number: int) -> dict[str, Any]:
        """Bill detail with ``latestAction``, e.g. ``(119, "hr", 1)``.

        Raises:
            GovError: No ``CONGRESS_API_KEY``, or Congress.gov failed/rejected the request.
        """
        body = await self._congress(f"bill/{congress}/{bill_type}/{number}")
        return self._unwrap(body, "bill", f"{bill_type.upper()}.{number} (Congress {congress})")

    async def congress_nomination(self, congress: int, number: int) -> dict[str, Any]:
        """Presidential nomination detail (PN number) with ``latestAction`` and nominees."""
        body = await self._congress(f"nomination/{congress}/{number}")
        return self._unwrap(body, "nomination", f"PN{number} (Congress {congress})")

    async def fec_candidate_financials(self, query: str, cycle: int, limit: int = 10) -> list[dict[str, Any]]:
        """Candidate totals for ``cycle``, richest cash-on-hand first.

        The API cannot sort by cash on hand (HTTP 422), so the top candidates by
        receipts are fetched and re-sorted here.
        """
        body = await self._get(
            "FEC",
            f"{self._settings.fec_url}/candidates/totals/",
            {
                "api_key": self._settings.fec_api_key,
                "q": query.strip(),
                "election_year": cycle,
                "per_page": limit,
                "sort": "-receipts",
            },
        )
        rows = list(body.get("results") or [])
        return sorted(rows, key=lambda r: _money(r.get("cash_on_hand_end_period")) or 0.0, reverse=True)

    async def courtlistener_dockets(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Federal (RECAP) dockets matching ``query``, newest filing first; ``results`` cut to ``limit``.

        Raises:
            GovError: No ``COURTLISTENER_API_TOKEN``, or CourtListener failed.
        """
        body = await self._courtlistener(
            "search/",
            {"q": query.strip(), "type": "r", "order_by": "dateFiled desc"},
        )
        return {"count": body.get("count"), "results": list(body.get("results") or [])[:limit]}

    async def courtlistener_opinions(
        self, query: str, court: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        """Judicial opinions (``type="o"``) matching ``query`` (and optional ``court`` slug), newest first."""
        params: dict[str, str | int] = {"q": query.strip(), "type": "o", "order_by": "dateFiled desc"}
        if court and court.strip():
            params["court"] = court.strip()
        body = await self._courtlistener("search/", params)
        return {"count": body.get("count"), "results": list(body.get("results") or [])[:limit]}

    async def courtlistener_docket_entries(self, docket_id: int, limit: int = 15) -> dict[str, Any]:
        """Recent filings, orders, stays, and injunctions on one CourtListener RECAP ``docket_id``."""
        if docket_id <= 0:
            raise GovError("docket_id must be a positive integer.")
        body = await self._courtlistener(
            "docket-entries/",
            {"docket": docket_id, "order_by": "-date_filed"},
        )
        return {"docket_id": docket_id, "count": body.get("count"), "results": list(body.get("results") or [])[:limit]}

    async def _courtlistener(self, path: str, params: Mapping[str, str | int]) -> dict[str, Any]:
        if not self._settings.courtlistener_token:
            raise GovError("COURTLISTENER_API_TOKEN is not set; CourtListener v4 requires a (free) token.")
        return await self._get(
            "CourtListener",
            f"{self._settings.courtlistener_url}/{path}",
            params,
            headers={"Authorization": f"Token {self._settings.courtlistener_token}"},
        )

    async def _congress(self, path: str) -> dict[str, Any]:
        if not self._settings.congress_api_key:
            raise GovError("CONGRESS_API_KEY is not set. Get a free key at https://api.congress.gov/sign-up/.")
        return await self._get(
            "Congress.gov",
            f"{self._settings.congress_url}/{path}",
            {"api_key": self._settings.congress_api_key, "format": "json"},
        )

    @staticmethod
    def _unwrap(body: dict[str, Any], key: str, label: str) -> dict[str, Any]:
        item = body.get(key)
        if not isinstance(item, dict):
            raise GovError(f"Congress.gov returned no {key} for {label}.")
        return item

    async def _get(
        self,
        source: str,
        url: str,
        params: Mapping[str, str | int],
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        logger.info("%s GET %s", source, {k: v for k, v in params.items() if k != "api_key"})
        try:
            response = await self._http.get(url, params=params, headers=headers, timeout=self._settings.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise GovError(f"{source} request timed out, please retry.") from exc
        except httpx.RequestError as exc:
            # str(exc) can embed the URL, and with it the key; report the type only.
            raise GovError(f"Could not reach {source} ({type(exc).__name__}).") from exc
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.is_success and isinstance(body, dict):
            return body
        if response.is_success:
            raise GovError(f"{source} returned an unexpected body (HTTP {response.status_code}).")
        raise GovError(f"{source} error {response.status_code}: {self._reason(body)}")

    @staticmethod
    def _reason(body: Any) -> str:
        """Pull the message out of the error shapes these APIs use (data.gov nests it under ``error``)."""
        if not isinstance(body, dict):
            return "unexpected response body"
        error = body.get("error")
        if isinstance(error, dict):
            error = error.get("message") or error.get("code")
        return str(body.get("message") or body.get("detail") or error or "unexpected response body")
