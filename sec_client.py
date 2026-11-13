"""SEC EDGAR filings client (EFTS search + company submissions) and M&A spread math.

No MCP code here. ``MergerMath`` is pure and deterministic; ``SecClient`` queries
SEC EDGAR's public JSON endpoints with a polite rate limit (<= 10 requests/s) and
required ``User-Agent`` header (configured via ``SEC_USER_AGENT``).

References:
    https://www.sec.gov/os/webmaster-faq#developers
    https://efts.sec.gov/LATEST/search-index
    https://data.sec.gov/submissions/CIK0000320193.json
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx2 as httpx

logger = logging.getLogger(__name__)

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_SEC_USER_AGENT = "PolymarketQuantMCP/0.1 (set-SEC_USER_AGENT-in-env@polymarket-mcp.local)"
SEC_SEARCH_MAX = 50
SEC_SUBMISSIONS_MAX = 50

DEFAULT_SEARCH_FORMS: tuple[str, ...] = ("8-K", "S-4", "DEFM14A", "13D", "10-Q", "10-K")
DEFAULT_SUBMISSION_FORMS: tuple[str, ...] = ("8-K", "10-Q", "10-K", "S-4")

EIGHT_K_ITEMS: dict[str, str] = {
    "1.01": "Entry into Material Definitive Agreement",
    "1.02": "Termination of Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition",
    "2.02": "Results of Operations and Financial Condition",
    "2.05": "Restructuring / Exit Activities",
    "2.06": "Material Impairments",
    "3.01": "Delisting or Exchange Rule Notice",
    "4.01": "Changes in Certifying Accountant",
    "4.02": "Non-Reliance on Prior Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Officer/Director Departure or Appointment",
    "5.07": "Shareholder Vote Results",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Material Events",
    "9.01": "Financial Statements and Exhibits",
}

_CIK_RE = re.compile(r"^(?:CIK)?(\d{1,10})$", re.IGNORECASE)
_LAST_SEC_REQ_TS: float = 0.0
_TICKER_CACHE: dict[str, dict[str, tuple[str, str]]] = {}


class SecError(Exception):
    """Any SEC EDGAR failure (HTTP, malformed payload, unknown ticker) or invalid M&A math input."""


@dataclass(frozen=True)
class SecSettings:
    """SEC EDGAR settings; no API key required, ``SEC_USER_AGENT`` identifies the caller."""

    user_agent: str = DEFAULT_SEC_USER_AGENT
    using_default_user_agent: bool = True
    efts_url: str = EFTS_URL
    submissions_url: str = SUBMISSIONS_URL
    tickers_url: str = TICKERS_URL
    archives_url: str = ARCHIVES_URL
    min_interval_seconds: float = 0.11  # <= 9.1 req/s, respecting SEC's 10 req/s ceiling
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> SecSettings:
        """Read ``SEC_USER_AGENT`` (e.g. ``'YourOrg research@yourdomain.com'``) with safe fallback."""
        ua = env.get("SEC_USER_AGENT", "").strip()
        return cls(
            user_agent=ua or DEFAULT_SEC_USER_AGENT,
            using_default_user_agent=not bool(ua),
            efts_url=env.get("SEC_EFTS_URL", EFTS_URL),
            submissions_url=env.get("SEC_SUBMISSIONS_URL", SUBMISSIONS_URL).rstrip("/"),
            tickers_url=env.get("SEC_TICKERS_URL", TICKERS_URL),
            archives_url=env.get("SEC_ARCHIVES_URL", ARCHIVES_URL).rstrip("/"),
        )


@dataclass(frozen=True)
class SecFilingHit:
    """One filing hit from SEC EDGAR EFTS full-text search."""

    filed_date: str
    form: str
    entity: str
    cik: str
    accession_number: str
    items: tuple[str, ...]
    description: str
    document_url: str


@dataclass(frozen=True)
class SecSearchResult:
    """EFTS search response plus ``using_default_user_agent`` notice flag."""

    query: str
    total_hits: int
    hits: tuple[SecFilingHit, ...]
    using_default_user_agent: bool = False


@dataclass(frozen=True)
class CompanySubmission:
    """One recent company submission from ``data.sec.gov/submissions/CIK...``."""

    filing_date: str
    form: str
    accession_number: str
    items: tuple[str, ...]
    title: str
    document_url: str


@dataclass(frozen=True)
class CompanySubmissionsReport:
    """Recent submissions for one resolved company CIK."""

    cik: str
    entity_name: str
    tickers: tuple[str, ...]
    submissions: tuple[CompanySubmission, ...]
    using_default_user_agent: bool = False


@dataclass(frozen=True)
class MergerSpreadAnalysis:
    """Deterministic M&A deal-closing probability, spread, and annualized IRR."""

    target_price: float
    deal_offer_value: float
    downside_break_price: float
    raw_implied_prob: float
    implied_prob: float
    gross_spread_usd: float
    gross_spread_pct: float
    downside_pct: float
    days_to_close: float | None = None
    annualized_simple: float | None = None
    annualized_irr: float | None = None
    regime_note: str = ""


class MergerMath:
    """Pure deterministic merger-arbitrage pricing math."""

    @staticmethod
    def analyze(
        target_price: float,
        deal_offer_value: float,
        downside_break_price: float,
        days_to_close: float | None = None,
    ) -> MergerSpreadAnalysis:
        """Compute implied deal closing probability ``(P - Break) / (Offer - Break)``, spread, and IRR.

        Raises:
            SecError: Non-finite values, non-positive target/offer, ``offer <= break``,
                or non-positive ``days_to_close``.
        """
        for label, val in (
            ("target_price", target_price),
            ("deal_offer_value", deal_offer_value),
            ("downside_break_price", downside_break_price),
        ):
            if not math.isfinite(val):
                raise SecError(f"{label} must be a finite number.")
        if target_price <= 0 or deal_offer_value <= 0:
            raise SecError("target_price and deal_offer_value must be positive.")
        if downside_break_price < 0:
            raise SecError("downside_break_price cannot be negative.")
        if deal_offer_value <= downside_break_price:
            raise SecError(
                f"deal_offer_value ({deal_offer_value}) must exceed downside_break_price ({downside_break_price})."
            )
        if days_to_close is not None and (not math.isfinite(days_to_close) or days_to_close <= 0):
            raise SecError("days_to_close must be a positive finite number.")

        raw_prob = (target_price - downside_break_price) / (deal_offer_value - downside_break_price)
        implied_prob = min(1.0, max(0.0, raw_prob))
        spread_usd = deal_offer_value - target_price
        spread_pct = spread_usd / target_price
        downside_pct = (downside_break_price - target_price) / target_price

        note = ""
        if target_price > deal_offer_value:
            note = "TRADING_ABOVE_OFFER: target exceeds offer (competing-bid / bump premium priced in; prob clamped to 1.0)"
        elif target_price < downside_break_price:
            note = "BELOW_BREAK_PRICE: target trades below break estimate (harsher standalone downside priced in; prob clamped to 0.0)"

        ann_simple: float | None = None
        ann_irr: float | None = None
        if days_to_close is not None:
            exponent = 365.0 / days_to_close
            ann_simple = spread_pct * exponent
            try:
                ann_irr = (deal_offer_value / target_price) ** exponent - 1.0
            except OverflowError as exc:
                raise SecError("Annualized IRR overflowed because days_to_close is too small.") from exc

        return MergerSpreadAnalysis(
            target_price=target_price,
            deal_offer_value=deal_offer_value,
            downside_break_price=downside_break_price,
            raw_implied_prob=raw_prob,
            implied_prob=implied_prob,
            gross_spread_usd=spread_usd,
            gross_spread_pct=spread_pct,
            downside_pct=downside_pct,
            days_to_close=days_to_close,
            annualized_simple=ann_simple,
            annualized_irr=ann_irr,
            regime_note=note,
        )


def _format_items(raw_items: Any) -> tuple[str, ...]:
    """Normalize 8-K item codes (e.g. ``'1.01'`` or ``'Item 5.02'``) and attach concise labels."""
    if isinstance(raw_items, str):
        tokens = [t.strip() for t in raw_items.split(",") if t.strip()]
    elif isinstance(raw_items, list | tuple):
        tokens = [str(t).strip() for t in raw_items if str(t).strip()]
    else:
        return ()
    out: list[str] = []
    for tok in tokens:
        code = re.sub(r"^item\s+", "", tok, flags=re.IGNORECASE).strip()
        label = EIGHT_K_ITEMS.get(code)
        out.append(f"{code} ({label})" if label else code)
    return tuple(out)


def _archive_url(archives_base: str, cik: str, accession_number: str, doc_name: str | None) -> str:
    """Build direct SEC EDGAR archive URL ready for ``firecrawl_scrape``."""
    adsh_clean = accession_number.replace("-", "")
    if not cik or not adsh_clean:
        return ""
    try:
        cik_int = int(cik)
    except ValueError:
        return ""
    doc = (doc_name or "").strip() or f"{accession_number}-index.htm"
    return f"{archives_base}/{cik_int}/{adsh_clean}/{doc}"


class SecClient:
    """Async SEC EDGAR EFTS & company-submissions client; never closes the injected HTTP client."""

    def __init__(self, settings: SecSettings, http: httpx.AsyncClient) -> None:
        """Create the client with settings and an open async HTTP client."""
        self._settings = settings
        self._http = http
        self._lock = asyncio.Lock()

    async def search_filings(
        self,
        query: str,
        forms: Sequence[str] = DEFAULT_SEARCH_FORMS,
        start_date: date | None = None,
        limit: int = 10,
    ) -> SecSearchResult:
        """Full-text search across SEC EDGAR filings via EFTS ``/LATEST/search-index``."""
        clean_q = query.strip()
        if not clean_q:
            raise SecError("SEC search query cannot be blank.")
        params: dict[str, str | int] = {"q": clean_q, "from": 0, "size": limit}
        clean_forms = [f.strip().upper() for f in forms if f.strip()]
        if clean_forms:
            params["forms"] = ",".join(clean_forms)
        if start_date is not None:
            params["dateRange"] = "custom"
            params["startdt"] = start_date.isoformat()

        body = await self._get("SEC EFTS", self._settings.efts_url, params)
        hits_block = body.get("hits")
        if not isinstance(hits_block, dict) or not isinstance(hits_block.get("hits"), list):
            raise SecError("SEC EFTS returned an unexpected response structure.")

        raw_total = hits_block.get("total")
        total = int(raw_total.get("value", 0)) if isinstance(raw_total, dict) else int(raw_total or 0)

        results: list[SecFilingHit] = []
        for entry in hits_block["hits"][:limit]:
            if not isinstance(entry, dict):
                continue
            src = entry.get("_source") or {}
            if not isinstance(src, dict):
                continue
            hit_id = str(entry.get("_id") or "")
            adsh = str(src.get("adsh") or (hit_id.split(":", 1)[0] if ":" in hit_id else ""))
            doc = hit_id.split(":", 1)[1] if ":" in hit_id else None
            ciks = src.get("ciks") or []
            cik = str(ciks[0]) if isinstance(ciks, list) and ciks else ""
            names = src.get("display_names") or []
            entity = str(names[0]) if isinstance(names, list) and names else (cik or "unknown")
            results.append(
                SecFilingHit(
                    filed_date=str(src.get("file_date") or "n/a"),
                    form=str(src.get("form") or src.get("file_type") or "?"),
                    entity=entity,
                    cik=cik,
                    accession_number=adsh,
                    items=_format_items(src.get("items")),
                    description=str(src.get("file_description") or ""),
                    document_url=_archive_url(self._settings.archives_url, cik, adsh, doc),
                )
            )
        return SecSearchResult(
            query=clean_q,
            total_hits=total,
            hits=tuple(results),
            using_default_user_agent=self._settings.using_default_user_agent,
        )

    async def company_submissions(
        self,
        ticker_or_cik: str,
        forms: Sequence[str] = DEFAULT_SUBMISSION_FORMS,
        limit: int = 15,
    ) -> CompanySubmissionsReport:
        """Resolve ticker or CIK and return recent SEC filings with 8-K item tags and archive URLs."""
        cik, fallback_title = await self._resolve_cik(ticker_or_cik)
        body = await self._get("SEC Submissions", f"{self._settings.submissions_url}/CIK{cik}.json", None)
        recent = (body.get("filings") or {}).get("recent") if isinstance(body.get("filings"), dict) else None
        if not isinstance(recent, dict) or not isinstance(recent.get("form"), list):
            raise SecError(f"SEC Submissions returned no recent filings table for CIK {cik}.")

        allowed = {f.strip().upper() for f in forms if f.strip()}
        forms_col: list[Any] = recent.get("form") or []
        dates_col: list[Any] = recent.get("filingDate") or []
        acc_col: list[Any] = recent.get("accessionNumber") or []
        doc_col: list[Any] = recent.get("primaryDocument") or []
        desc_col: list[Any] = recent.get("primaryDocDescription") or []
        items_col: list[Any] = recent.get("items") or []

        submissions: list[CompanySubmission] = []
        for i, raw_form in enumerate(forms_col):
            form = str(raw_form or "").strip().upper()
            if allowed and form not in allowed and form.removesuffix("/A") not in allowed:
                continue
            adsh = str(acc_col[i]) if i < len(acc_col) else ""
            doc = str(doc_col[i]) if i < len(doc_col) else ""
            fdate = str(dates_col[i]) if i < len(dates_col) else "n/a"
            title = str(desc_col[i]) if i < len(desc_col) and desc_col[i] else form
            raw_items = items_col[i] if i < len(items_col) else ()
            submissions.append(
                CompanySubmission(
                    filing_date=fdate,
                    form=form,
                    accession_number=adsh,
                    items=_format_items(raw_items),
                    title=title,
                    document_url=_archive_url(self._settings.archives_url, cik, adsh, doc),
                )
            )
            if len(submissions) >= limit:
                break

        raw_tickers = body.get("tickers")
        tickers = tuple(str(t) for t in raw_tickers) if isinstance(raw_tickers, list) else ()
        entity_name = str(body.get("name") or fallback_title or f"CIK {cik}")
        return CompanySubmissionsReport(
            cik=cik,
            entity_name=entity_name,
            tickers=tickers,
            submissions=tuple(submissions),
            using_default_user_agent=self._settings.using_default_user_agent,
        )

    async def _resolve_cik(self, ticker_or_cik: str) -> tuple[str, str | None]:
        raw = ticker_or_cik.strip()
        if not raw:
            raise SecError("ticker_or_cik cannot be blank.")
        m = _CIK_RE.fullmatch(raw)
        if m:
            return f"{int(m.group(1)):010d}", None

        cache = _TICKER_CACHE.get(self._settings.tickers_url)
        if cache is None:
            body = await self._get("SEC Company Tickers", self._settings.tickers_url, None)
            cache = {}
            for entry in body.values():
                if isinstance(entry, dict) and "ticker" in entry and "cik_str" in entry:
                    try:
                        padded = f"{int(entry['cik_str']):010d}"
                    except (TypeError, ValueError):
                        continue
                    cache[str(entry["ticker"]).strip().upper()] = (padded, str(entry.get("title") or ""))
            _TICKER_CACHE[self._settings.tickers_url] = cache

        key = raw.upper()
        hit = cache.get(key) or cache.get(key.replace(".", "-")) or cache.get(key.replace("-", "."))
        if hit is None:
            raise SecError(f"Ticker {raw!r} not found in SEC company_tickers.json; pass a 10-digit CIK directly.")
        return hit

    async def _throttle(self) -> None:
        global _LAST_SEC_REQ_TS
        if self._settings.min_interval_seconds <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._settings.min_interval_seconds - (now - _LAST_SEC_REQ_TS)
            if wait > 0:
                await asyncio.sleep(wait)
            _LAST_SEC_REQ_TS = time.monotonic()

    async def _get(
        self,
        source: str,
        url: str,
        params: Mapping[str, str | int] | None,
    ) -> dict[str, Any]:
        await self._throttle()
        logger.info("%s GET %s", source, dict(params or {}))
        headers = {"User-Agent": self._settings.user_agent, "Accept": "application/json"}
        try:
            response = await self._http.get(
                url, params=params, headers=headers, timeout=self._settings.timeout_seconds
            )
        except httpx.TimeoutException as exc:
            raise SecError(f"{source} request timed out, please retry.") from exc
        except httpx.RequestError as exc:
            raise SecError(f"Could not reach {source} ({type(exc).__name__}).") from exc
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.is_success and isinstance(body, dict):
            return body
        if response.is_success:
            raise SecError(f"{source} returned an unexpected body (HTTP {response.status_code}).")
        if response.status_code == 403:
            raise SecError(
                f"{source} error 403 (Forbidden): set SEC_USER_AGENT='YourOrg contact@yourdomain.com' in .env "
                "per SEC EDGAR fair-access policy (<=10 req/s)."
            )
        detail = ""
        if isinstance(body, dict):
            detail = str(body.get("message") or body.get("error") or "")
        raise SecError(f"{source} error {response.status_code}: {detail or 'unexpected response body'}")
