"""Self-improving SQLite epistemic ledger and Brier calibration engine.

No MCP code here. ``CalibrationMath`` is pure and deterministic; ``LedgerClient``
persists structured forecasts, active theses, invalidation triggers, and settled
outcomes in stdlib ``sqlite3`` (configured via ``QUANT_LEDGER_PATH``, defaulting
to ``.quant_ledger.sqlite3`` beside ``mcp_server.py``).
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

DEFAULT_LEDGER_PATH = str(Path(__file__).resolve().parent / ".quant_ledger.sqlite3")

MarketFamily = Literal["macro", "crypto_long", "corporate", "regulatory_legal", "politics", "tech"]
ForecastSide = Literal["YES", "NO", "BUY", "SELL", "PASS"]

VALID_FAMILIES: frozenset[str] = frozenset(get_args(MarketFamily))
VALID_SIDES: frozenset[str] = frozenset(get_args(ForecastSide))

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS forecasts (
    market_id TEXT PRIMARY KEY,
    market_slug TEXT NOT NULL,
    family TEXT NOT NULL,
    horizon_days REAL NOT NULL,
    market_prob REAL NOT NULL,
    fair_prob REAL NOT NULL,
    fair_prob_sd REAL NOT NULL,
    side TEXT NOT NULL,
    recommended_usd REAL NOT NULL,
    order_id TEXT,
    thesis_summary TEXT NOT NULL,
    key_sources_json TEXT NOT NULL,
    invalidation_triggers_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    outcome INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_forecasts_status_family ON forecasts(status, family);
"""


class LedgerError(Exception):
    """Any epistemic ledger validation, lookup, or SQLite persistence failure."""


@dataclass
class _MemoryConnectionHolder:
    """Keeps a single ``:memory:`` SQLite connection alive across ``Gateway.call()`` invocations."""

    conn: sqlite3.Connection | None = None


@dataclass(frozen=True)
class LedgerSettings:
    """SQLite ledger configuration loaded from ``QUANT_LEDGER_PATH``."""

    db_path: str = DEFAULT_LEDGER_PATH
    _mem: _MemoryConnectionHolder = field(default_factory=_MemoryConnectionHolder, repr=False, compare=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> LedgerSettings:
        """Read ``QUANT_LEDGER_PATH``, defaulting to ``.quant_ledger.sqlite3`` beside ``mcp_server.py``."""
        raw = env.get("QUANT_LEDGER_PATH", "").strip()
        return cls(db_path=raw or DEFAULT_LEDGER_PATH)


@dataclass(frozen=True)
class ForecastRecord:
    """One structured forecast row stored in the epistemic ledger."""

    market_id: str
    market_slug: str
    family: MarketFamily
    horizon_days: float
    market_prob: float
    fair_prob: float
    fair_prob_sd: float
    side: ForecastSide
    recommended_usd: float
    order_id: str | None
    thesis_summary: str
    key_sources: tuple[str, ...]
    invalidation_triggers: tuple[str, ...]
    status: Literal["open", "resolved"]
    outcome: int | None
    created_at: str
    updated_at: str
    resolved_at: str | None


@dataclass(frozen=True)
class DecileBucket:
    """Calibration statistics for one 10%-wide predicted YES probability bin."""

    label: str
    count: int
    mean_fair_prob: float
    actual_win_rate: float

    @property
    def calibration_gap(self) -> float:
        """``mean_fair_prob - actual_win_rate`` (>0 overconfident high, <0 underpredicted)."""
        return self.mean_fair_prob - self.actual_win_rate


@dataclass(frozen=True)
class FamilyCalibration:
    """Brier score, skill vs. market, and empirical ``fair_prob_sd`` multiplier for one family."""

    family: str
    count: int
    agent_brier: float
    market_brier: float
    brier_skill_score: float | None
    suggested_sd_multiplier: float


@dataclass(frozen=True)
class CalibrationReport:
    """Full Brier calibration scorecard across resolved forecasts."""

    resolved_just_now: ForecastRecord | None
    total_resolved: int
    agent_brier: float | None
    market_brier: float | None
    brier_skill_score: float | None
    families: tuple[FamilyCalibration, ...]
    deciles: tuple[DecileBucket, ...]


class CalibrationMath:
    """Pure Brier-score, Brier-skill-score, decile calibration, and uncertainty-scaling math."""

    @staticmethod
    def score(
        records: Sequence[ForecastRecord],
        resolved_just_now: ForecastRecord | None = None,
    ) -> CalibrationReport:
        """Score settled forecasts (`outcome` in {0, 1}) overall, per family, and by decile."""
        settled = [r for r in records if r.status == "resolved" and r.outcome in (0, 1)]
        if not settled:
            return CalibrationReport(resolved_just_now, 0, None, None, None, (), ())

        agent_brier = statistics.fmean((r.fair_prob - float(r.outcome or 0)) ** 2 for r in settled)
        market_brier = statistics.fmean((r.market_prob - float(r.outcome or 0)) ** 2 for r in settled)
        bss = 1.0 - agent_brier / market_brier if market_brier > 1e-12 else None

        by_fam: dict[str, list[ForecastRecord]] = {}
        for r in settled:
            by_fam.setdefault(r.family, []).append(r)

        fam_rows: list[FamilyCalibration] = []
        for fam in sorted(by_fam):
            rows = by_fam[fam]
            f_agent = statistics.fmean((r.fair_prob - float(r.outcome or 0)) ** 2 for r in rows)
            f_mkt = statistics.fmean((r.market_prob - float(r.outcome or 0)) ** 2 for r in rows)
            f_bss = 1.0 - f_agent / f_mkt if f_mkt > 1e-12 else None
            mult = CalibrationMath._suggest_sd_multiplier(rows, f_agent, f_mkt, f_bss)
            fam_rows.append(FamilyCalibration(fam, len(rows), f_agent, f_mkt, f_bss, mult))

        buckets: dict[int, list[ForecastRecord]] = {}
        for r in settled:
            idx = min(9, max(0, int(r.fair_prob * 10)))
            buckets.setdefault(idx, []).append(r)

        deciles = tuple(
            DecileBucket(
                label=f"{idx / 10:.1f}-{(idx + 1) / 10:.1f}",
                count=len(buckets[idx]),
                mean_fair_prob=statistics.fmean(r.fair_prob for r in buckets[idx]),
                actual_win_rate=statistics.fmean(float(r.outcome or 0) for r in buckets[idx]),
            )
            for idx in sorted(buckets)
        )

        return CalibrationReport(
            resolved_just_now=resolved_just_now,
            total_resolved=len(settled),
            agent_brier=agent_brier,
            market_brier=market_brier,
            brier_skill_score=bss,
            families=tuple(fam_rows),
            deciles=deciles,
        )

    @staticmethod
    def _suggest_sd_multiplier(
        rows: Sequence[ForecastRecord],
        agent_brier: float,
        market_brier: float,
        bss: float | None,
    ) -> float:
        """Empirical multiplier on ``fair_prob_sd`` bounded to ``[0.5, 3.0]``.

        Excess squared error above irreducible Bernoulli noise ``p*(1-p)`` is compared
        against mean stated variance ``sd^2``. Families with negative skill vs. market
        are floored at ``1.25x`` (shrinking Kelly), and small samples (``n < 3``) are
        floored at ``1.0x``.
        """
        bernoulli_var = statistics.fmean(r.fair_prob * (1.0 - r.fair_prob) for r in rows)
        mean_sd_sq = statistics.fmean(max(r.fair_prob_sd, 0.01) ** 2 for r in rows)
        excess_var = max(0.0, agent_brier - bernoulli_var)
        raw = math.sqrt(excess_var / mean_sd_sq) if excess_var > 0 else 0.5
        mult = min(3.0, max(0.5, raw))
        if bss is not None and bss < 0.0:
            ratio = math.sqrt(agent_brier / market_brier) if market_brier > 1e-12 else 1.25
            mult = min(3.0, max(1.25, mult, ratio))
        if len(rows) < 3:
            mult = max(1.0, mult)
        return round(mult, 2)


def _normalize_list(items: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(items, str):
        parts = [p.strip() for p in items.replace("\n", ";").split(";") if p.strip()]
        return tuple(parts)
    return tuple(str(x).strip() for x in items if str(x).strip())


class LedgerClient:
    """SQLite-backed epistemic ledger for slow-horizon thesis tracking and Brier calibration."""

    def __init__(self, settings: LedgerSettings) -> None:
        """Create the ledger client with ``settings``."""
        self._settings = settings

    def record_forecast(
        self,
        *,
        market_id: str,
        market_slug: str,
        family: MarketFamily,
        horizon_days: float,
        market_prob: float,
        fair_prob: float,
        fair_prob_sd: float = 0.05,
        side: ForecastSide = "PASS",
        recommended_usd: float = 0.0,
        order_id: str | None = None,
        thesis_summary: str = "",
        key_sources: Sequence[str] | str = (),
        invalidation_triggers: Sequence[str] | str = (),
    ) -> ForecastRecord:
        """Validate and upsert an open forecast keyed by ``market_id``."""
        mid = market_id.strip()
        slug = market_slug.strip()
        if not mid or not slug:
            raise LedgerError("market_id and market_slug are required.")
        if family not in VALID_FAMILIES:
            raise LedgerError(f"family must be one of {sorted(VALID_FAMILIES)}, got {family!r}.")
        if side not in VALID_SIDES:
            raise LedgerError(f"side must be one of {sorted(VALID_SIDES)}, got {side!r}.")
        if not math.isfinite(horizon_days) or horizon_days <= 0:
            raise LedgerError("horizon_days must be a positive finite number.")
        if not (math.isfinite(market_prob) and 0.0 < market_prob < 1.0):
            raise LedgerError("market_prob must be a YES probability strictly in (0, 1).")
        if not (math.isfinite(fair_prob) and 0.0 < fair_prob < 1.0):
            raise LedgerError("fair_prob must be a YES probability strictly in (0, 1).")
        if not (math.isfinite(fair_prob_sd) and 0.0 <= fair_prob_sd < 1.0):
            raise LedgerError("fair_prob_sd must lie in [0, 1).")
        if not math.isfinite(recommended_usd) or recommended_usd < 0.0:
            raise LedgerError("recommended_usd cannot be negative.")

        sources = _normalize_list(key_sources)
        triggers = _normalize_list(invalidation_triggers)
        clean_order = order_id.strip() if order_id and order_id.strip() else None
        now = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            existing = conn.execute("SELECT status FROM forecasts WHERE market_id = ?", (mid,)).fetchone()
            if existing is not None and existing["status"] == "resolved":
                raise LedgerError(
                    f"Market {mid!r} is already resolved in the ledger; refusing to overwrite settled calibration data."
                )
            conn.execute(
                """
                INSERT INTO forecasts (
                    market_id, market_slug, family, horizon_days, market_prob, fair_prob, fair_prob_sd,
                    side, recommended_usd, order_id, thesis_summary, key_sources_json,
                    invalidation_triggers_json, status, outcome, created_at, updated_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, ?, ?, NULL)
                ON CONFLICT(market_id) DO UPDATE SET
                    market_slug = excluded.market_slug,
                    family = excluded.family,
                    horizon_days = excluded.horizon_days,
                    market_prob = excluded.market_prob,
                    fair_prob = excluded.fair_prob,
                    fair_prob_sd = excluded.fair_prob_sd,
                    side = excluded.side,
                    recommended_usd = excluded.recommended_usd,
                    order_id = COALESCE(excluded.order_id, forecasts.order_id),
                    thesis_summary = excluded.thesis_summary,
                    key_sources_json = excluded.key_sources_json,
                    invalidation_triggers_json = excluded.invalidation_triggers_json,
                    updated_at = excluded.updated_at
                """,
                (
                    mid,
                    slug,
                    family,
                    horizon_days,
                    market_prob,
                    fair_prob,
                    fair_prob_sd,
                    side,
                    recommended_usd,
                    clean_order,
                    thesis_summary.strip(),
                    json.dumps(sources),
                    json.dumps(triggers),
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM forecasts WHERE market_id = ?", (mid,)).fetchone()
        return self._row_to_record(row)

    def get_active_theses(
        self,
        market_id: str | None = None,
        family: MarketFamily | None = None,
    ) -> tuple[ForecastRecord, ...]:
        """Return unresolved ('open') theses, newest update first."""
        if family is not None and family not in VALID_FAMILIES:
            raise LedgerError(f"family must be one of {sorted(VALID_FAMILIES)}, got {family!r}.")
        clauses = ["status = 'open'"]
        params: list[str] = []
        if market_id and market_id.strip():
            clauses.append("market_id = ?")
            params.append(market_id.strip())
        if family:
            clauses.append("family = ?")
            params.append(family)
        sql = f"SELECT * FROM forecasts WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return tuple(self._row_to_record(r) for r in rows)

    def resolve_and_score(
        self,
        market_id: str | None = None,
        outcome: int | None = None,
        family: MarketFamily | None = None,
    ) -> CalibrationReport:
        """Optionally mark ``market_id`` settled (``outcome`` 0=NO, 1=YES) and score Brier calibration."""
        if (market_id is None) != (outcome is None):
            raise LedgerError(
                "Provide both market_id and outcome (0 or 1) to settle a market, or omit both to score existing history."
            )
        if outcome is not None and (isinstance(outcome, bool) or outcome not in (0, 1)):
            raise LedgerError(f"outcome must be 0 (NO) or 1 (YES), got {outcome!r}.")
        if family is not None and family not in VALID_FAMILIES:
            raise LedgerError(f"family must be one of {sorted(VALID_FAMILIES)}, got {family!r}.")

        just_resolved: ForecastRecord | None = None
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            if market_id is not None and outcome is not None:
                mid = market_id.strip()
                cur = conn.execute(
                    "UPDATE forecasts SET status = 'resolved', outcome = ?, resolved_at = ?, updated_at = ? WHERE market_id = ?",
                    (outcome, now, now, mid),
                )
                if cur.rowcount == 0:
                    raise LedgerError(f"Market {mid!r} not found in ledger; call record_forecast before resolving.")
                row = conn.execute("SELECT * FROM forecasts WHERE market_id = ?", (mid,)).fetchone()
                just_resolved = self._row_to_record(row)

            if family:
                rows = conn.execute(
                    "SELECT * FROM forecasts WHERE status = 'resolved' AND family = ? ORDER BY resolved_at DESC",
                    (family,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM forecasts WHERE status = 'resolved' ORDER BY resolved_at DESC"
                ).fetchall()

        records = [self._row_to_record(r) for r in rows]
        return CalibrationMath.score(records, resolved_just_now=just_resolved)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._settings.db_path == ":memory:":
            try:
                if self._settings._mem.conn is None:
                    conn = sqlite3.connect(":memory:")
                    conn.row_factory = sqlite3.Row
                    conn.executescript(_SCHEMA_SQL)
                    self._settings._mem.conn = conn
                conn = self._settings._mem.conn
                yield conn
                conn.commit()
            except sqlite3.Error as exc:
                if self._settings._mem.conn is not None:
                    self._settings._mem.conn.rollback()
                raise LedgerError(f"SQLite ledger error ({type(exc).__name__}): {exc}") from exc
            return

        db_file = Path(self._settings.db_path)
        try:
            db_file.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_file))
        except (OSError, sqlite3.Error) as exc:
            raise LedgerError(f"Cannot open ledger at {self._settings.db_path!r}: {exc}") from exc

        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(_SCHEMA_SQL)
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise LedgerError(f"SQLite ledger error ({type(exc).__name__}): {exc}") from exc
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ForecastRecord:
        sources_raw = json.loads(row["key_sources_json"] or "[]")
        triggers_raw = json.loads(row["invalidation_triggers_json"] or "[]")
        fam: MarketFamily = row["family"]
        side: ForecastSide = row["side"]
        status: Literal["open", "resolved"] = row["status"]
        return ForecastRecord(
            market_id=str(row["market_id"]),
            market_slug=str(row["market_slug"]),
            family=fam,
            horizon_days=float(row["horizon_days"]),
            market_prob=float(row["market_prob"]),
            fair_prob=float(row["fair_prob"]),
            fair_prob_sd=float(row["fair_prob_sd"]),
            side=side,
            recommended_usd=float(row["recommended_usd"]),
            order_id=str(row["order_id"]) if row["order_id"] else None,
            thesis_summary=str(row["thesis_summary"]),
            key_sources=tuple(str(x) for x in sources_raw),
            invalidation_triggers=tuple(str(x) for x in triggers_raw),
            status=status,
            outcome=int(row["outcome"]) if row["outcome"] is not None else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            resolved_at=str(row["resolved_at"]) if row["resolved_at"] else None,
        )
