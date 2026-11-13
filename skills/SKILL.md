---
name: polymarket-quant
description: Slow-horizon (days-to-months) quantitative research, settlement-rule auditing, cross-venue pricing, Brier-calibrated forecasting, and maker execution on Polymarket using Polymarket Quant MCP, Falcon MCP, and Firecrawl MCP. Use when researching prediction markets, hunting mispricings in macro/crypto/corporate/legal/politics/tech events, auditing resolution rules, sizing Kelly bets, managing resting orders, or running postmortem Brier calibration.
version: 1.0.0
metadata:
  hermes:
    tags: [polymarket, kalshi, quant, prediction-markets, forecasting, calibration, mcp]
---

# Polymarket Slow-Horizon Quantitative Research & Execution Skill

## 1. Target Regime Doctrine

LLM agents fail systematically when competing against co-located Rust/C++ bots on sub-minute latency arbitrage, sports live-betting, or 15-minute crypto candles. By the time an LLM reads a breaking headline, the CLOB top-of-book has already repriced and crossing the spread incurs adverse selection plus taker slippage.

**Target Regime — Slow-Horizon Structural Markets (Days to Months):**
- **Informational Half-Life of Hours to Weeks:** Target markets where fair value depends on synthesizing primary documents (SEC S-4 merger covenants, CourtListener RECAP briefing schedules, Federal Register Public Inspection desks, Deribit multi-month option surfaces, RCP poll dropout mechanics) that retail bettors do not read and speed bots cannot parse structurally.
- **Horizon-Aware Capital Efficiency:** A `+3c` edge resolving in 4 days (`+75 bps/day`) compounds far faster than a `+4c` edge locked for 240 days (`+1.7 bps/day`). Always pass `days_to_resolution` to `simulate_clob_fill` and reject low-carry capital lockups unless annualized log-growth and deal IRR justify collateral tie-up.
- **Passive GTC Maker Execution:** Prefer posting resting Good-Til-Cancelled (`GTC`) limit orders inside or at the passive side of the bid/ask (`place_order`), earning the spread instead of paying it, and actively pulling orders via `cancel_order` whenever a thesis `invalidation_trigger` fires.

---

## 2. 3-Server Tool Routing Matrix

Never use a generic web snippet when a deterministic primary oracle or specialized cross-venue MCP tool exists:

| Capability Layer | **Polymarket Quant MCP** (Primary Oracles, Math, Ledger, Execution) | **Falcon MCP** (`polygunllc.mintlify.app` — Cross-Venue & Smart Money) | **Firecrawl MCP** (Full-Document & Table Extraction) |
| :--- | :--- | :--- | :--- |
| **Discovery & Structure** | `search_markets` (paginated CLOB liquid scan), `search_events` (`/public-search` + `/events`), `get_market_resolution_rules` (UMA audit), `scan_neg_risk_event` | `Market 360`, `Candlesticks` (historical Polymarket OHLCV & volume regimes), `Kalshi Markets` | `firecrawl_map` (changelog/docs URL discovery) |
| **Smart Money & Sentiment** | `get_market_holders` (top-N share, HHI, weighted avg entry), `search_x_context`, `search_gdelt_news`, `get_gdelt_volume` | `Wallet 360`, `F-Score` leaderboards (validate if `get_market_holders` whales are true sharps), `Social Pulse` | `firecrawl_search` (long-form investigative/analyst writeups) |
| **Primary Oracles** | `get_series`, `search_series`, `get_fed_releases` (macro); `search_sec_filings`, `get_company_submissions` (EDGAR); `search_federal_register`, `get_bill_status`, `get_nomination_status`, `get_candidate_financials`, `search_court_dockets`, `get_docket_entries`, `search_court_opinions` | Regulated US event benchmarks (`Kalshi Markets` orderbooks & settlement history) | `firecrawl_scrape` (`formats=['markdown']`) on SEC Archive URLs, RECAP court PDFs, RCP poll tables, FDA/FCC letters |
| **Deterministic Pricing** | `price_crypto_expiry` (`expiry` & `touch`, 1h/1d + Deribit IV skew), `calculate_merger_spread_prob`, `analyze_poll_dropout`, `get_bookmaker_odds`, `search_predictit`, `simulate_clob_fill` | Cross-venue implied prob & liquidity comparison (`Kalshi` vs `Polymarket`) | Structured table extraction feeding `analyze_poll_dropout` |
| **Calibration & Execution** | `get_active_theses`, `record_forecast`, `resolve_and_score_calibration`, `place_order`, `list_open_orders`, `cancel_order` (Deposit Wallet sig type 3) | Read-only analytics (never executes orders) | Read-only web extraction |

**Rate-Limit & Quota Discipline:**
- `GDELT` (`search_gdelt_news`, `get_gdelt_volume`): strictly **<= 1 request per 5 seconds**.
- `PolyNode X` (`search_x_context`): **1 request/s**, metered monthly quota—query primary gov/SEC/FRED tools first.
- `SEC EDGAR` (`search_sec_filings`, `get_company_submissions`): **<= 10 requests/s**, ensure `SEC_USER_AGENT` is set.
- `The Odds API` (`get_bookmaker_odds`): **500 credits/month** on free tier—call once per politics research pass.

---

## 3. 6-Gate Autonomous Research-to-Trade Protocol

Execute every gate in order. Abort immediately (`side="PASS"` in `record_forecast`) if any gate fails.

### Gate 1 — Ledger, Calibration & Exposure Check
1. Call `get_active_theses` (filtered by `market_id` or `family`) and, when `POLY_PRIVATE_KEY` is configured, `list_open_orders` to prevent duplicate exposure or conflicting orders.
2. Call `resolve_and_score_calibration(family=...)` to retrieve the family's empirical `suggested_sd_mult` (default `1.0x` when no settled history exists).

### Gate 2 — Settlement Rule & `p3` Ambiguity Audit
1. Call `get_market_resolution_rules(market_id_or_slug)`. Read the **entire** `rules:` text, `source:`, `negRisk`, `tick_size`, and `min_order_size`.
2. Inspect every `AMBIGUITY FLAGS` warning (`TIMEZONE_CUTOFF`, `ANNOUNCED_VS_EFFECTIVE`, `TOUCH_VS_CLOSE`, `UNNAMED_CONSENSUS_SOURCE`, `P3_50_50_FALLBACK`, `NEG_RISK_OTHER_TRAP`).
3. If `source` names a specific external webpage/methodology, fetch it with Firecrawl `firecrawl_scrape(formats=['markdown'])`. If settlement hinges on unquantifiable subjective interpretation, **PASS**.

### Gate 3 — Primary-Source Base-Rate & Smart-Money Synthesis
1. Establish an outside-view historical base rate before reading commentary.
2. Pull primary documents from the domain oracle (`FRED`, `SEC EDGAR`, `CourtListener`, `Federal Register`, `Congress.gov`, `FEC`) and scrape key exhibits/orders via `firecrawl_scrape`.
3. Call `get_market_holders(market_slug=...)` and cross-check top wallet addresses in Falcon MCP `Wallet 360` / `F-Score`. If high-`F-Score` sharps are heavily concentrated **against** your thesis at recent entries, widen `fair_prob_sd` or re-audit what catalyst you missed.

### Gate 4 — Deterministic Model & Cross-Venue Benchmark
1. Run the family's deterministic model (`price_crypto_expiry`, `calculate_merger_spread_prob`, `analyze_poll_dropout`, `get_bookmaker_odds`, `scan_neg_risk_event`).
2. Benchmark implied YES probability against Falcon MCP `Kalshi Markets`, `search_predictit`, and Falcon `Candlesticks` (avoid buying vertical illiquid spikes without a fresh primary filing).
3. Express your estimate strictly as a **YES probability `fair_prob` in `(0, 1)`** plus baseline `fair_prob_sd`, scaled by `effective_sd = min(0.45, base_sd * suggested_sd_mult)`.

### Gate 5 — Horizon-Aware Shrunk Kelly & CLOB Depth Walk
1. Call `simulate_clob_fill(token_id=..., side=..., amount_usd=..., fair_prob=token_win_prob, fair_prob_sd=effective_sd, days_to_resolution=...)`.
   - Note: `simulate_clob_fill` expects `fair_prob` for **that specific token** (`fair_prob_yes` if walking the YES token; `1 - fair_prob_yes` if walking the NO token).
2. Require:
   - `kelly_shrunk > 0` after uncertainty shrinkage,
   - Net edge `>= 0.03` (300 bps) after VWAP slippage (or positive riskless bundle edge in `scan_neg_risk_event`),
   - Meaningful horizon carry (`carry >= 1.5 bps/d` or `ann_ret >= 15%`) so capital is not trapped in dead money.

### Gate 6 — Epistemic Ledger Commit & Two-Step `dry_run` Execution
1. Call `record_forecast(...)` with **YES** `market_prob` and **YES** `fair_prob`, `effective_sd`, `thesis_summary`, primary `key_sources`, and explicit observable `invalidation_triggers`.
2. Call `place_order(..., dry_run=True)` first. Verify `POLY_MAX_ORDER_USD`, tick alignment, min share size, and how much crosses immediately vs. rests passively as a GTC maker order.
3. Only when live trading is explicitly authorized and `dry_run=True` succeeded cleanly, call `place_order(..., dry_run=False)` and update `record_forecast(..., order_id=...)` with the returned `order_id`.

---

## 4. Domain Playbooks for All 6 Market Families

### 4.1 Macroeconomics (`family="macro"`)
- **Tool Chain:** `get_market_resolution_rules` $\to$ `search_series` $\to$ `get_series(units="pc1"|"pca"|"lin")` $\to$ `get_fed_releases` $\to$ `search_federal_register` $\to$ Falcon `Kalshi Markets` $\to$ `simulate_clob_fill`.
- **Retail Traps to Exploit/Avoid:**
  - *Headline vs. Core & SA vs. NSA:* Verify whether UMA rules cite `CPIAUCSL` (seasonally adjusted) vs. `CPIAUCNS` (non-seasonally adjusted) or `PCEPILFE` (Core PCE) and exact rounding precision (1 vs. 2 decimal places).
  - *First Print vs. Vintage Revisions:* Most Polymarket GDP/payrolls markets settle on the **initial BLS/BEA release**, whereas historical FRED series reflect revised vintages.
  - *FOMC Cutoffs:* Check whether "Fed cuts by Month X" settles on FOMC statement release time (14:00 ET) vs. effective federal funds rate print.

### 4.2 Long-Dated Crypto (`family="crypto_long"`)
- **Tool Chain:** `get_market_resolution_rules` $\to$ `price_crypto_expiry(symbol="BTCUSDT", strike=..., horizon_hours=..., contract_type="touch"|"expiry")` $\to$ Falcon `Candlesticks` $\to$ `simulate_clob_fill`.
- **Retail Traps to Exploit/Avoid:**
  - *One-Touch vs. Terminal Expiry (The 2x Reflection Trap):* Almost all multi-month Polymarket crypto markets ("Will BTC hit $150,000 before Dec 31?" or "dip to $60,000?") settle YES on **any 1-minute candle high/low touching the barrier** (`contract_type="touch"`). Under zero-drift Brownian reflection, $P_{\text{touch}} \approx 2 \cdot P_{\text{expiry}}$. Never price a `"touch"` contract with `"expiry"`!
  - *Deribit IV Skew & Breeden-Litzenberger Adjustment:* Read `iv_note` from `price_crypto_expiry`: positive upside call skew (`d_iv/dK > 0`) lowers risk-neutral digital probability relative to flat-vol $N(d_2)$, whereas steep downside put skew raises downside digital tail mass.
  - *Reference Venue & Pair:* Confirm whether settlement uses Binance `BTCUSDT` (`USDT` basis risk during stablecoin stress), Coinbase `BTC-USD`, or CoinGecko composite.

### 4.3 Corporate Events & M&A (`family="corporate"`)
- **Tool Chain:** `get_market_resolution_rules` $\to$ `get_company_submissions(ticker_or_cik=..., forms=["8-K","S-4","DEFM14A","10-Q"])` $\to$ `search_sec_filings` $\to$ Firecrawl `firecrawl_scrape` on `sec.gov/Archives/...` $\to$ `calculate_merger_spread_prob` $\to$ `simulate_clob_fill`.
- **Retail Traps to Exploit/Avoid:**
  - *Announced vs. Closed M&A:* An `8-K Item 1.01` (Entry into Material Definitive Agreement) resolves *"Will X agree to acquire Y?"* YES, whereas *"Will X complete acquisition of Y by Date?"* requires `Item 2.01` closing before the market's `endDate` (check S-4 "Outside Date" and HSR Second Request extensions!).
  - *Equity Merger-Arb Benchmark:* Use `calculate_merger_spread_prob(target_price, deal_offer_value, downside_break_price, days_to_close)`. Institutional merger-arb desks price target stock daily; when Polymarket YES deviates >8-10 percentage points from `implied_close_prob` without a timing mismatch between `endDate` and the S-4 Outside Date, investigate the discrepancy.
  - *Leadership & Guidance Catalysts:* Monitor `8-K Item 5.02` for CEO/CFO departures and `Item 1.02` for merger terminations.

### 4.4 Regulatory & Legal (`family="regulatory_legal"`)
- **Tool Chain:** `get_market_resolution_rules` $\to$ `search_court_dockets` $\to$ `get_docket_entries` $\to$ `search_court_opinions` $\to$ Firecrawl `firecrawl_scrape` on RECAP PDFs $\to$ `search_federal_register(include_public_inspection=True)` $\to$ `simulate_clob_fill`.
- **Retail Traps to Exploit/Avoid:**
  - *Procedural Posture Confusion:* Retail routinely conflates a brief **administrative stay** (preserving status quo for days while judges read briefs) with a **preliminary injunction** (likelihood of success on the merits) or a final judgment. Always read the actual docket order via `get_docket_entries` + `firecrawl_scrape`.
  - *Federal Register Public Inspection Desk:* Executive actions, tariffs, and agency final rules appear on `public_inspection` in `search_federal_register` before formal Federal Register publication—check whether rules require *signing*, *filing on public inspection*, or *publication*.

### 4.5 Politics & Elections (`family="politics"`)
- **Tool Chain:** `get_market_resolution_rules` $\to$ `get_candidate_financials` $\to$ `get_bill_status` / `get_nomination_status` $\to$ Firecrawl `firecrawl_scrape` (RCP table) $\to$ `analyze_poll_dropout` $\to$ `get_bookmaker_odds` + `search_predictit` + Falcon `Kalshi Markets` $\to$ `scan_neg_risk_event` $\to$ `simulate_clob_fill`.
- **Retail Traps to Exploit/Avoid:**
  - *Mechanical Poll-Average Dropouts:* Approval and matchup bracket markets resolving to RealClearPolitics averages move predictably when outlier polls age out of the rolling window—compute exact shifts with `analyze_poll_dropout`.
  - *Campaign Solvency Signals:* In primary drop-out markets, `get_candidate_financials` showing burn `>95%` + shrinking cash-on-hand + rising committee debt leads press drop-out rumors by weeks.
  - *Cross-Venue Distortion:* PredictIt caps positions at $850 and charges a 10% fee on profits (biasing longshots upward and heavy favorites downward near 93-95c); weight Falcon `Kalshi Markets` and no-vig `get_bookmaker_odds` more heavily than raw PredictIt prices.

### 4.6 Tech Milestones & Product Launches (`family="tech"`)
- **Tool Chain:** `get_market_resolution_rules` $\to$ Firecrawl `firecrawl_map` & `firecrawl_scrape` (docs, changelogs, model cards, SDK registries, FDA/FCC databases) $\to$ `search_sec_filings` $\to$ `search_x_context` (official lab/founder handles) $\to$ `simulate_clob_fill`.
- **Retail Traps to Exploit/Avoid:**
  - *Waitlist / Research Preview vs. General Availability:* Audit `get_market_resolution_rules` carefully—most AI model/feature markets require **public general availability** (usable without waitlist by any paying/free user), so closed alpha blog posts resolve NO.
  - *Artifact Verification:* Check SDK release tags, API model-list endpoints, and regulatory certifications via `firecrawl_map` / `firecrawl_scrape` rather than rumor accounts on X.

---

## 5. Self-Improvement & Scheduled Sweep Loop

Run this autonomous maintenance loop on scheduled/cron sweeps before scanning for new trades:

1. **Reconcile Active Theses & Resting Orders:**
   - Call `get_active_theses()` and (if live trading is configured) `list_open_orders()`.
   - For every open thesis, check its `invalidation_triggers` against the relevant primary oracle (`get_docket_entries`, `get_company_submissions`, `search_federal_register`, `get_series`, `price_crypto_expiry`).
2. **Pull Invalidated Maker Quotes Immediately:**
   - If an `invalidation_trigger` has fired (or fair value shifted across your resting limit price) and `order_id` is open in `list_open_orders`, run `cancel_order(order_id=..., dry_run=True)` followed by `cancel_order(order_id=..., dry_run=False)` so stale maker liquidity is never picked off.
3. **Score Settled Markets & Recalibrate Uncertainty:**
   - Check open theses whose `endDate` has passed via `get_market_resolution_rules(market_id)`.
   - Once UMA resolution is finalized (`outcome=1` for YES, `outcome=0` for NO), call `resolve_and_score_calibration(market_id=..., outcome=...)`.
   - Inspect `brier_skill_score` (`1 - agent_brier / market_brier`), the decile calibration table (`cal_gap`), and each family's `suggested_sd_mult`:
     - Families with negative skill (`BSS < 0`) automatically return `suggested_sd_mult >= 1.25x`, widening `fair_prob_sd` in `simulate_clob_fill` and shrinking future Kelly sizing until primary-source modeling improves.
     - Families with sustained positive skill (`BSS > 0`, `n >= 3`) earn tighter `suggested_sd_mult`, compounding bankroll into proven domain edges.
