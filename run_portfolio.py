"""
S&P 500 Cross-Sectional Portfolio Backtest  +  RL Agent Visualisation
======================================================================
Steps:
  1. Fetch current S&P 500 constituents from S&P500.csv + WRDS CCM linker
  2. Download point-in-time daily returns (crsp.dsf) for 2015-2024
  3. Construct three strategies: Equal-Weight, VRP Quartile, Momentum
  4. Backtest each vs SPX buy-and-hold  (TC from PIPELINE_TC_BPS env var, default 1 bps)
  5. Load saved RL OOS results from run_pipeline.py
  6. Generate and save two dashboards:
       data/figures/portfolio_dashboard.png
       data/figures/rl_agent_dashboard.png

Usage
-----
    # Run the RL pipeline first (saves OOS results):
    python run_pipeline.py

    # Then run this script:
    python run_portfolio.py [--force-refresh]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.econometrics.sp500_loader import load_sp500_data
from src.portfolio.cross_section import (
    backtest_portfolio,
    build_portfolio_weights,
    compute_cross_sectional_signals,
    compute_tc_drag,
    print_portfolio_report,
)
from src.evaluation.plots import (
    portfolio_dashboard, rl_agent_dashboard,
    institutional_scorecard, attribution_tearsheet,
    monthly_returns_heatmap, rolling_metrics_panel,
    return_decomposition_waterfall,
)
from src.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

DATA_START = cfg.data.start
DATA_END   = cfg.data.end
TC_BPS     = cfg.portfolio.transaction_cost_bps
CACHE_DIR  = ROOT / cfg.paths.cache_dir
FIG_DIR    = ROOT / cfg.paths.figures_dir
RL_PARQUET = CACHE_DIR / f"rl_oos_results{cfg.paths.results_tag}.parquet"
RL_META    = CACHE_DIR / f"rl_oos_results{cfg.paths.results_tag}.json"


def _load_spx_returns() -> pd.Series:
    path = CACHE_DIR / f"prices_{DATA_START.replace('-','')}_{DATA_END.replace('-','')}.parquet"
    if not path.exists():
        log.error("SPX cache missing — run run_pipeline.py first")
        sys.exit(1)
    spx = pd.read_parquet(path).iloc[:, 0]
    spx.index = pd.to_datetime(spx.index)
    return spx.pct_change().dropna().rename("spx")


def _load_vix() -> pd.Series | None:
    path = CACHE_DIR / f"vix_{DATA_START.replace('-','')}_{DATA_END.replace('-','')}.parquet"
    if path.exists():
        s = pd.read_parquet(path).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s
    return None


def _load_rl_results() -> dict | None:
    if not RL_PARQUET.exists() or not RL_META.exists():
        log.warning("RL results not found — run run_pipeline.py first to get RL plots")
        return None
    df = pd.read_parquet(RL_PARQUET)
    df.index = pd.to_datetime(df.index)
    meta = json.loads(RL_META.read_text())
    return {
        "oos_dates":       df.index,
        "oos_rl_returns":  df["rl_returns"].values,
        "oos_spx_returns": df["spx_returns"].values,
        "oos_actions":     np.abs(df["actions"].values),
        "training_rewards": meta["rewards"],
        "training_cvar":    meta["cvar"],
        "training_eta":     meta["eta"],
        "ww_half_width":    meta["ww_half_width"],
        # institutional metrics (present if pipeline was run at current version)
        "dsr":           meta.get("dsr"),
        "profit_factor": meta.get("profit_factor"),
        "wfe":           meta.get("wfe"),
        "mc_pvalue":     meta.get("mc_pvalue"),
        "feature_names": meta.get("feature_names", []),
        # raw series for new plots
        "rl_returns_series":  pd.Series(df["rl_returns"].values,  index=df.index),
        "spx_returns_series": pd.Series(df["spx_returns"].values, index=df.index),
    }


def main(args) -> None:
    t0 = time.perf_counter()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  S&P 500 CROSS-SECTIONAL PORTFOLIO BACKTEST")
    print(f"  Window : {DATA_START} -> {DATA_END}")
    print(f"  TC     : {TC_BPS:.0f} bps round-trip")
    print(f"{'='*64}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("=== STEP 1: Loading data ===")
    spx_ret = _load_spx_returns()
    vix     = _load_vix()

    log.info("Fetching S&P 500 constituent return panel ...")
    returns, universe = load_sp500_data(
        start=DATA_START,
        end=DATA_END,
        use_cache=not args.force_refresh,
        force_refresh=args.force_refresh,
    )

    print(f"\n  SPX benchmark : {len(spx_ret):>5} days")
    print(f"  Constituents  : {returns.shape[1]:>5} stocks (after 50%% NaN filter)")
    print(f"  Price panel   : {returns.shape[0]:>5} trading days")
    missing_pct = returns.isnull().mean().mean() * 100
    print(f"  Missing data  : {missing_pct:.1f}% (point-in-time filtered)")
    print(f"  Universe      : {len(universe)} stocks total from S&P500.csv")
    entry_stats = universe["entry_date"].describe()
    print(f"  Entry dates   : min={entry_stats['min'].date()}  max={entry_stats['max'].date()}")

    # ── 2. Compute cross-sectional signals ────────────────────────────────────
    log.info("=== STEP 2: Computing signals ===")
    signals = compute_cross_sectional_signals(returns, vix=vix)
    log.info("  Signals: %d (date x stock) observations", len(signals))

    # ── 3. Build weights and backtest each strategy ───────────────────────────
    log.info("=== STEP 3: Backtesting strategies ===")
    strategies = {
        "Equal-Weight": "equal_weight",
        "VRP Quartile": "vrp_quartile",
        "Momentum":     "momentum",
    }
    results = {}
    for label, method in strategies.items():
        w = build_portfolio_weights(signals, method=method)
        avg_n  = (w > 0).sum(axis=1).replace(0, np.nan).mean()
        avg_to = w.diff().abs().sum(axis=1).mean()
        log.info("  %-18s  avg_n=%.0f stocks  avg_daily_turnover=%.4f",
                 label, avg_n, avg_to)
        results[label] = backtest_portfolio(w, returns, spx_ret, TC_BPS)

    # ── 4. Print reports ──────────────────────────────────────────────────────
    for label, res in results.items():
        print_portfolio_report(res, method=label)

    # ── 5. Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"  SUMMARY  (TC={TC_BPS:.0f} bps  |  point-in-time constituents  |  survivorship bias noted)")
    print(f"{'='*74}")
    print(f"  {'Strategy':<22}  {'Ann Ret':>9}  {'Sharpe':>8}  {'Max DD':>9}  {'IR':>8}  {'Cum Ret':>9}")
    print(f"  {'-'*70}")
    bmk = results["Equal-Weight"]["metrics"]["benchmark"]
    print(f"  {'SPX Buy & Hold':<22}  {bmk['annual_return']:>8.2%}  {bmk['sharpe']:>8.3f}"
          f"  {bmk['max_drawdown']:>8.2%}  {'':>8}  {bmk['cum_return']:>8.2%}")
    print(f"  {'-'*70}")
    for label, res in results.items():
        p = res["metrics"]["portfolio"]
        r = res["metrics"]["relative"]
        print(f"  {label:<22}  {p['annual_return']:>8.2%}  {p['sharpe']:>8.3f}"
              f"  {p['max_drawdown']:>8.2%}  {r['information_ratio']:>8.3f}  {p['cum_return']:>8.2%}")
    print(f"{'='*74}\n")

    # ── 6. Generate portfolio dashboard plot ──────────────────────────────────
    log.info("=== STEP 4: Generating portfolio dashboard ===")
    port_plot = portfolio_dashboard(results, spx_ret)
    print(f"  Portfolio dashboard saved -> {port_plot}")

    # ── 7. Generate RL agent dashboard (if pipeline results exist) ────────────
    rl_res = _load_rl_results()
    if rl_res is not None:
        log.info("=== STEP 5: Generating RL agent dashboard ===")
        rl_plot = rl_agent_dashboard(rl_res)
        print(f"  RL agent dashboard saved  -> {rl_plot}")

        # ── 8. Institutional scorecard ────────────────────────────────────────
        log.info("=== STEP 6: Generating institutional scorecard ===")
        rl_series  = rl_res["rl_returns_series"]
        spx_series = rl_res["spx_returns_series"]
        # Geometric CAGR instead of arithmetic annualization
        _cum_rl = float((1 + rl_series).prod())
        _n_yrs  = len(rl_series) / 252.0
        ann_ret_rl = float(_cum_rl ** (1 / _n_yrs) - 1) if _n_yrs > 0 else 0.0
        ann_vol_rl = float(rl_series.std() * np.sqrt(252))
        _cum_bm = float((1 + spx_series).prod())
        ann_ret_bm = float(_cum_bm ** (1 / _n_yrs) - 1) if _n_yrs > 0 else 0.0
        cum_rl = (1 + rl_series).cumprod()
        mdd_rl = float(((cum_rl - cum_rl.cummax()) / cum_rl.cummax()).min())
        # Subtract daily RF before computing Sharpe
        _rf_daily = ff_factors["RF"].reindex(rl_series.index).fillna(0.0) if not ff_factors.empty else 0.0
        _rl_excess = rl_series - _rf_daily
        sharpe_rl = float(_rl_excess.mean() / _rl_excess.std() * np.sqrt(252)) if _rl_excess.std() > 0 else 0.0
        active = rl_series - spx_series
        te = float(active.std() * np.sqrt(252))
        # Clamp IR to 0.0 when tracking error is zero
        ir_rl = float((active.mean() * 252) / te) if te > 0 else 0.0

        scorecard_metrics = {
            "sharpe":            sharpe_rl,
            "mdd":               mdd_rl,
            "profit_factor":     rl_res.get("profit_factor") or 0.0,
            "information_ratio": ir_rl,
            "dsr":               rl_res.get("dsr") or 0.0,
            "wfe":               rl_res.get("wfe") or 0.0,
            "mc_pvalue":         rl_res.get("mc_pvalue") or 1.0,
            "ann_return_rl":     ann_ret_rl,
            "ann_return_bm":     ann_ret_bm,
            "ann_vol_rl":        ann_vol_rl,
        }
        sc_plot = institutional_scorecard(scorecard_metrics)
        print(f"  Institutional scorecard   -> {sc_plot}")

        # ── 9. Monthly returns heatmap ────────────────────────────────────────
        log.info("=== STEP 7: Generating monthly returns heatmap ===")
        hm_plot = monthly_returns_heatmap(rl_series, spx_series)
        print(f"  Monthly returns heatmap   -> {hm_plot}")

        # ── 10. Rolling metrics panel ─────────────────────────────────────────
        log.info("=== STEP 8: Generating rolling metrics panel ===")
        rm_plot = rolling_metrics_panel(
            rl_series, spx_series, rl_res["ww_half_width"]
        )
        print(f"  Rolling metrics panel     -> {rm_plot}")

        # ── 11. Attribution tearsheet + waterfall ─────────────────────────────
        # Load attribution data if saved alongside RL results
        attr_parquet = CACHE_DIR / "attribution.parquet"
        if attr_parquet.exists():
            log.info("=== STEP 9: Generating attribution tearsheet ===")
            attr_df = pd.read_parquet(attr_parquet)
            at_plot = attribution_tearsheet(attr_df)
            print(f"  Attribution tearsheet     -> {at_plot}")

            log.info("=== STEP 10: Generating return decomposition waterfall ===")
            tc_drag = compute_tc_drag(TC_BPS, float(rl_res["oos_actions"].mean()))
            wf_plot = return_decomposition_waterfall(
                attr_df, ann_ret_rl, ann_ret_bm, tc_drag
            )
            print(f"  Return waterfall          -> {wf_plot}")
        else:
            log.info("  [SKIP] Attribution plots — no attribution.parquet found")
    else:
        print(f"\n  [SKIP] RL dashboard — run run_pipeline.py first to generate RL results\n")

    print(f"  Total runtime: {time.perf_counter() - t0:.1f}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 portfolio backtest + RL plots")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-fetch all WRDS data, bypassing cache")
    main(parser.parse_args())
