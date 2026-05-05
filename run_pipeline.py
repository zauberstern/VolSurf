#!/usr/bin/env python
"""
Friction-Aware Alpha Generation Framework — End-to-End Pipeline
================================================================
Phases I → IV with comprehensive console reporting.

Usage
-----
    python run_pipeline.py

On the first run this downloads 10 years of WRDS data (~30-60 s) and writes
parquet cache files to data/wrds_cache/.  Subsequent runs load from cache.

Configuration
-------------
Edit the constants below (DATA_START / DATA_END / INSAMPLE_END / N_* flags)
to adjust the data window or DRL complexity.
"""

from __future__ import annotations

import gc
import logging
import math
import os as _os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ── Framework modules ─────────────────────────────────────────────────────────
from src.econometrics.wrds_loader import (
    load_wrds_data, fetch_ff_factors, fetch_vxo, fetch_spx_bid_ask_halfspread,
    fetch_zerocd_local, fetch_opvold_local,
    fetch_cboe_multivol_local, fetch_optionm_rv_dispersion_local,
)
from src.econometrics.data_ingestion import build_state_vector
from src.econometrics.constituent_vsurfd import compute_dispersion_index
from src.econometrics.constituent_options import (
    compute_constituent_option_signals,
)
from src.econometrics.gating import gate_signals
from src.simulation.vecm_engine import simulate_paths, apply_ssvi_bounds
from src.drl_policy.policy import (
    ActorCritic, compute_reward, LagrangianCVaR, training_step, build_optimizer,
    ppo_loss, apply_gradient_step, compute_reward_step, compute_gae,
    compute_drift_weights,
)
from src.evaluation.attribution import (
    attribution_regression, interpret_alpha, whalley_wilmott_width,
    deflated_sharpe_ratio, profit_factor, walk_forward_efficiency,
    mc_permutation_pvalue, information_ratio,
)
from src.evaluation.plots import (
    policy_surface_3d, rl_regime_analysis, feature_sensitivity_heatmap,
    trade_activity_calendar, rolling_regime_metrics,
    streamgraph_allocation, terrain_miner_3d, friction_labyrinth,
    volatility_loom, alpha_sonar_radar, tactical_execution_dashboard,
    tail_risk_topography, constellation_risk, cumulative_return_comparison,
)
from src.config import cfg

# ── Results cache (used by run_portfolio.py for combined plots) ───────────────
# cfg.paths.results_tag disambiguates output files across ablation / WFO runs.
_RESULTS_TAG     = cfg.paths.results_tag
_RL_RESULTS_PATH = ROOT / cfg.paths.cache_dir / f"rl_oos_results{_RESULTS_TAG}.parquet"
_ATTR_PATH       = ROOT / cfg.paths.cache_dir / f"attribution{_RESULTS_TAG}.parquet"
_PATHS_CACHE_DIR = ROOT / cfg.paths.cache_dir

# ── Logging ───────────────────────────────────────────────────────────────────
# Default: use plain basicConfig so tests that import this module are unaffected.
# When run as __main__, reconfigure to a UTF-8 StreamHandler so Unicode
# box-drawing characters don't crash on Windows cp1252 consoles.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Pipeline configuration  (sourced from config.yaml via src.config)
# ═════════════════════════════════════════════════════════════════════════════
DATA_START   = cfg.data.start
DATA_END     = cfg.data.end
INSAMPLE_END = cfg.data.insample_end
OOS_END      = cfg.data.oos_end or cfg.data.end  # null → same as data.end

# Chronological sanity-check — catches misconfigured env vars before any I/O
assert pd.Timestamp(DATA_START) < pd.Timestamp(INSAMPLE_END) < pd.Timestamp(OOS_END), (
    f"Date order violated: DATA_START={DATA_START}, INSAMPLE_END={INSAMPLE_END}, "
    f"OOS_END={OOS_END}.  Must satisfy DATA_START < INSAMPLE_END < OOS_END."
)

# DRL — GPU auto-detected; 20K paths per spec (SKILL.md §2)
N_SIM_PATHS    = cfg.simulation.n_paths
N_SIM_STEPS    = cfg.simulation.n_steps
N_OUTER_EPOCHS = cfg.training.n_epochs
K              = cfg.portfolio.K
GAMMA          = cfg.training.gamma
GAE_LAMBDA     = cfg.training.gae_lambda
N_PPO_UPDATES  = cfg.training.n_ppo_updates
MINI_BATCH     = cfg.training.mini_batch
CVAR_ALPHA     = cfg.cvar.alpha
CVAR_CBAR      = cfg.cvar.c_bar
LR             = cfg.training.lr
TRANSACTION_COST = cfg.training.transaction_cost
RISK_AVERSION  = cfg.evaluation.risk_aversion
PPO_CLIP_EPS   = cfg.training.ppo_clip_eps
C_ENT          = cfg.training.entropy_coef
C_ENT_FLOOR    = cfg.training.entropy_floor   # minimum entropy coefficient

BANNER = "═" * 70
HR     = "─" * 70


def _sec(title: str) -> None:
    log.info("")
    log.info(BANNER)
    log.info("  %s", title)
    log.info(BANNER)


def _hr() -> None:
    log.info(HR)


def _mbb_sample_residuals(
    residuals: np.ndarray,
    n_steps: int,
    block: int = 21,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Moving Block Bootstrap: draw n_steps residuals in consecutive blocks.

    Preserves within-block temporal dependence (ARCH effects, autocorrelation)
    by sampling contiguous blocks of size ``block`` from the residual array.

    Parameters
    ----------
    residuals : (T, n) array of VAR(1) residuals
    n_steps   : total number of steps required
    block     : block length in trading days (default 21 ≈ 1 month)
    rng       : numpy Generator; created fresh if None

    Returns
    -------
    (n_steps, n) array of resampled residuals
    """
    if rng is None:
        rng = np.random.default_rng()
    T_resid = len(residuals)
    n_blocks = int(np.ceil(n_steps / block))
    # Circular wrapping: indices are taken modulo T_resid so every residual is reachable.
    starts = rng.integers(0, T_resid, size=n_blocks)
    # Global pre-centering preserves volatility clustering; per-block centering
    # would destroy ARCH structure by removing within-block variance drift.
    residuals_centered = residuals - residuals.mean(axis=0)
    blocks = []
    for s in starts:
        blk = residuals_centered[[(s + j) % T_resid for j in range(block)]]
        blocks.append(blk)
    return np.concatenate(blocks, axis=0)[:n_steps]


def _bootstrap_paths(
    data: np.ndarray, n_steps: int, n_paths: int, block: int = 21,
) -> np.ndarray:
    """Residual-based VAR(1) bootstrap fallback when VECM simulation fails.

    Fits a VAR(1) model to the historical state matrix, then generates
    synthetic paths by resampling residuals via Moving Block Bootstrap (MBB).
    MBB preserves within-block temporal dependence (ARCH effects, volatility
    clustering) by drawing consecutive blocks of residuals rather than IID
    samples.  The ``block`` parameter controls block size (default 21 trading
    days ≈ 1 month).
    """
    T, n = data.shape

    # Ridge regularisation (λ=1e-4) stabilises OLS on highly collinear state columns.
    Y     = data[1:]                                      # (T-1, n)
    X_aug = np.column_stack([np.ones(T - 1), data[:-1]])  # (T-1, n+1)
    # Z-score standardize feature columns before ridge regression.
    # Without standardization, λ=1e-4 applied to raw X_aug with heterogeneous scales
    # (VIX≈20, yields≈0.04) aggressively shrinks large-magnitude regressors, distorting
    # the companion matrix.  Standardize, solve in the standardized space, then
    # inverse-transform coefficients back to original scale.
    X_features = X_aug[:, 1:]                               # (T-1, n)
    feat_mean = X_features.mean(axis=0)                     # (n,)
    feat_std  = X_features.std(axis=0).clip(min=1e-8)       # (n,)
    X_std = np.column_stack(
        [np.ones(T - 1), (X_features - feat_mean) / feat_std]
    )                                                        # (T-1, n+1)
    ridge_lambda = 1e-4
    XtX  = X_std.T @ X_std
    XtX  += ridge_lambda * np.eye(XtX.shape[0])
    coef_std = np.linalg.solve(XtX, X_std.T @ Y)            # (n+1, n)
    # Inverse-transform: Y ≈ const_std + (X_feat-mean)/std @ A_std
    # → A_orig[i,:] = A_std[i,:] / feat_std[i]
    # → const_orig   = const_std - feat_mean @ A_orig
    A_std = coef_std[1:]                                     # (n, n)
    A     = A_std / feat_std[:, None]                        # (n, n)
    const = coef_std[0] - feat_mean @ A                      # (n,)

    # Schur decomposition deflates only explosive roots, leaving stable
    # cointegrating roots untouched.  Uniform scalar division A/ρ would
    # artificially accelerate every eigenvalue including mean-reversion roots.
    eigs = np.linalg.eigvals(A)
    rho  = float(np.abs(eigs).max())
    if rho > 1.0:
        log.warning(
            "  _bootstrap_paths: VAR(1) spectral radius %.4f > 1 — "
            "deflating unstable roots via Schur decomposition", rho,
        )
        # Real Schur: A = Z T Z'  where T is quasi-upper-triangular.
        # Scale only the diagonal blocks whose eigenvalue magnitude > 1.
        from scipy.linalg import schur, rsf2csf
        T_schur, Z = schur(A, output="real")
        abs_diag = np.abs(np.diag(T_schur))
        scale = np.where(abs_diag > 1.0, abs_diag + 1e-4, 1.0)
        T_schur = T_schur / scale[:, None]
        A = Z @ T_schur @ Z.T

    residuals = Y - (const + data[:-1] @ A)  # (T-1, n) zero-mean by OLS construction

    paths = np.zeros((n_paths, n_steps, n))

    for i in range(n_paths):
        resid_seq = _mbb_sample_residuals(residuals, n_steps, block=block)
        # Initialise each path at the last in-sample state, anchoring the
        # bootstrap distribution at the IS/OOS boundary.
        state     = data[-1].copy()
        for t in range(n_steps):
            state = const + state @ A + resid_seq[t]
            paths[i, t] = state
    return paths


# ═════════════════════════════════════════════════════════════════════════════
def _cache_audit() -> None:
    """Print a cache-hit / cache-miss banner so the user knows what will be fetched."""
    from src.econometrics.wrds_loader import _CACHE_DIR as _WD
    import src.econometrics.constituent_vsurfd as _cvd
    import src.econometrics.sp500_loader as _spl
    import glob

    log.info("")
    log.info(BANNER)
    log.info("  CACHE STATUS AUDIT")
    log.info(BANNER)

    def _hit(path) -> str:
        p = Path(path)
        if p.exists():
            mb = p.stat().st_size / 1e6
            return f"HIT    {mb:5.1f} MB  ({p.name})"
        return "MISS   — will fetch/compute"

    # WRDS option panel (may be multiple files matching pattern)
    op_files = sorted(Path(_WD).glob("option_panel_*.parquet"))
    log.info("  %-32s : %s",
             "WRDS option panel (vsurfd)",
             f"HIT {len(op_files)} file(s)" if op_files
             else "MISS — will re-fetch from WRDS")

    log.info("  %-32s : %s", "WRDS prices (SPX)",
             _hit(Path(_WD) / f"prices_{DATA_START.replace('-','')}_{DATA_END.replace('-','')}.parquet"))
    log.info("  %-32s : %s", "Zero-curve rates",
             _hit(Path(_WD) / f"zerocd_rates_{DATA_START.replace('-','')}_{DATA_END.replace('-','')}.parquet"))
    log.info("  %-32s : %s", "SPX log put-call ratio",
             _hit(Path(_WD) / f"spx_log_pcr_{DATA_START.replace('-','')}_{DATA_END.replace('-','')}.parquet"))
    log.info("  %-32s : %s", "IV dispersion index",
             _hit(_cvd._CACHE_FILE))
    log.info("  %-32s : %s", "S&P 500 PIT membership",
             _hit(_spl._PIT_CACHE))

    # Sim paths cache
    sim_files = sorted(Path(_WD).glob("sim_paths_*.npy"))
    log.info("  %-32s : %s",
             "VECM sim paths",
             f"HIT {len(sim_files)} file(s)" if sim_files
             else "MISS — will simulate (slow)")

    log.info("  %-32s : %s", "Policy weights",
             _hit(Path(_WD) / "policy_weights.pt"))
    log.info("")


def main() -> None:
    global K  # K may be updated inside main if fewer than cfg.portfolio.K stocks are available
    # Reconfigure root logger to a UTF-8 StreamHandler so Unicode characters
    # (═, →, ·, μ, etc.) don't crash on Windows cp1252 consoles.
    import io as _io
    _utf8_stream = _io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", line_buffering=True
    )
    _handler = logging.StreamHandler(_utf8_stream)
    _handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    _root = logging.getLogger()
    _root.handlers.clear()
    _root.addHandler(_handler)

    t_pipeline = time.perf_counter()
    _cache_audit()

    # ══════════════════════════════════════════════════════════════════════════
    _sec("PHASE I · Data Ingestion  (10-year WRDS panel)")
    # ══════════════════════════════════════════════════════════════════════════

    t_phase_I = time.perf_counter()
    log.info("Window : %s → %s", DATA_START, DATA_END)
    log.info("Sources: CRSP (SPX) · OptionMetrics (vsurfd) · CBOE free CSV (VIX) · FRED DGS10 (10Y)")
    log.info("         Ken French (Mkt-RF, RF) · OptionMetrics opprcd (ATM spreads)")
    log.info("")

    data          = load_wrds_data(start=DATA_START, end=DATA_END)
    prices        = data["prices"]
    option_panel  = data["option_panel"]
    vix           = data["vix"]
    treasury_10y  = data["treasury_10y"]

    log.info("")
    log.info("  Raw data summary")
    _hr()
    log.info("  %-18s : %d rows   %.0f – %.0f",
             "SPX prices", len(prices), prices.min(), prices.max())
    log.info("  %-18s : %d rows   iv_30 μ=%.4f   skew μ=%.4f",
             "Option panel", len(option_panel),
             option_panel["iv_30"].mean(), option_panel["skew_25d"].mean())
    log.info("  %-18s : %d rows   μ=%.4f   max=%.4f",
             "VIX", len(vix), vix.mean(), vix.max())
    log.info("  %-18s : %d rows   μ=%.4f   max=%.4f",
             "Treasury 10Y", len(treasury_10y),
             treasury_10y.mean(), treasury_10y.max())

    # ── Supplementary: Ken French factors + VXO + ATM spreads ─────────────────
    log.info("")
    log.info("  Fetching supplementary data (free sources) …")
    ff_factors  = fetch_ff_factors(DATA_START, DATA_END)
    vxo         = fetch_vxo(DATA_START, DATA_END)   # pre-2003 VIX substitute

    if not ff_factors.empty:
        log.info("  %-18s : %d rows   Mkt-RF μ=%+.4f   RF μ=%.5f",
                 "FF factors", len(ff_factors),
                 ff_factors["Mkt-RF"].mean(), ff_factors["RF"].mean())
    else:
        log.info("  FF factors      : unavailable — SPX proxy will be used")

    if vxo is not None:
        log.info("  %-18s : %d rows (pre-2003 VIX substitute, discontinued Sep 2021)",
                 "VXO", len(vxo))

    # ── New local CSV data sources ────────────────────────────────────────────
    log.info("")
    log.info("  Loading local OptionMetrics flat-file exports …")
    zerocd    = fetch_zerocd_local(DATA_START, DATA_END)
    log_pcr   = fetch_opvold_local(DATA_START, DATA_END)

    if not zerocd.empty:
        log.info("  %-18s : %d rows   short_rate μ=%.4f   slope μ=%.4f",
                 "Zero curve (30d/365d)", len(zerocd),
                 zerocd["short_rate"].mean(), zerocd["curve_slope"].mean())
    else:
        log.info("  Zero curve      : not available (SPX_zerocd.csv missing)")

    if len(log_pcr) > 0:
        log.info("  %-18s : %d rows   log(P/C) μ=%.4f",
                 "Log PCR", len(log_pcr), log_pcr.mean())
    else:
        log.info("  Log PCR         : not available (SPX_opvold.csv missing)")

    log.info("  Loading constituent IV dispersion index …")
    dispersion_df  = compute_dispersion_index()
    iv_dispersion  = dispersion_df["iv_dispersion"] if not dispersion_df.empty else None

    if iv_dispersion is not None:
        log.info("  %-18s : %d rows   iv_dispersion μ=%.4f   (2003–2021, fwd-filled OOS)",
                 "IV dispersion", len(iv_dispersion), float(iv_dispersion.mean()))
    else:
        log.info("  IV dispersion   : not available (constituent vsurfd CSV missing)")

    # ── Per-constituent option signals (2003-2024, slim OptionMetrics export) ─
    log.info("  Loading per-constituent option signals (2003-2024) …")
    _consopt_csv = getattr(cfg.paths, "constituent_options_csv", None)
    _consopt_df  = compute_constituent_option_signals(csv_path=_consopt_csv or None)
    if not _consopt_df.empty:
        # Update iv_dispersion for full history: raw ATM IV dispersion supersedes
        # the OptionMetrics vsurfd interpolation wherever observations overlap.
        _iv_disp_raw = _consopt_df["iv_dispersion_raw"].dropna()
        if iv_dispersion is not None:
            iv_dispersion = _iv_disp_raw.combine_first(iv_dispersion)
        else:
            iv_dispersion = _iv_disp_raw

        # Guard: constituent signals are only useful if they cover the IS period.
        # The all-constituents CSV may only span 2023-2024; if it doesn't have
        # enough IS observations the features would be constant (ffill from a
        # single OOS date) and would collapse the state tensor via dropna().
        _is_start_ts = pd.Timestamp(DATA_START)
        _is_end_ts   = pd.Timestamp(INSAMPLE_END)
        # Fix parquet µs/ns timestamp mismatch: index may be stored as µs but
        # read back as ns, yielding 1970 dates.  Reinterpret if first date < 2000.
        if len(_consopt_df) > 0 and _consopt_df.index[0].year < 2000:
            log.warning(
                "  constituent_opts: detected wrong timestamps (year=%d) — "
                "reinterpreting index as µs (pandas/pyarrow precision mismatch).",
                _consopt_df.index[0].year,
            )
            _consopt_df.index = pd.to_datetime(
                _consopt_df.index.astype("int64"), unit="us"
            )
        _is_coverage = _consopt_df.loc[_is_start_ts:_is_end_ts]
        if len(_is_coverage) >= 10:
            # Slim file covers full history: use directly.
            _constituent_iv_full   = _consopt_df["mean_constituent_iv"]
            _constituent_skew_full = _consopt_df["mean_constituent_skew"]
        else:
            log.warning(
                "  constituent_opts: only %d IS obs in %s–%s — "
                "dropping constituent_iv/skew features (CSV does not cover IS period).",
                len(_is_coverage), DATA_START, INSAMPLE_END,
            )
            _constituent_iv_full   = None
            _constituent_skew_full = None
        log.info(
            "  %-18s : %d obs   mean_iv μ=%.4f  "
            "iv_disp μ=%.4f  skew μ=%.4f  n_stocks μ=%.0f",
            "Constituent opts",
            len(_consopt_df),
            float(_consopt_df["mean_constituent_iv"].mean()),
            float(_consopt_df["iv_dispersion_raw"].mean()),
            float(_consopt_df["mean_constituent_skew"].mean()),
            float(_consopt_df["n_stocks"].mean()),
        )
    else:
        _constituent_iv_full   = None
        _constituent_skew_full = None
        log.info("  Constituent opts  : not available (CSV missing or not found)")

    # ── CBOE cross-index vol spreads (VXN, VXD) ──────────────────────────────
    log.info("  Loading CBOE multi-vol indices (VXN, VXD) …")
    _cboe_multivol = fetch_cboe_multivol_local(DATA_START, DATA_END)
    vxn_ser = _cboe_multivol["vxn"] if not _cboe_multivol.empty else None
    vxd_ser = _cboe_multivol["vxd"] if not _cboe_multivol.empty else None
    if not _cboe_multivol.empty:
        log.info("  %-18s : %d rows   vxn μ=%.4f   vxd μ=%.4f",
                 "CBOE VXN/VXD", len(_cboe_multivol),
                 float(_cboe_multivol["vxn"].mean()), float(_cboe_multivol["vxd"].mean()))
    else:
        log.info("  CBOE multi-vol  : not available (cboe_vix_vox_2003_2024.csv missing)")

    # ── OptionMetrics constituent realized-vol dispersion ────────────────────
    if cfg.signals.include_rv_dispersion:
        log.info("  Computing OptionMetrics constituent RV dispersion …")
        rv_disp_ser = fetch_optionm_rv_dispersion_local(DATA_START, DATA_END)
        if len(rv_disp_ser) > 0:
            log.info("  %-18s : %d rows   rv_dispersion μ=%.4f",
                     "RV dispersion", len(rv_disp_ser), float(rv_disp_ser.mean()))
        else:
            log.info("  RV dispersion   : not available (OptionMetrics HV CSV missing)")
    else:
        rv_disp_ser = None

    # ── State vector ──────────────────────────────────────────────────────────
    log.info("")
    log.info("  Building state tensor s_t …")
    state, fwd_returns = build_state_vector(
        option_panel=option_panel,
        underlying_prices=prices,
        vix=vix,
        treasury_10y=treasury_10y,
        vxo=vxo,
        log_pcr=log_pcr if len(log_pcr) > 0 else None,
        zerocd=zerocd if not zerocd.empty else None,
        iv_dispersion=iv_dispersion,
        vxn=vxn_ser if cfg.signals.include_cross_vol_spread else None,
        vxd=vxd_ser if cfg.signals.include_cross_vol_spread else None,
        rv_dispersion=rv_disp_ser,
        constituent_iv=_constituent_iv_full,
        constituent_skew=_constituent_skew_full,
    )
    # Ensure plain float64 throughout — pandas 2.x may return nullable FloatingArray
    # which breaks numpy matmul, lstsq, and torch tensor construction.
    state       = state.astype(np.float64)
    fwd_returns = fwd_returns.astype(np.float64)
    state_dim = state.shape[1]

    log.info("  Shape  : %s   (%d features)", state.shape, state_dim)
    log.info("  Dates  : %s → %s", state.index[0].date(), state.index[-1].date())
    log.info("  Cols   : %s", list(state.columns))
    log.info("")
    log.info("  Descriptive statistics (state tensor)")
    _hr()
    log.info("  %-18s  %+10s  %10s  %10s  %10s",
             "Feature", "Mean", "Std", "Min", "Max")
    _hr()
    for col in state.columns:
        s = state[col]
        log.info("  %-18s  %+10.5f  %10.5f  %10.5f  %10.5f",
                 col, s.mean(), s.std(), s.min(), s.max())

    # ── Train / OOS split ─────────────────────────────────────────────────────
    # Enforce 91-day purge window between IS end and OOS start.
    _purge_days = getattr(cfg.data, 'purge_days', 91)
    _oos_start  = pd.Timestamp(INSAMPLE_END) + pd.Timedelta(days=_purge_days)
    is_mask   = state.index <= pd.Timestamp(INSAMPLE_END)
    oos_mask  = (state.index >= _oos_start) & (state.index <= pd.Timestamp(OOS_END))
    state_is  = state[is_mask]
    state_oos = state[oos_mask]
    fwd_is    = fwd_returns[is_mask]
    fwd_oos   = fwd_returns[oos_mask]

    log.info("")
    log.info("  Train/test split : %d in-sample  |  %d OOS", len(state_is), len(state_oos))
    log.info("  IS  : %s → %s", state_is.index[0].date(), state_is.index[-1].date())
    log.info("  OOS : %s → %s  (purge=%dd)", state_oos.index[0].date(), state_oos.index[-1].date(), _purge_days)

    # ── Constituent return panel: top-K by IS data density ────────────────────
    log.info("")
    log.info("  Loading S&P 500 constituent return panel (selecting top-%d by IS density) ...", K)
    try:
        from src.econometrics.sp500_loader import load_sp500_data
        _const_ret_raw, _ = load_sp500_data(start=DATA_START, end=DATA_END, use_cache=True)
        # Select top-K permnos by fraction of non-missing IS observations.
        # Point-in-time survivorship filter is applied inside load_sp500_data.
        _is_density = _const_ret_raw[
            _const_ret_raw.index <= pd.Timestamp(INSAMPLE_END)
        ].notna().mean()
        top_k_permnos = _is_density.nlargest(K).index.tolist()
        constituent_returns_k = _const_ret_raw[top_k_permnos].fillna(0.0)
        K = len(top_k_permnos)          # update if fewer than K stocks available
        log.info("  Selected K=%d constituents  (min IS density = %.1f%%)",
                 K, float(_is_density[top_k_permnos].min()) * 100)
        log.info("  Sample PERMNOs: %s%s", top_k_permnos[:5], " ..." if K > 5 else "")

        # Build permno → ticker lookup for display in plots
        try:
            _prc_csv = Path("data/sp500_prices_2003_2024_all_constituents.csv")
            _snap = pd.read_csv(_prc_csv, usecols=["PERMNO", "TICKER"])
            permno_to_ticker = _snap.groupby("PERMNO")["TICKER"].last().to_dict()
        except Exception:
            permno_to_ticker = {}
        asset_labels_k = [
            permno_to_ticker.get(int(p) if str(p).isdigit() else p, str(p))
            for p in top_k_permnos
        ]
    except Exception as _exc:
        log.warning("  Constituent data unavailable (%s) -- falling back to SPX-only (K=1)", _exc)
        K = 1
        top_k_permnos = ["SPX"]
        constituent_returns_k = fwd_returns.rename(0).to_frame()
        permno_to_ticker = {}
        asset_labels_k = ["SPX"]

    # ── Per-stock bid-ask spread + ADV: market-realistic transaction costs ────
    # Loads quoted half-spread and 20-day rolling dollar ADV from the CRSP panel.
    # Gap 1: VIX-regime-conditional Glosten-Harris effective/quoted ratio
    #   (Petersen & Fialkowski 1994 JFE: 0.55 calm / 0.50 normal / 0.45 crisis)
    # Gap 3: log-linear VIX→spread β fitted from IS data for simulation scaling.
    # Impact is normalised by ADV following Almgren-Chriss: η_k = a_imp*(NAV/ADV_k)^β
    log.info("")
    log.info("  Loading per-stock bid-ask spread and ADV from CRSP panel …")
    try:
        from src.econometrics.sp500_loader import load_spread_adv_from_local_csv
        _hs_wide, _adv_wide = load_spread_adv_from_local_csv(
            start=DATA_START, end=DATA_END, use_cache=True
        )
        _is_end = pd.Timestamp(INSAMPLE_END)

        # ── Gap 1: per-date VIX-conditional Glosten-Harris factor ────────────
        _vix_aligned = vix.reindex(_hs_wide.index).ffill().fillna(0.20)
        _eff_series = pd.Series(
            np.where(
                _vix_aligned <= cfg.training.spread_eff_vix_low,
                cfg.training.spread_eff_factor_low,
                np.where(
                    _vix_aligned <= cfg.training.spread_eff_vix_high,
                    cfg.training.spread_eff_factor_mid,
                    cfg.training.spread_eff_factor_high,
                ),
            ),
            index=_hs_wide.index,
        )
        # Effective spread = quoted × VIX-regime GH factor (per date × per stock)
        _hs_eff_wide = _hs_wide.multiply(_eff_series, axis=0)

        # 1. Market-wide median effective half-spread → new state signal
        _mean_spread_ts = (
            _hs_eff_wide
            .median(axis=1)
            .reindex(state.index)
            .ffill()
            .fillna(TRANSACTION_COST)
        )
        state = state.copy()
        state["mean_spread"] = _mean_spread_ts
        state_is  = state[is_mask]
        state_oos = state[oos_mask]  # oos_mask already includes purge window
        state_dim = state.shape[1]
        log.info(
            "  Added mean_spread signal: IS median=%.2f bps  OOS median=%.2f bps",
            float(state_is["mean_spread"].median()) * 1e4,
            float(state_oos["mean_spread"].median()) * 1e4,
        )

        # 2. Per-stock IS-median effective half-spread for reward computation
        _k_in_hs  = [p for p in top_k_permnos if p in _hs_eff_wide.columns]
        _hs_is    = _hs_eff_wide.loc[_hs_eff_wide.index <= _is_end, _k_in_hs]
        _hs_med   = _hs_is.median(axis=0)  # (K_avail,) per permno
        a_tc_k = np.array(
            [float(_hs_med.get(p, TRANSACTION_COST)) for p in top_k_permnos],
            dtype=np.float32,
        )

        # 3. Per-stock IS-median dollar ADV → Almgren-Chriss impact coefficient
        _k_in_adv = [p for p in top_k_permnos if p in _adv_wide.columns]
        _adv_is   = _adv_wide.loc[_adv_wide.index <= _is_end, _k_in_adv]
        _adv_med  = _adv_is.median(axis=0)
        _nav      = float(cfg.training.portfolio_nav)
        _beta     = cfg.training.impact_beta
        _a_imp    = cfg.training.impact_coef
        impact_coef_k = np.array(
            [
                float(_a_imp * (_nav / float(_adv_med.get(p, 1e8))) ** _beta)
                for p in top_k_permnos
            ],
            dtype=np.float32,
        )

        # ── Gap 3: OLS fit of log-linear VIX→spread β on IS data ────────────
        # log(median_spread_t) = log(s0) + β * VIX_t  (Chordia et al. 2001 JF)
        _mean_eff_is = (
            _hs_eff_wide.loc[_hs_eff_wide.index <= _is_end]
            .median(axis=1)
            .dropna()
        )
        _vix_is = vix.reindex(_mean_eff_is.index).ffill().dropna()
        _cidx   = _mean_eff_is.index.intersection(_vix_is.index)
        _log_s  = np.log(_mean_eff_is.loc[_cidx].clip(lower=1e-8).values)
        _vix_v  = _vix_is.loc[_cidx].values
        _X      = np.column_stack([np.ones_like(_vix_v), _vix_v])
        _ols    = np.linalg.lstsq(_X, _log_s, rcond=None)[0]
        spread_beta    = float(_ols[1])          # elasticity per decimal VIX unit
        vix_IS_median  = float(_vix_is.median())
        log.info(
            "  VIX→spread OLS:  β=%.3f  VIX_IS_med=%.3f  "
            "(VIX+0.10 → spread ×%.2f)",
            spread_beta, vix_IS_median, float(np.exp(spread_beta * 0.10)),
        )

        log.info(
            "  Per-stock spread (bps): median=%.2f  min=%.2f  max=%.2f",
            float(np.median(a_tc_k)) * 1e4,
            float(a_tc_k.min()) * 1e4,
            float(a_tc_k.max()) * 1e4,
        )
        log.info(
            "  Per-stock impact coef:  median=%.2e  min=%.2e  max=%.2e  (NAV=$%.0fM)",
            float(np.median(impact_coef_k)),
            float(impact_coef_k.min()),
            float(impact_coef_k.max()),
            _nav / 1e6,
        )
    except Exception as _se:
        log.warning(
            "  Spread/ADV unavailable (%s) — using scalar fallback (a_tc=%.0e)", _se, TRANSACTION_COST
        )
        a_tc_k        = np.full(K, TRANSACTION_COST, dtype=np.float32)
        impact_coef_k = np.full(K, cfg.training.impact_coef, dtype=np.float32)
        spread_beta   = 1.5   # Chordia et al. (2001) default (per decimal VIX unit)
        vix_IS_median = float(vix[vix.index <= pd.Timestamp(INSAMPLE_END)].median())

    log.info("")
    log.info("  Phase I · Data Ingestion complete  (%.1f s)", time.perf_counter() - t_phase_I)

    # VIX column index in finalised state tensor (used for simulation scaling)
    VIX_IDX = list(state.columns).index("vix")  # invariant: always present

    # ── Forward return stats ──────────────────────────────────────────────────
    ann_ret_is = fwd_is.mean() * 252
    ann_vol_is = fwd_is.std() * np.sqrt(252)
    sharpe_is  = ann_ret_is / ann_vol_is if ann_vol_is > 0 else 0.0
    log.info("")
    log.info("  SPX forward return stats (in-sample):")
    _hr()
    log.info("  Annual return  = %+.4f  (%.2f%%)", ann_ret_is, ann_ret_is * 100)
    log.info("  Annual vol     = %.4f   (%.2f%%)", ann_vol_is, ann_vol_is * 100)
    log.info("  Sharpe (raw)   = %.4f", sharpe_is)
    log.info("  Skewness       = %.4f", float(fwd_is.skew()))
    log.info("  Excess Kurt    = %.4f", float(fwd_is.kurt()))

    # ══════════════════════════════════════════════════════════════════════════
    _sec("PHASE I · Econometric Signal Gating  (Benjamini-Yekutieli FDR)")
    # ══════════════════════════════════════════════════════════════════════════
    t_phase_gating = time.perf_counter()

    # Use actual SPX ATM bid-ask half-spreads from OptionMetrics opprcd tables.
    # Falls back to 5 bps constant if table access fails.
    db_for_spread = None
    try:
        import wrds, os as _os
        db_for_spread = wrds.Connection(wrds_username=_os.environ["WRDS_USERNAME"])
        raw_spreads = fetch_spx_bid_ask_halfspread(db_for_spread, DATA_START, INSAMPLE_END)
        db_for_spread.close()
        median_spread = float(raw_spreads.median())
        log.info("  ATM half-spread : %.5f (%.1f bps median, from OptionMetrics opprcd)",
                 median_spread, median_spread * 1e4)
    except Exception as exc:
        # Literature-based fallback: 25 bps (Muravyev & Pearson RFS 2020;
        # Broadie, Chernov & Johannes JF 2009: SPX ATM half-spreads 20–50 bps)
        median_spread = cfg.gating.half_spread_fallback
        log.warning("  ATM half-spread : %.5f (%.1f bps — literature fallback; opprcd err: %s)",
                    median_spread, median_spread * 1e4, str(exc)[:60])

    half_spreads = pd.Series({col: median_spread for col in state_is.columns})
    passed = gate_signals(state_is, fwd_is, half_spreads, alpha=cfg.gating.alpha)

    log.info("  Null hypothesis : H₀: μᵢ ≤ c̄ᵢ  (return ≤ c̄ half-spread)")
    log.info("  HAC lag window  : L = floor(0.75·T^(1/3))")
    log.info("  FDR control     : Benjamini-Yekutieli step-up @ α=5%%")
    log.info("  Signals pass    : %d / %d", passed.sum(), len(passed))
    _hr()
    log.info("  %-18s  %s", "Signal", "Decision")
    _hr()
    for sig, ok in passed.items():
        mark = "✓  REJECT H₀  (signal survives gating)" if ok else "✗  FAIL TO REJECT (filtered out)"
        log.info("  %-18s  %s", sig, mark)

    active_signals = list(passed[passed].index)
    log.info("")
    log.info("  Active signals forwarded to DRL: %s", active_signals if active_signals else "[none — all retained per spec]")
    log.info("  Phase I · Gating complete  (%.1f s)", time.perf_counter() - t_phase_gating)

    # ══════════════════════════════════════════════════════════════════════════
    _sec("PHASE II · VECM Simulation Engine")
    # ══════════════════════════════════════════════════════════════════════════

    t_phase_II = time.perf_counter()
    log.info("  Fitting VECM on in-sample state (%d obs, %d dims) …", len(state_is), state_dim)
    log.info("  Reinsel-Ahn finite-sample correction applied to Johansen trace statistics")
    log.info("  Generating %d paths × %d steps …", N_SIM_PATHS, N_SIM_STEPS)

    # ── Paths cache: keyed by shape + state_dim + start date ──────────────────
    _paths_key = (
        f"sim_paths_{N_SIM_PATHS}_{N_SIM_STEPS}"
        f"_D{state_dim}_T{len(state_is)}_{state_is.index[0].date()}"
    )
    _paths_cache = _PATHS_CACHE_DIR / f"{_paths_key}.npy"

    vecm_ok = True
    if _paths_cache.exists():
        paths = np.load(str(_paths_cache))
        log.info("  Paths cache hit  : %s  → shape %s", _paths_cache.name, paths.shape)
    else:
        t0 = time.perf_counter()
        try:
            paths = simulate_paths(
                state_is, n_steps=N_SIM_STEPS, n_paths=N_SIM_PATHS,
                alpha_sig=cfg.simulation.johansen_alpha,
            )
            sim_elapsed = time.perf_counter() - t0
            log.info("  Simulation OK  : %s  (%.1fs)", str(paths.shape), sim_elapsed)
        except Exception as exc:
            vecm_ok = False
            log.warning("  VECM failed (%s) — falling back to block bootstrap", exc)
            paths = _bootstrap_paths(
                state_is.values, N_SIM_STEPS, N_SIM_PATHS,
                block=cfg.simulation.bootstrap_block,
            )
            log.info("  Bootstrap paths shape: %s", str(paths.shape))
        _PATHS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(_paths_cache), paths)
        log.info("  Paths cached     → %s", _paths_cache.name)

    # Apply SSVI arbitrage-free bounds to IV columns (Gatheral-Jacquier SVI framework).
    # Enforces Lee (2004) non-negativity, upper bound, and calendar spread no-arbitrage.
    # Applied unconditionally — idempotent on freshly simulated paths, corrects any
    # residual violations in cached paths that pre-date this constraint.
    paths = apply_ssvi_bounds(paths, list(state_is.columns))
    log.info("  SSVI bounds applied: iv_30/iv_91 constrained to [0.01, 3.0], calendar spread enforced")

    sim_means = paths.reshape(-1, paths.shape[2]).mean(axis=0)
    hist_means = state_is.values.mean(axis=0)
    log.info("")
    log.info("  Simulation fidelity check (simulated vs historical means):")
    _hr()
    log.info("  %-18s  %+12s  %+12s  %+12s", "Feature", "Sim mean", "Hist mean", "Δ")
    _hr()
    for i, col in enumerate(state_is.columns):
        log.info("  %-18s  %+12.5f  %+12.5f  %+12.5f",
                 col, sim_means[i], hist_means[i], sim_means[i] - hist_means[i])

    # ── Return models: K independent OLS regressions (IS only, no leakage) ───
    # Pseudo-inverse (lstsq) handles multicollinearity between IV/RV features.
    # Coefficients are fit EXCLUSIVELY on IS data and locked for OOS inference.
    X_is = np.column_stack([np.ones(len(state_is)), state_is.values])   # (T_is, D+1)
    const_is_np = constituent_returns_k.reindex(state_is.index).fillna(0.0).values  # (T_is, K)
    beta_K, _, _, _ = np.linalg.lstsq(X_is, const_is_np, rcond=None)   # (D+1, K)
    resid_K     = const_is_np - X_is @ beta_K                           # (T_is, K)
    resid_std_K = resid_K.std(axis=0)                                   # (K,)
    r2_K = np.array([
        1.0 - np.var(resid_K[:, k]) / (np.var(const_is_np[:, k]) + 1e-12)
        for k in range(K)
    ])

    log.info("")
    log.info("  Return prediction models (K=%d OLS regressions — IS only):", K)
    _hr()
    log.info("  %-6s  %-14s  %8s  %10s", "#", "PERMNO", "R²", "Resid σ")
    _hr()
    for k, perm in enumerate(top_k_permnos):
        log.info("  %-6d  %-14s  %8.4f  %10.6f", k, str(perm), r2_K[k], resid_std_K[k])
    log.info("  Mean R² = %.4f   Mean resid σ = %.6f", r2_K.mean(), resid_std_K.mean())

    # Build predicted return tensor pred_r_K (N, T, K) for Phase III training
    flat_states = paths.reshape(-1, state_dim)                    # (N*T, D)
    aug_flat    = np.column_stack([np.ones(len(flat_states)), flat_states])
    pred_flat   = aug_flat @ beta_K                               # (N*T, K)
    noise_flat  = np.random.randn(*pred_flat.shape) * resid_std_K # (N*T, K)
    pred_r_K    = (pred_flat + noise_flat).reshape(
        N_SIM_PATHS, N_SIM_STEPS, K
    ).astype(np.float32)                                          # (N, T, K)

    log.info("  Phase II · VECM Simulation complete  (%.1f s)", time.perf_counter() - t_phase_II)

    # ══════════════════════════════════════════════════════════════════════════
    _sec("PHASE III · GRU-64 Actor-Critic  +  Lagrangian CVaR")
    # ══════════════════════════════════════════════════════════════════════════

    action_dim = K  # K-dimensional softmax portfolio weights
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=cfg.model.hidden_size,
    ).to(device)
    optimizer = build_optimizer(model, lr=LR)
    cvar_ctrl = LagrangianCVaR(
        alpha=CVAR_ALPHA, c_bar=CVAR_CBAR, gamma=cfg.cvar.dual_step,
        rho=getattr(cfg.cvar, 'rho', 1.0),
    )

    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Device          : %s", str(device).upper())
    log.info("  Architecture    : GRU(input=%d, hidden=%d) -> Softmax Actor (K=%d) + Critic",
             state_dim, cfg.model.hidden_size, K)
    log.info("  Parameters      : %d  (actor_log_std included)", n_params)
    log.info("  Optimizer       : Adam(lr=%.0e)", LR)
    log.info("  CVaR α          : %.2f   c̄=%.2f   dual step γ=1e-3", CVAR_ALPHA, CVAR_CBAR)
    log.info("  Grad clip       : always clip to max_norm=5  (abort if norm>100)")
    log.info(
        "  Friction        : per-stock spread  a_tc median=%.2e  a_imp median=%.2e  β=%.1f",
        float(np.median(a_tc_k)), float(np.median(impact_coef_k)), cfg.training.impact_beta,
    )
    log.info(
        "  VIX→spread      : β=%.3f  VIX_IS_med=%.3f  (simulation scaling active)",
        spread_beta, vix_IS_median,
    )
    log.info("  PPO             : clip_eps=%.2f  entropy_coef=%.3f", PPO_CLIP_EPS, C_ENT)
    log.info("  GAE             : γ=%.2f  λ=%.2f", GAMMA, GAE_LAMBDA)
    log.info("")
    log.info("  Training : %d outer epochs × %d PPO inner updates × %d mini-batches",
             N_OUTER_EPOCHS, N_PPO_UPDATES, N_SIM_PATHS // MINI_BATCH)
    log.info("             = %d total gradient steps",
             N_OUTER_EPOCHS * N_PPO_UPDATES * (N_SIM_PATHS // MINI_BATCH))
    _hr()
    log.info("  %-6s  %-12s  %-12s  %-12s  %-10s  %-8s  %-12s  %s",
             "Epoch", "Mean Reward", "Best Reward", "PPO Loss", "CVaR", "Dual η", "Time(s)", "ETA")
    _hr()
    t_phase_III = time.perf_counter()

    epoch_rewards: list[float] = []
    epoch_cvars:   list[float] = []
    epoch_etas:    list[float] = []
    n_aborted = 0
    best_reward   = float("-inf")

    # ── Resume from final checkpoint if present (skip full training) ──────────
    _final_ckpt = _PATHS_CACHE_DIR / f"policy_weights_ep{N_OUTER_EPOCHS:03d}.pt"
    if _final_ckpt.exists():
        log.info("  Final checkpoint found: %s — skipping training and loading weights.",
                 _final_ckpt.name)
        model.load_state_dict(torch.load(_final_ckpt, map_location=device, weights_only=True))
        epoch_rewards = [0.0] * N_OUTER_EPOCHS
        epoch_cvars   = [0.0] * N_OUTER_EPOCHS
        epoch_etas    = [0.0] * N_OUTER_EPOCHS
        log.info("  Phase III · DRL Training SKIPPED — using saved weights  (%.1f s)",
                 time.perf_counter() - t_phase_III)
    else:
        # ── Transfer tensors to device ─────────────────────────────────────────
        paths_tensor  = torch.tensor(paths, dtype=torch.float32).to(device)    # (N, T, D)
        pred_r_tensor = torch.tensor(pred_r_K, dtype=torch.float32).to(device) # (N, T, K)
        # Per-stock cost tensors: shape (K,) — broadcast over (N, T, K) in reward
        a_tc_t        = torch.tensor(a_tc_k,        dtype=torch.float32).to(device)  # (K,)
        impact_coef_t = torch.tensor(impact_coef_k, dtype=torch.float32).to(device)  # (K,)
        # Gap 3: log-linear VIX→spread scaling scalars for time-varying simulation costs
        spread_beta_t   = torch.tensor(spread_beta,   dtype=torch.float32, device=device)
        vix_median_t    = torch.tensor(vix_IS_median, dtype=torch.float32, device=device)

        _ep_times: list[float] = []    # rolling window for ETA

        for outer in range(1, N_OUTER_EPOCHS + 1):
            t_ep = time.perf_counter()

            # ═══════════════════════════════════════════════════════════════════
            # ROLLOUT PHASE — fully vectorised over all N paths and T steps
            # ═══════════════════════════════════════════════════════════════════
            model.eval()
            with torch.no_grad():
                gru_out, _ = model.gru(paths_tensor)                  # (N, T, 64)
                mu_all     = model.actor_head(gru_out)                 # (N, T, K)
                val_all    = model.critic_head(gru_out).squeeze(-1)    # (N, T)

                # Sample: stick-breaking bijection from N(mu, std) on first K-1 dims
                std_r    = model.actor_log_std[:-1].clamp(-4.0, 0.0).exp()     # (K-1,)
                dist_r   = torch.distributions.Normal(mu_all[..., :-1], std_r) # (N,T,K-1)
                x_all    = dist_r.rsample()                                     # (N, T, K-1)
                z_all    = torch.sigmoid(x_all)                                 # (N, T, K-1)

                # Build weights via stick-breaking
                w_all      = torch.empty(N_SIM_PATHS, N_SIM_STEPS, K, device=device)
                remaining  = torch.ones(N_SIM_PATHS, N_SIM_STEPS, device=device)
                log_jac_all = torch.zeros(N_SIM_PATHS, N_SIM_STEPS, device=device)
                for _k in range(K - 1):
                    w_all[..., _k] = remaining * z_all[..., _k]
                    log_jac_all = (
                        log_jac_all
                        + remaining.clamp(min=1e-8).log()
                        + (z_all[..., _k] * (1.0 - z_all[..., _k])).clamp(min=1e-8).log()
                    )
                    remaining = remaining * (1.0 - z_all[..., _k])
                w_all[..., -1] = remaining

                # Log-prob on simplex (for PPO denominator)
                lp_all = dist_r.log_prob(x_all).sum(dim=-1) - log_jac_all  # (N, T)

                # Per-step reward with L1 turnover cost.
                # Prepend zero weights at t=0 (no prior position at episode start).
                # Drift-adjust the previous target weights to account for passive
                # market movement between rebalancing dates before computing turnover.
                w_prev_raw = torch.cat([
                    torch.zeros(N_SIM_PATHS, 1, K, device=device),
                    w_all[:, :-1, :]
                ], dim=1)                                             # (N, T, K)
                r_prev_raw = torch.cat([
                    torch.zeros(N_SIM_PATHS, 1, K, device=device),
                    pred_r_tensor[:, :-1, :]
                ], dim=1)                                             # (N, T, K)
                w_prev_drift = compute_drift_weights(w_prev_raw, r_prev_raw)  # (N, T, K)
                delta_w_k  = (w_all - w_prev_drift).abs()              # (N, T, K) per-stock
                port_ret   = (w_all * pred_r_tensor).sum(dim=-1)       # (N, T)
                # Gap 3: VIX-conditional spread scaling — paths_tensor[:,:,VIX_IDX] gives
                # simulated VIX at each step; exp(β*(VIX_t - VIX_IS_med)) scales a_tc_k.
                _vix_sim      = paths_tensor[:, :, VIX_IDX]            # (N, T)
                _spread_scale = torch.exp(spread_beta_t * (_vix_sim - vix_median_t))  # (N, T)
                _a_tc_nt      = a_tc_t[None, None, :] * _spread_scale[:, :, None]     # (N, T, K)
                # Beta-neutral reward: subtract equal-weight market return so the agent
                # optimises excess return over the index rather than raw return.
                # r_mkt = (N,T) equal-weight mean of K constituent returns.
                r_mkt       = pred_r_tensor.mean(dim=-1)                # (N, T)
                _beta_pen   = getattr(cfg.training, 'beta_penalty', 1.0)
                reward_all = (
                    port_ret
                    - _beta_pen * r_mkt                                 # beta-neutral: subtract market
                    - (_a_tc_nt * delta_w_k).sum(dim=-1)
                    - (impact_coef_t * delta_w_k.pow(1.0 + cfg.training.impact_beta)).sum(dim=-1)
                )                                                       # (N, T)

                # GAE advantage estimates
                advantages  = compute_gae(reward_all, val_all, GAMMA, GAE_LAMBDA)
                returns_tgt = (advantages + val_all).detach()        # critic targets
                # Per-timestep normalisation (across N paths at each step t);
                # global batch normalisation would leak future volatility shocks
                # backward through time.
                adv_mean = advantages.mean(dim=0, keepdim=True)  # (1, T)
                adv_std  = advantages.std(dim=0, keepdim=True).clamp(min=1e-8)  # (1, T)
                adv_norm = (advantages - adv_mean) / adv_std      # (N, T)
                adv_flat = adv_norm.reshape(-1)

                # CVaR on per-path sum-reward (tail risk per episode)
                path_rewards = reward_all.sum(dim=1)                 # (N,)
                v_epoch = torch.quantile(-path_rewards, 1.0 - CVAR_ALPHA).detach()

            # ═══════════════════════════════════════════════════════════════════
            # PPO UPDATE PHASE — mini-batch over paths (full T-step sequence each)
            # ═══════════════════════════════════════════════════════════════════
            model.train()
            total_ppo_loss = 0.0
            n_updates = 0

            for _ in range(N_PPO_UPDATES):
                perm = torch.randperm(N_SIM_PATHS, device=device)
                for mb_start in range(0, N_SIM_PATHS, MINI_BATCH):
                    idx = perm[mb_start: mb_start + MINI_BATCH]

                    mb_paths    = paths_tensor[idx]         # (B, T, D)
                    mb_x_old    = x_all[idx].detach()       # (B, T, K-1) pre-sigmoid samples
                    mb_w_old    = w_all[idx].detach()       # (B, T, K) old weights
                    mb_old_lp   = lp_all[idx].detach()      # (B, T)
                    mb_log_jac  = log_jac_all[idx].detach() # (B, T) Jacobian from rollout
                    mb_adv    = adv_norm[idx]              # (B, T)
                    mb_ret    = returns_tgt[idx]           # (B, T) critic targets
                    mb_pred_r = pred_r_tensor[idx]         # (B, T, K)

                    # Forward pass through full T-step GRU sequence
                    gru_mb, _ = model.gru(mb_paths)       # (B, T, 64)
                    mu_mb     = model.actor_head(gru_mb)  # (B, T, K)
                    val_mb    = model.critic_head(gru_mb).squeeze(-1)  # (B, T)

                    std_new  = model.actor_log_std[:-1].clamp(-4.0, 0.0).exp()   # (K-1,)
                    dist_new = torch.distributions.Normal(mu_mb[..., :-1], std_new)
                    # Re-use old pre-sigmoid samples; Jacobian cancels in PPO ratio
                    # but must be included for absolute log-prob consistency.
                    new_lp = (
                        dist_new.log_prob(mb_x_old).sum(dim=-1) - mb_log_jac
                    )                                          # (B, T)

                    # Flatten B×T for PPO ratio computation
                    # Clamp log-ratio before exp() to prevent IS ratio overflow.
                    log_ratio = (new_lp.reshape(-1) - mb_old_lp.reshape(-1)).clamp(-20.0, 2.0)
                    ratio  = log_ratio.exp()
                    adv_f  = mb_adv.reshape(-1)
                    surr1  = ratio * adv_f
                    surr2  = ratio.clamp(1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * adv_f
                    loss_policy = -torch.min(surr1, surr2).mean()

                    # Critic MSE
                    loss_critic = torch.nn.functional.mse_loss(
                        val_mb.reshape(-1), mb_ret.reshape(-1)
                    )

                    # CVaR on mini-batch path rewards (sum over T per path)
                    w_mb_prev = torch.cat([
                        torch.zeros(len(idx), 1, K, device=device),
                        mb_w_old[:, :-1, :]
                    ], dim=1)
                    dw_mb_k = (mb_w_old - w_mb_prev).abs()                # (B, T, K)
                    pr_mb   = (mb_w_old * mb_pred_r).sum(dim=-1)           # (B, T)
                    # Gap 3: VIX-conditional scaling on mini-batch paths
                    _vix_mb      = mb_paths[:, :, VIX_IDX]                 # (B, T)
                    _scale_mb    = torch.exp(spread_beta_t * (_vix_mb - vix_median_t))
                    _a_tc_mb     = a_tc_t[None, None, :] * _scale_mb[:, :, None]  # (B, T, K)
                    rew_mb  = (
                        pr_mb
                        - (_a_tc_mb * dw_mb_k).sum(dim=-1)
                        - (impact_coef_t * dw_mb_k.pow(1.0 + cfg.training.impact_beta)).sum(dim=-1)
                    )
                    loss_cvar = cvar_ctrl.penalized_loss_with_threshold(
                        -rew_mb.sum(dim=1), v_epoch
                    )

                    # Gaussian entropy bonus (stable; correct up to additive constant)
                    entropy_bonus = dist_new.entropy().sum(dim=-1).mean()

                    # Per-epoch entropy decay floored at C_ENT_FLOOR to prevent exploration collapse.
                    c_ent_t = max(C_ENT * (cfg.training.entropy_decay ** (outer - 1)), C_ENT_FLOOR)
                    total_loss = loss_policy + loss_critic + loss_cvar - c_ent_t * entropy_bonus

                    optimizer.zero_grad()
                    total_loss.backward()
                    applied = apply_gradient_step(
                        model, optimizer,
                        grad_norm_abort=cfg.training.grad_norm_abort,
                        grad_clip_max=cfg.training.grad_clip_max,
                    )
                    if applied:
                        total_ppo_loss += float(total_loss.detach())
                        n_updates += 1
                    else:
                        n_aborted += 1

            # ── CVaR dual update (full-batch estimate once per epoch) ─────────────
            # Robbins-Monro dual step decay: γ_t = γ₀/√epoch satisfies two-timescale
            # conditions (Borkar 1997), preventing limit cycles in the dual iterate.
            cvar_val = float(cvar_ctrl.cvar_loss(-path_rewards).detach())
            cvar_ctrl.gamma = cfg.cvar.dual_step / math.sqrt(outer)
            cvar_ctrl.dual_update(cvar_val)

            mean_reward = float(path_rewards.mean())
            avg_loss    = total_ppo_loss / max(n_updates, 1)
            epoch_rewards.append(mean_reward)
            epoch_cvars.append(cvar_val)
            epoch_etas.append(float(cvar_ctrl.eta))
            best_reward = max(best_reward, mean_reward)

            ep_time = time.perf_counter() - t_ep
            _ep_times.append(ep_time)
            if len(_ep_times) > 5:
                _ep_times.pop(0)
            avg_ep = sum(_ep_times) / len(_ep_times)
            remaining = (N_OUTER_EPOCHS - outer) * avg_ep
            eta_str = f"{int(remaining // 60)}m{int(remaining % 60):02d}s"
            pct = outer / N_OUTER_EPOCHS * 100
            log.info(
                "  %3d/%-3d [%5.1f%%]  r=%+.6f  best=%+.6f  loss=%+.6f  CVaR=%.4f  η=%.4f  %6.1fs  ETA≈%s",
                outer, N_OUTER_EPOCHS, pct,
                mean_reward, best_reward, avg_loss,
                cvar_val, cvar_ctrl.eta, ep_time, eta_str,
            )

            if outer % cfg.training.checkpoint_every == 0:
                ckpt = _PATHS_CACHE_DIR / f"policy_weights_ep{outer:03d}.pt"
                torch.save(model.cpu().state_dict(), ckpt)
                model.to(device)
                log.info("  [CHECKPOINT ep%03d] weights saved → %s", outer, ckpt.name)

            # Explicit cleanup: without del, both old and new epoch tensors (~5.7 GB
            # each) coexist at the start of the next epoch during reallocation,
            # causing a segfault under Windows virtual-memory pressure around ep24.
            del (gru_out, mu_all, val_all, x_all, z_all, w_all, lp_all,
                 log_jac_all, remaining, w_prev_raw, r_prev_raw, w_prev_drift,
                 delta_w_k, port_ret, _vix_sim, _spread_scale, _a_tc_nt,
                 reward_all, advantages, returns_tgt, adv_flat, adv_norm,
                 path_rewards, v_epoch)
            gc.collect()

            log.info("")
        log.info("  Training summary:")
        _hr()
        log.info("  Reward  : %+.6f → %+.6f   Δ=%+.6f",
                 epoch_rewards[0], epoch_rewards[-1], epoch_rewards[-1] - epoch_rewards[0])
        log.info("  Best    : %+.6f (epoch %d)",
                 best_reward, epoch_rewards.index(best_reward) + 1)
        log.info("  CVaR    : %.6f → %.6f", epoch_cvars[0], epoch_cvars[-1])
        log.info("  Dual η  : %.4f", cvar_ctrl.eta)
        log.info("  Total updates : %d gradient steps  (%d aborted — grad norm > 100)",
                 N_OUTER_EPOCHS * N_PPO_UPDATES * (N_SIM_PATHS // MINI_BATCH), n_aborted)
        if len(epoch_rewards) >= 10:
            tail10 = epoch_rewards[-10:]
            log.info("  Trend (last 10 epochs): %s", " → ".join(f"{r:+.4f}" for r in tail10))
        log.info("  Phase III · DRL Training complete  (%.1f s)", time.perf_counter() - t_phase_III)

    # ══════════════════════════════════════════════════════════════════════════
    _sec("PHASE IV · Out-of-Sample Evaluation  (2022–2024)")
    # ══════════════════════════════════════════════════════════════════════════
    t_phase_IV = time.perf_counter()

    model.eval()
    with torch.no_grad():
        oos_tensor  = torch.tensor(
            state_oos.values[np.newaxis, :, :], dtype=torch.float32,
        ).to(device)                                         # (1, T_oos, D)
        gru_oos, _  = model.gru(oos_tensor)                 # (1, T_oos, 64)
        mu_oos         = model.actor_head(gru_oos[0])       # (T_oos, K)
        oos_weights_k  = model.mean_action(mu_oos).cpu().numpy()  # (T_oos, K) deterministic
        oos_hidden_np  = gru_oos[0].cpu().numpy()            # (T_oos, 64)

    # OOS portfolio returns: weighted sum over K constituent returns
    const_oos_np = constituent_returns_k.reindex(state_oos.index).fillna(0.0).values  # (T_oos, K)
    oos_portfolio_returns = pd.Series(
        (oos_weights_k * const_oos_np).sum(axis=1),
        index=fwd_oos.index,
        name="rl_returns",
    )
    # Scalar L1 per-step turnover proxy (for W-W analysis and plots expecting 1-D)
    oos_weights = np.abs(
        oos_weights_k - np.vstack([np.zeros((1, K)), oos_weights_k[:-1]])
    ).sum(axis=1)                                            # (T_oos,)

    # ── Performance metrics ───────────────────────────────────────────────────
    pr = oos_portfolio_returns
    bm = fwd_oos.rename("benchmark")

    ann_ret_rl  = pr.mean() * 252
    ann_vol_rl  = pr.std() * np.sqrt(252)
    sharpe_rl   = ann_ret_rl / ann_vol_rl if ann_vol_rl > 0 else 0.0

    ann_ret_bm  = bm.mean() * 252
    ann_vol_bm  = bm.std() * np.sqrt(252)
    sharpe_bm   = ann_ret_bm / ann_vol_bm if ann_vol_bm > 0 else 0.0

    cum_rl     = (1 + pr).cumprod()
    cum_bm     = (1 + bm).cumprod()
    mdd_rl     = float(((cum_rl - cum_rl.cummax()) / cum_rl.cummax()).min())
    mdd_bm     = float(((cum_bm - cum_bm.cummax()) / cum_bm.cummax()).min())
    final_rl   = float(cum_rl.iloc[-1] - 1)
    final_bm   = float(cum_bm.iloc[-1] - 1)

    hit_rate   = float((np.sign(oos_portfolio_returns.values) == np.sign(fwd_oos.values)).mean())
    turnover   = float(oos_weights.mean())
    mean_wt    = float(oos_weights_k.max(axis=1).mean())
    ir_rl      = information_ratio(pr, bm)

    log.info("  OOS period : %s → %s  (%d trading days)",
             state_oos.index[0].date(), state_oos.index[-1].date(), len(state_oos))
    log.info("")
    log.info("  Performance vs Benchmark (passive SPX buy-and-hold)")
    _hr()
    log.info("  %-32s  %-14s  %-14s", "Metric", "RL Policy", "SPX Buy-Hold")
    _hr()
    log.info("  %-32s  %+12.4f%%  %+12.4f%%", "Annualised Return",
             ann_ret_rl * 100, ann_ret_bm * 100)
    log.info("  %-32s  %12.4f%%  %12.4f%%",   "Annualised Volatility",
             ann_vol_rl * 100, ann_vol_bm * 100)
    log.info("  %-32s  %+12.4f    %+12.4f",   "Sharpe Ratio",
             sharpe_rl, sharpe_bm)
    log.info("  %-32s  %12.4f%%  %12.4f%%",   "Max Drawdown",
             mdd_rl * 100, mdd_bm * 100)
    log.info("  %-32s  %+12.4f%%  %+12.4f%%", "Cumulative Return",
             final_rl * 100, final_bm * 100)
    log.info("  %-32s  %12.4f%%  %s", "Direction Hit Rate",
             hit_rate * 100, "     N/A")
    log.info("  %-32s  %12.6f    %s", "Mean |Action Weight|",
             mean_wt, "     N/A")
    log.info("  %-32s  %12.6f    %s", "Mean Daily Turnover",
             turnover, "     ~0")
    ir_str = f"{ir_rl:+.4f}" if np.isfinite(ir_rl) else "   N/A"
    log.info("  %-32s  %12s    %s", "Information Ratio (vs SPX)",
             ir_str, "     N/A")

    # ── Whalley-Wilmott no-trade zone ─────────────────────────────────────────
    log.info("")
    log.info("  Whalley-Wilmott Asymptotic No-Trade Zone:")
    _hr()
    gamma_atm = cfg.evaluation.ww_gamma_atm
    S_ref     = float(prices.iloc[-1])
    # risk_aversion from config is Arrow-Pratt relative risk aversion.
    # W-W uses absolute RA: a_abs = a_rel / S, so the formula becomes
    # w = (3cΓ²S² / (2×a_rel))^(1/3).
    a_abs     = RISK_AVERSION / S_ref                               # absolute RA
    ww_width  = whalley_wilmott_width(c=TRANSACTION_COST, gamma=gamma_atm,
                                       S=S_ref, a=a_abs)
    efficiency = "EFFICIENT (action > W-W threshold)" \
                 if mean_wt > ww_width else "WITHIN NO-TRADE ZONE"
    log.info("  c=%.4f  Γ=%.4f  S=%.0f  a_rel=%.1f  a_abs=%.6f", TRANSACTION_COST, gamma_atm, S_ref, RISK_AVERSION, a_abs)
    log.info("  W-W no-trade half-width w = (3cΓ²S / 2a)^(1/3) = %.6f", ww_width)
    log.info("  Mean |action|             = %.6f", mean_wt)
    log.info("  Policy turnover status    : %s", efficiency)

    # ── Attribution regression ────────────────────────────────────────────────
    _sec("PHASE IV · Attribution Regression  (Newey-West HAC, L=1)")
    log.info("  Model: r_RL = α + β_m·r_M + β_c·r_carry + β_v·r_vol + β_VRP·r_VRP")
    log.info("  HAC   : Newey-West  L = 1  (non-overlapping daily returns; no multi-period MA overlap)")
    log.info("")

    # Market factor: Ken French Mkt-RF (gold standard) → excess market return
    # Risk-free rate: Ken French RF (daily 1-month T-bill)
    if not ff_factors.empty:
        r_market = ff_factors["Mkt-RF"].reindex(fwd_oos.index).rename("r_M")
        rf_daily  = ff_factors["RF"].reindex(fwd_oos.index).fillna(0.0)
        rl_excess = (oos_portfolio_returns - rf_daily).rename("rl_returns")
        log.info("  Market factor   : Ken French Mkt-RF (CRSP value-weighted excess return)")
    else:
        r_market  = fwd_oos.rename("r_M")
        rl_excess = oos_portfolio_returns.rename("rl_returns")
        log.info("  Market factor   : raw SPX return (FF factors unavailable)")

    r_carry  = treasury_10y.reindex(fwd_oos.index).diff().rename("r_carry")
    r_vol    = vix.reindex(fwd_oos.index).diff().rename("r_v")
    r_vrp    = state_oos["vrp"].rename("r_VRP")

    attr = attribution_regression(
        rl_returns=rl_excess,
        market_returns=r_market,
        carry_returns=r_carry,
        vol_returns=r_vol,
        vrp_returns=r_vrp,
    )

    _hr()
    log.info("  %-22s  %+10s  %10s  %+10s  %10s  %4s",
             "Factor", "Coef", "HAC-SE", "t-stat", "p-value", "Sig")
    _hr()
    for name, row in attr.iterrows():
        sig = ("***" if row["p_value"] < 0.01
               else "**"  if row["p_value"] < 0.05
               else "*"   if row["p_value"] < 0.10
               else "")
        log.info("  %-22s  %+10.6f  %10.6f  %+10.4f  %10.4f  %4s",
                 name, row["coef"], row["hac_se"], row["t_stat"], row["p_value"], sig)
    log.info("")
    log.info("  Significance: *** p<1%%   ** p<5%%   * p<10%%")

    # Save attribution DataFrame for run_portfolio.py tearsheet + waterfall
    attr.to_parquet(_ATTR_PATH)

    outcome = interpret_alpha(attr)
    alpha_row = attr.loc["alpha (beta_0)"]
    log.info("")
    log.info("  ┌─────────────────────────────────────────────────────┐")
    log.info("  │  Alpha intercept (β₀): %+.6f   t=%+.3f   p=%.4f  │",
             alpha_row["coef"], alpha_row["t_stat"], alpha_row["p_value"])
    log.info("  │  ══ INTERPRETATION : %-30s  │", outcome)
    log.info("  └─────────────────────────────────────────────────────┘")

    # ── Save OOS results + hidden states for combined plots ──────────────────
    import json
    _RL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Build extended parquet: returns + actions + K weights + state features + 64 hidden dims
    hidden_cols  = {f"h_{i}": oos_hidden_np[:, i] for i in range(oos_hidden_np.shape[1])}
    state_cols   = {f"s_{c}": state_oos[c].values for c in state_oos.columns}
    weight_cols  = {f"w_{k}": oos_weights_k[:, k] for k in range(K)}
    rl_save = pd.DataFrame(
        {"rl_returns": oos_portfolio_returns.values,
         "spx_returns": fwd_oos.values,
         "actions": oos_weights,
         **weight_cols, **state_cols, **hidden_cols},
        index=fwd_oos.index,
    )
    rl_save.to_parquet(_RL_RESULTS_PATH)

    # Save model weights + inference bundle (everything needed to reload and run)
    _WEIGHTS_PATH = _RL_RESULTS_PATH.with_name("policy_weights.pt")
    torch.save(model.cpu().state_dict(), _WEIGHTS_PATH)

    # Self-contained inference bundle: weights + architecture metadata.
    # Load with:
    #   bundle = torch.load("policy_inference.pt", weights_only=False)
    #   model  = ActorCritic(bundle["state_dim"], bundle["action_dim"])
    #   model.load_state_dict(bundle["state_dict"])
    _INFERENCE_PATH = _RL_RESULTS_PATH.with_name("policy_inference.pt")
    torch.save(
        {
            "state_dict":    model.cpu().state_dict(),
            "state_dim":     state_dim,
            "action_dim":    action_dim,
            "K":             K,
            "top_k_permnos": top_k_permnos,
            "feature_names": list(state_oos.columns),
            "insample_end":  INSAMPLE_END,
            "data_start":    DATA_START,
            "data_end":      DATA_END,
            "cvar_alpha":    CVAR_ALPHA,
            "transaction_cost": TRANSACTION_COST,
            "risk_aversion": RISK_AVERSION,
        },
        _INFERENCE_PATH,
    )

    training_meta = {
        "rewards": epoch_rewards,
        "cvar":    epoch_cvars,
        "eta":     [float(e) for e in epoch_etas],
        "ww_half_width": float(ww_width),
        "feature_names": list(state_oos.columns),
        "state_dim":     state_dim,
        "action_dim":    action_dim,
    }
    log.info("  RL results saved    → %s", _RL_RESULTS_PATH)
    log.info("  Model weights       → %s", _WEIGHTS_PATH)
    log.info("  Inference bundle    → %s  (reload: ActorCritic(state_dim=%d, action_dim=%d))",
             _INFERENCE_PATH.name, state_dim, action_dim)

    # ── Institutional validation metrics (benchmarking.md) ───────────────────
    _sec("PHASE IV · Institutional Validation Metrics")
    from scipy.stats import skew as _skew, kurtosis as _kurt

    sr_obs      = sharpe_rl
    skew_obs    = float(_skew(pr.values))
    exkurt_obs  = float(_kurt(pr.values))
    dsr         = deflated_sharpe_ratio(
                      sr=sr_obs, T=len(pr),
                      skewness=skew_obs, excess_kurtosis=exkurt_obs,
                      n_trials=1,
                  )
    log.info("  Deflated Sharpe Ratio     : %.4f  (>0.95 = significant at 5%%)", dsr)

    pf = profit_factor(pr)
    log.info("  Profit Factor             : %.4f  (target: 1.75\u20133.0)", pf)

    # WFE: OOS / IS where both are policy annual returns.
    # IS policy return is approximated from the mean training reward × 252.
    # (fwd_is is the SPX index return — not the policy's IS performance.)
    is_ann_ret_policy = float(np.mean(epoch_rewards)) * 252
    wfe = walk_forward_efficiency(is_ann_return=is_ann_ret_policy, oos_ann_return=ann_ret_rl)
    log.info("  Walk-Forward Efficiency   : %.4f  (>0.50 = robust OOS transfer)", wfe)

    mc_p = mc_permutation_pvalue(
        pr,
        n_trials=cfg.evaluation.mc_n_trials,
        seed=cfg.evaluation.mc_seed,
        block_size=cfg.evaluation.mc_block_size,
    )
    log.info("  MC Permutation p-value    : %.4f  (<0.05 = signal not random)", mc_p)

    # Persist all metrics to json (overwrite with complete record)
    training_meta["dsr"]          = float(dsr)
    training_meta["profit_factor"] = float(pf) if np.isfinite(pf) else None
    training_meta["wfe"]          = float(wfe) if np.isfinite(wfe) else None
    training_meta["mc_pvalue"]    = float(mc_p)
    _RL_RESULTS_PATH.with_suffix(".json").write_text(json.dumps(training_meta))

    # ── 3D policy transparency plot ───────────────────────────────────────────
    log.info("  Generating 3D policy transparency surface ...")
    VIX_IDX = list(state_oos.columns).index("vix") if "vix" in state_oos.columns else 7
    VRP_IDX = list(state_oos.columns).index("vrp") if "vrp" in state_oos.columns else 6
    n_grid = cfg.evaluation.policy_surface_grid
    oos_vals = state_oos.values
    vix_q = np.quantile(oos_vals[:, VIX_IDX], [0.05, 0.95])
    vrp_q = np.quantile(oos_vals[:, VRP_IDX], [0.05, 0.95])
    vix_axis = np.linspace(vix_q[0], vix_q[1], n_grid)
    vrp_axis = np.linspace(vrp_q[0], vrp_q[1], n_grid)
    VIX_G, VRP_G = np.meshgrid(vix_axis, vrp_axis)
    mean_state   = oos_vals.mean(axis=0)
    action_surf  = np.zeros((n_grid, n_grid), dtype=np.float32)
    model_cpu = model  # already on CPU after torch.save above
    model_cpu.eval()
    with torch.no_grad():
        for gi in range(n_grid):
            for gj in range(n_grid):
                s = mean_state.copy()
                s[VIX_IDX] = VIX_G[gi, gj]
                s[VRP_IDX] = VRP_G[gi, gj]
                t = torch.FloatTensor(s[None, None, :])
                gru_s, _ = model_cpu.gru(t)
                mu_s = model_cpu.actor_head(gru_s[0])
                # Peak concentration: max softmax weight (portfolio conviction)
                action_surf[gi, gj] = float(model_cpu.mean_action(mu_s)[0].max())
    surf_path = policy_surface_3d(
        vix_grid=VIX_G,
        vrp_grid=VRP_G,
        action_surface=action_surf,
        oos_states=oos_vals,
        hidden_states=oos_hidden_np,
        oos_actions=oos_weights,
        oos_returns=fwd_oos.values,
        feature_names=list(state_oos.columns),
    )
    log.info("  3D policy surface → %s", surf_path)

    # ── Regime analysis ───────────────────────────────────────────────────────
    log.info("  Generating VIX-regime analysis …")
    vix_oos_series = vix.reindex(fwd_oos.index).ffill()
    rl_res_dict = {
        "oos_dates":      fwd_oos.index,
        "oos_rl_returns": oos_portfolio_returns.values,
        "oos_spx_returns": fwd_oos.values,
        "oos_actions":    oos_weights,          # (T_oos,) scalar turnover per step
        "oos_weights_k":  oos_weights_k,        # (T_oos, K) full K-weight matrix
        "training_rewards": epoch_rewards,
        "training_cvar":    epoch_cvars,
        "training_eta":     epoch_etas,
        "ww_half_width":    ww_width,
    }
    regime_path = rl_regime_analysis(rl_res_dict, vix_oos_series)
    log.info("  Regime analysis   → %s", regime_path)

    # ── Feature sensitivity heatmap ───────────────────────────────────────────
    log.info("  Generating feature sensitivity heatmap (numerical gradients) …")
    model_cpu = model if next(model.parameters()).device.type == "cpu" else model.cpu()
    feat_path = feature_sensitivity_heatmap(
        model=model_cpu,
        oos_states=state_oos.values.astype(np.float32),
        oos_dates=fwd_oos.index,
        feature_names=list(state_oos.columns),
    )
    log.info("  Feature heatmap   → %s", feat_path)

    # ── Trade activity calendar ───────────────────────────────────────────────
    log.info("  Generating trade activity calendar …")
    cal_path = trade_activity_calendar(oos_portfolio_returns, fwd_oos.index)
    log.info("  Activity calendar → %s", cal_path)

    # ── Rolling regime metrics ────────────────────────────────────────────────
    log.info("  Generating rolling regime metrics panel …")
    roll_path = rolling_regime_metrics(
        rl_returns=oos_portfolio_returns,
        spx_returns=fwd_oos,
        vix_oos=vix_oos_series,
        oos_dates=fwd_oos.index,
    )
    log.info("  Rolling metrics   → %s", roll_path)

    # ── Streamgraph allocation ────────────────────────────────────────────────
    log.info("  Generating streamgraph allocation plot …")
    sg_path = streamgraph_allocation(
        oos_dates=fwd_oos.index,
        oos_actions=oos_weights_k,   # (T_oos, K) full K-asset weight matrix
        vix_series=vix_oos_series,
        asset_labels=asset_labels_k,
    )
    log.info("  Streamgraph       → %s", sg_path)

    # ── 3D terrain miner ─────────────────────────────────────────────────────
    log.info("  Generating 3D terrain miner plot …")
    tm_path = terrain_miner_3d(
        oos_states=state_oos.values.astype(np.float32),
        oos_actions=oos_weights,
        oos_dates=fwd_oos.index,
        feature_names=list(state_oos.columns),
    )
    log.info("  Terrain miner     → %s", tm_path)

    # ── Friction labyrinth ────────────────────────────────────────────────────
    log.info("  Generating friction labyrinth plot …")
    fl_path = friction_labyrinth(
        epoch_rewards=rl_res_dict.get("training_rewards", []),
        ww_half_width=ww_width,
    )
    log.info("  Friction labyrinth→ %s", fl_path)

    # ── Volatility loom ───────────────────────────────────────────────────────
    log.info("  Generating volatility loom tapestry …")
    vl_path = volatility_loom(
        oos_states=state_oos.values.astype(np.float32),
        oos_actions=oos_weights,
        feature_names=list(state_oos.columns),
        oos_dates=fwd_oos.index,
    )
    log.info("  Volatility loom   → %s", vl_path)

    # ── Alpha sonar radar ─────────────────────────────────────────────────────
    log.info("  Generating alpha sonar radar …")
    attr_df_saved = pd.read_parquet(_ATTR_PATH) if _ATTR_PATH.exists() else pd.DataFrame()
    radar_path = alpha_sonar_radar(
        attr_df=attr_df_saved,
        oos_dates=fwd_oos.index,
    )
    log.info("  Alpha sonar       → %s", radar_path)

    # ── Tactical execution dashboard ──────────────────────────────────────────
    log.info("  Generating tactical execution dashboard …")
    oos_rewards_arr = np.asarray(rl_res_dict.get("oos_rl_returns", oos_portfolio_returns.values))
    tac_path = tactical_execution_dashboard(
        oos_states=state_oos.values.astype(np.float32),
        oos_actions=oos_weights,
        oos_rewards=oos_rewards_arr,
        vix_series=vix_oos_series,
        ww_half_width=ww_width,
        oos_dates=fwd_oos.index,
        feature_names=list(state_oos.columns),
    )
    log.info("  Tactical dashboard→ %s", tac_path)

    # ── Tail-risk topography ──────────────────────────────────────────────────
    log.info("  Generating tail-risk topography …")
    _iv_disp_oos = (
        iv_dispersion.reindex(fwd_oos.index)
        if iv_dispersion is not None else None
    )
    trt_path = tail_risk_topography(
        oos_returns=oos_portfolio_returns,
        oos_actions=oos_weights,
        iv_dispersion=_iv_disp_oos,
        cvar_limit=-float(TRANSACTION_COST) * 2,
        oos_dates=fwd_oos.index,
    )
    log.info("  Tail-risk topo    → %s", trt_path)

    # ── Constellation risk ────────────────────────────────────────────────────
    log.info("  Generating constellation risk graph …")
    cst_path = constellation_risk(
        oos_states=state_oos.values.astype(np.float32),
        oos_actions=oos_weights,
        oos_returns=oos_portfolio_returns,
        feature_names=list(state_oos.columns),
        oos_dates=fwd_oos.index,
    )
    log.info("  Constellation     → %s", cst_path)

    # ── Cumulative return comparison ──────────────────────────────────────────
    log.info("  Generating cumulative return comparison plot …")
    # Build benchmark dict: SPX B&H is the primary benchmark; include EW S&P
    # if constituent data is available.
    _bm_dict: dict[str, pd.Series] = {
        "SPX B&H": fwd_oos,
    }
    try:
        _ew_ret = constituent_returns_k.reindex(fwd_oos.index).mean(axis=1)
        _bm_dict["Equal-Weight S&P 500"] = _ew_ret
    except Exception:
        pass
    crc_path = cumulative_return_comparison(
        rl_returns=oos_portfolio_returns,
        benchmark_returns=_bm_dict,
    )
    log.info("  Cumulative return → %s", crc_path)

    log.info("  Phase IV · Evaluation & Plotting complete  (%.1f s)", time.perf_counter() - t_phase_IV)

    # ══════════════════════════════════════════════════════════════════════════
    _sec("PIPELINE COMPLETE — Summary Report")
    # ══════════════════════════════════════════════════════════════════════════

    elapsed = time.perf_counter() - t_pipeline
    log.info("  Total runtime          : %.1f s", elapsed)
    log.info("")
    log.info("  Data")
    log.info("  %-30s : %s → %s", "Window", DATA_START, DATA_END)
    log.info("  %-30s : %d rows × %d features", "State tensor", len(state), state_dim)
    log.info("  %-30s : %d rows", "In-sample", len(state_is))
    log.info("  %-30s : %d rows", "OOS", len(state_oos))
    log.info("")
    log.info("  Signal Gating")
    log.info("  %-30s : %d / %d pass BY-FDR", "Signals", passed.sum(), len(passed))
    log.info("")
    log.info("  Simulation")
    log.info("  %-30s : %s", "VECM", "OK" if vecm_ok else "SVD fallback (VAR(1) residual bootstrap)")
    log.info("  %-30s : %d paths × %d steps", "Paths generated", N_SIM_PATHS, N_SIM_STEPS)
    log.info("")
    log.info("  DRL Training")
    log.info("  %-30s : %d outer epochs × %d PPO updates", "Training", N_OUTER_EPOCHS, N_PPO_UPDATES)
    log.info("  %-30s : %d / %d gradient steps", "Total updates",
             N_OUTER_EPOCHS * N_PPO_UPDATES * (N_SIM_PATHS // MINI_BATCH),
             N_OUTER_EPOCHS * N_PPO_UPDATES * (N_SIM_PATHS // MINI_BATCH))
    log.info("  %-30s : %+.6f", "Final mean reward", epoch_rewards[-1])
    log.info("  %-30s : %.4f", "Dual variable η", cvar_ctrl.eta)
    log.info("")
    log.info("  OOS Performance")
    log.info("  %-30s : %+.4f%% (SPX: %+.4f%%)", "Annual return",
             ann_ret_rl * 100, ann_ret_bm * 100)
    log.info("  %-30s : %+.4f  (SPX: %+.4f)", "Sharpe ratio", sharpe_rl, sharpe_bm)
    log.info("  %-30s : %.4f%%", "Max drawdown", mdd_rl * 100)
    log.info("  %-30s : %.2f%%", "Direction hit rate", hit_rate * 100)
    log.info("")
    log.info("  Attribution")
    log.info("  %-30s : %+.6f  (t=%+.3f  p=%.4f)",
             "Alpha (β₀)", alpha_row["coef"], alpha_row["t_stat"], alpha_row["p_value"])
    log.info("  %-30s : %s", "Outcome", outcome)
    log.info("")
    log.info(BANNER)


if __name__ == "__main__":
    main()
