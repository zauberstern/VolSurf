"""
Phase I extension — Per-Constituent Option Signal Extractor.

Streams the OptionMetrics per-constituent option prices CSV (slim 2003–2024
export) and computes daily cross-sectional implied-volatility signals.

File: D:/Downloads/option_prices_slim_2003_2024.csv
  - ~16 GB, 15 columns, covers 2003-2024 (full IS + OOS range)
  - Pre-filtered: DTE 14–45, bid ≥ $0.40, exercise_style='A'
  - Strike prices in cents (e.g. 25000 = $250.00)
  - key columns: secid, date, exdate, cp_flag, delta, impl_volatility,
    best_bid, best_offer

Extracted signals (daily cross-sectional aggregates):
  mean_constituent_iv   — mean ATM 30d call IV across all S&P 500 constituents
  iv_dispersion_raw     — coefficient of variation (std/mean) of per-stock ATM IVs
  mean_constituent_skew — mean (25d put IV – 25d call IV) across constituents
  constituent_ba_spread — mean relative bid-ask of ATM options (option liquidity)
  n_stocks              — number of constituents with valid data each day

Full 2003–2024 coverage eliminates the need for IS VIX-ratio imputation
(impute_constituent_iv_for_is is retained for backward compatibility but
is no longer called by the main pipeline).

Results are cached to a small parquet file; the large CSV is read only once.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_CSV  = Path(r"D:\Downloads\option_prices_slim_2003_2024.csv")
_CACHE_DIR    = Path(__file__).parents[2] / "data" / "wrds_cache"
_CACHE_FILE   = _CACHE_DIR / "constituent_option_signals_2003_2024.parquet"
_CHUNK_SIZE   = 5_000_000   # rows per read (~1.5 GB RAM per chunk at ~8 cols)

# Columns we actually need — read nothing else to minimise I/O
_USECOLS = [
    "secid", "date", "exdate", "cp_flag",
    "delta", "impl_volatility", "best_bid", "best_offer",
]

# DTE window for 30-day tier
_DTE_MIN_30 = 14   # 2 weeks floor
_DTE_MAX_30 = 45   # ~6 weeks cap


def compute_constituent_option_signals(
    csv_path: Path | str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Stream-compute daily constituent option signals from the raw opprcd CSV.

    The function reads the 30+ GB option prices file ONCE and produces a
    compact daily DataFrame.  On subsequent calls it returns the parquet cache.

    Processing logic per chunk:
      1. Compute DTE = exdate – date (calendar days).
      2. Restrict to DTE ∈ [14, 45] with valid IV and delta.
      3. ATM calls   (cp='C', delta ∈ [0.35, 0.65]) → iv_30 per (secid, date).
      4. 25d put     (cp='P', delta ∈ [–0.30, –0.20]) → iv_25dp.
      5. 25d call    (cp='C', delta ∈ [0.20, 0.30])   → iv_25dc.
      6. ATM option bid-ask: (offer–bid)/(offer+bid)/2 for ATM calls.

    After all chunks, aggregate cross-sectionally per date.

    Parameters
    ----------
    csv_path:
        Path to the raw option prices CSV.  Defaults to the 35 GB file at
        D:/Downloads/option_prices_2003-2024_sp500_all_constituents.csv.
    force_refresh:
        Re-process the full CSV even if a cache already exists.

    Returns
    -------
    pd.DataFrame
        Index: date (DatetimeIndex, daily).
        Columns: mean_constituent_iv, iv_dispersion_raw, mean_constituent_skew,
                 constituent_ba_spread, n_stocks.
        Empty DataFrame when the source CSV is not found.
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV

    if _CACHE_FILE.exists() and not force_refresh:
        df = pd.read_parquet(_CACHE_FILE)
        log.info(
            "Constituent options cache hit: %s  (%d obs  mean_iv μ=%.4f  "
            "iv_disp μ=%.4f)",
            _CACHE_FILE.name, len(df),
            float(df["mean_constituent_iv"].mean()),
            float(df["iv_dispersion_raw"].mean()),
        )
        return df

    if not path.exists():
        log.warning(
            "Constituent options CSV not found: %s — returning empty DataFrame",
            path,
        )
        return pd.DataFrame(
            columns=[
                "mean_constituent_iv", "iv_dispersion_raw",
                "mean_constituent_skew", "constituent_ba_spread", "n_stocks",
            ]
        )

    log.info(
        "Streaming constituent options CSV → daily signals  "
        "(one-time, ~10-40 min for 16 GB) …  %s",
        path,
    )
    t0 = time.perf_counter()

    # Accumulators: (secid, date_str) → list[float]
    # Sized for 500 stocks × 500 days × ≤10 matching options = small dict
    atm30_iv:    dict[tuple, list[float]] = {}
    put25_iv:    dict[tuple, list[float]] = {}
    call25_iv:   dict[tuple, list[float]] = {}
    atm30_ba:    dict[tuple, list[float]] = {}   # bid-ask half-spread

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        usecols=_USECOLS,
        dtype={
            "secid":          "int32",
            "cp_flag":        "category",
            "delta":          "float32",
            "impl_volatility": "float32",
            "best_bid":       "float32",
            "best_offer":     "float32",
        },
        chunksize=_CHUNK_SIZE,
        low_memory=False,
    ):
        total_rows += len(chunk)

        # Parse dates; compute DTE
        chunk["date"]   = pd.to_datetime(chunk["date"],   errors="coerce")
        chunk["exdate"] = pd.to_datetime(chunk["exdate"], errors="coerce")
        chunk["dte"]    = (chunk["exdate"] - chunk["date"]).dt.days

        # Global validity filter
        valid = (
            chunk["impl_volatility"].notna()
            & (chunk["impl_volatility"] > 0.0)
            & chunk["delta"].notna()
            & (chunk["dte"] >= _DTE_MIN_30)
            & (chunk["dte"] <= _DTE_MAX_30)
        )
        sub = chunk.loc[valid].copy()
        if sub.empty:
            continue

        cp = sub["cp_flag"].astype(str)

        # ── ATM calls: delta ∈ [0.35, 0.65] ─────────────────────────────────
        m_atm = (cp == "C") & sub["delta"].between(0.35, 0.65)
        _accumulate(sub.loc[m_atm], atm30_iv, "impl_volatility")
        # bid-ask half-spread for same options: (offer-bid)/(offer+bid)/2
        atm_sub = sub.loc[m_atm].copy()
        mid = (atm_sub["best_bid"] + atm_sub["best_offer"]) / 2.0
        ba_valid = (
            atm_sub["best_bid"].notna()
            & atm_sub["best_offer"].notna()
            & (mid > 0.0)
        )
        if ba_valid.any():
            atm_ba = atm_sub.loc[ba_valid].copy()
            atm_ba["ba_spread"] = (
                (atm_ba["best_offer"] - atm_ba["best_bid"])
                / (atm_ba["best_bid"] + atm_ba["best_offer"])
            )
            _accumulate(atm_ba, atm30_ba, "ba_spread")

        # ── 25-delta puts: delta ∈ [–0.30, –0.20] ───────────────────────────
        m_p25 = (cp == "P") & sub["delta"].between(-0.30, -0.20)
        _accumulate(sub.loc[m_p25], put25_iv, "impl_volatility")

        # ── 25-delta calls: delta ∈ [0.20, 0.30] ────────────────────────────
        m_c25 = (cp == "C") & sub["delta"].between(0.20, 0.30)
        _accumulate(sub.loc[m_c25], call25_iv, "impl_volatility")

        if total_rows % (20 * _CHUNK_SIZE) == 0:
            log.info(
                "  %.0fM rows processed | %d secid-date keys | %.0fs elapsed",
                total_rows / 1e6,
                len(atm30_iv),
                time.perf_counter() - t0,
            )

    log.info(
        "  CSV scan complete: %.0fM total rows | %d secid-date keys | %.0fs",
        total_rows / 1e6,
        len(atm30_iv),
        time.perf_counter() - t0,
    )

    # ── Build per-stock-day DataFrame ────────────────────────────────────────
    all_keys = set(atm30_iv) | set(put25_iv)
    records: list[dict] = []
    for key in all_keys:
        secid, dt = key
        iv30 = _safe_median(atm30_iv.get(key))
        iv25dp = _safe_median(put25_iv.get(key))
        iv25dc = _safe_median(call25_iv.get(key))
        ba = _safe_median(atm30_ba.get(key))
        skew = (iv25dp - iv25dc) if (not np.isnan(iv25dp) and not np.isnan(iv25dc)) else np.nan
        records.append({
            "secid": secid,
            "date":  dt,
            "iv_30": iv30,
            "skew":  skew,
            "ba":    ba,
        })

    stock_panel = pd.DataFrame(records)
    if stock_panel.empty:
        log.warning("No qualifying options found in %s", path)
        return pd.DataFrame(
            columns=[
                "mean_constituent_iv", "iv_dispersion_raw",
                "mean_constituent_skew", "constituent_ba_spread", "n_stocks",
            ]
        )

    # ── Cross-sectional daily aggregation ────────────────────────────────────
    def _cv(s: pd.Series) -> float:
        vals = s.dropna()
        if len(vals) < 3 or vals.mean() <= 0:
            return np.nan
        return float(vals.std() / vals.mean())

    daily = (
        stock_panel
        .groupby("date")
        .agg(
            mean_constituent_iv=("iv_30", "mean"),
            iv_dispersion_raw=(  "iv_30",  _cv),
            mean_constituent_skew=("skew", "mean"),
            constituent_ba_spread=("ba",   "mean"),
            n_stocks=("iv_30", "count"),
        )
    )
    daily.index = pd.to_datetime(daily.index.astype("int64"), unit="ns")
    daily.sort_index(inplace=True)

    # ── Cache ─────────────────────────────────────────────────────────────────
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(_CACHE_FILE)
    log.info(
        "  Cached to %s  (%d obs  mean_iv μ=%.4f  iv_disp μ=%.4f  "
        "elapsed=%.0fs)",
        _CACHE_FILE.name,
        len(daily),
        float(daily["mean_constituent_iv"].mean()),
        float(daily["iv_dispersion_raw"].mean()),
        time.perf_counter() - t0,
    )
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _accumulate(
    sub: pd.DataFrame,
    accum: dict[tuple, list[float]],
    col: str,
) -> None:
    """Group sub by (secid, date) and extend accum lists."""
    if sub.empty:
        return
    # Use integer date representation to avoid Timestamp hashing overhead
    dates_int = sub["date"].dt.normalize().values.astype("int64")
    for i, (secid, dt_int) in enumerate(zip(sub["secid"].values, dates_int)):
        key = (int(secid), int(dt_int))
        val = float(sub.iloc[i][col])
        if np.isfinite(val):
            if key not in accum:
                accum[key] = []
            accum[key].append(val)


def _safe_median(lst: list[float] | None) -> float:
    if not lst:
        return np.nan
    return float(np.median(lst))


def impute_constituent_iv_for_is(
    daily_oos: pd.DataFrame,
    vix_series: pd.Series,
    is_end: str = "2021-12-31",
) -> pd.DataFrame:
    """Impute IS mean_constituent_iv using vix × median ratio (2023-2024).

    Cross-sectional mean constituent IV ≈ c × VIX where c is typically
    1.1–1.3 (dispersion premium; see Drechsler & Yaron 2011).  We fit c
    from the available 2023-2024 data and then back-fill using VIX.

    Parameters
    ----------
    daily_oos:
        Output of compute_constituent_option_signals() (2023-2024 dates).
    vix_series:
        Full-history VIX series (decimal, e.g. 0.20 = 20 VIX points).
    is_end:
        In-sample end date; IS data is imputed for dates ≤ is_end.

    Returns
    -------
    pd.DataFrame
        Same columns as daily_oos, extended back to the start of vix_series.
    """
    if daily_oos.empty or vix_series.empty:
        return daily_oos

    # Align OOS IV with OOS VIX
    vix_oos = vix_series.reindex(daily_oos.index).dropna()
    iv_oos  = daily_oos["mean_constituent_iv"].reindex(vix_oos.index).dropna()
    common  = vix_oos.index.intersection(iv_oos.index)

    if len(common) < 10:
        log.warning(
            "impute_constituent_iv_for_is: too few common dates (%d); "
            "skipping IS imputation",
            len(common),
        )
        return daily_oos

    ratio = float((iv_oos.loc[common] / vix_oos.loc[common]).median())
    log.info(
        "  Constituent IV / VIX median ratio (2023-2024) = %.4f  "
        "(c ≈ 1.1–1.3 expected)",
        ratio,
    )

    # Build IS rows
    is_mask = vix_series.index <= pd.Timestamp(is_end)
    is_iv   = vix_series.loc[is_mask] * ratio

    is_df = pd.DataFrame(
        {
            "mean_constituent_iv":    is_iv.values,
            "iv_dispersion_raw":      np.nan,   # no IS data; VECM treats as near-constant
            "mean_constituent_skew":  np.nan,
            "constituent_ba_spread":  np.nan,
            "n_stocks":               0,
        },
        index=is_iv.index,
    )

    # Combine: prefer OOS observed values over any IS overlap
    combined = pd.concat([is_df, daily_oos])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    return combined
