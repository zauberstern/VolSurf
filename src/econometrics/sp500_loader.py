"""
S&P 500 Constituent Data Loader — Point-in-Time.

Derives the S&P 500 universe from a local CSV file of current constituents
(S&P500.csv) and applies point-in-time filtering using each stock's index-
entry date (fromdate) and index-exit date sourced from the fja05680/sp500
GitHub dataset (sp500_ticker_start_end.csv, 832 stars, MIT license).

Point-in-time guarantees
------------------------
  • Pre-entry returns are NaN'd (prevents look-ahead bias).
  • Post-exit returns are NaN'd (prevents survivorship bias).
  • Delisting dates from crsp.dsedelist are also applied.

Tables / files used
-------------------
- S&P500.csv          — current constituents with gvkey + fromdate (local)
- crsp.ccmxpf_lnkhist — gvkey → CRSP PERMNO link table
- crsp.dsf            — daily stock file (ret per PERMNO)
- fja05680/sp500      — GitHub ticker PIT membership CSV (auto-downloaded)

Cache keys
----------
  data/wrds_cache/sp500_permnos_YYYYMMDD.parquet     -- gvkey/permno/entry_date
  data/wrds_cache/sp500_prices_<start>_<end>.parquet -- wide return panel
  data/wrds_cache/sp500_pit_membership.parquet        -- fja05680 PIT data
"""

from __future__ import annotations

import io
import logging
import os
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

_CACHE_DIR       = Path(__file__).parents[2] / "data" / "wrds_cache"
_CSV_PATH        = Path(__file__).parents[2] / "S&P500.csv"
_PIT_URL         = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
_PIT_CACHE       = _CACHE_DIR / "sp500_pit_membership.parquet"
# Bundled snapshot so the model doesn't depend on GitHub staying up
_PIT_SNAPSHOT    = Path(__file__).parents[2] / "data" / "sp500_pit_membership_snapshot.csv"
# Locally downloaded WRDS crsp.dsf export (2003-2024, all historical S&P 500 constituents)
_LOCAL_CRSP_CSV  = Path(__file__).parents[2] / "data" / "sp500_prices_2003_2024_all_constituents.csv"


# ---------------------------------------------------------------------------
# PIT membership from fja05680/sp500 (GitHub)
# ---------------------------------------------------------------------------

def download_pit_membership(force_refresh: bool = False) -> pd.DataFrame:
    """Download and cache S&P 500 point-in-time membership from fja05680/sp500.

    Returns DataFrame with columns: ticker, start_date (pd.Timestamp),
    end_date (pd.Timestamp | NaT — NaT means still in index as of Jan 2026).

    Data provenance: https://github.com/fja05680/sp500 (MIT license, 832 stars).
    Maintained manually from Wikipedia + Andreas Clenow original data.
    Covers 1996-01-02 onwards with exact business-day precision.
    """
    if _PIT_CACHE.exists() and not force_refresh:
        df = pd.read_parquet(_PIT_CACHE)
        log.info("PIT membership cache hit: %d ticker-intervals", len(df))
        return df

    log.info("Downloading S&P 500 PIT membership from fja05680/sp500 …")
    raw: str | None = None
    try:
        with urllib.request.urlopen(_PIT_URL, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        if _PIT_SNAPSHOT.exists():
            log.warning(
                "PIT download failed (%s) — using bundled snapshot %s",
                exc, _PIT_SNAPSHOT.name,
            )
        else:
            log.warning("PIT download failed: %s  — returning empty DataFrame", exc)
            return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    df = pd.read_csv(io.StringIO(raw) if raw is not None else _PIT_SNAPSHOT)
    df.columns = [c.strip() for c in df.columns]
    df["ticker"]     = df["ticker"].str.strip()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    # Empty end_date → NaT (ticker still in index as of publication date)
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")

    df = df.dropna(subset=["ticker", "start_date"])
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PIT_CACHE, index=False)
    log.info(
        "PIT membership: %d ticker-intervals cached  (%.0f%% still active)",
        len(df), 100 * df["end_date"].isna().mean(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 1 - load constituent list from CSV + CCM link
# ---------------------------------------------------------------------------

def load_constituent_universe(db) -> pd.DataFrame:
    """Load S&P 500 universe: gvkey -> permno + entry_date.

    Reads S&P500.csv for the constituent list (gvkey, ticker, fromdate),
    then joins to crsp.ccmxpf_lnkhist for the CRSP permno.

    Returns DataFrame with columns [gvkey, ticker, company, entry_date, permno].
    """
    cache_path = _CACHE_DIR / f"sp500_permnos_{date.today().isoformat()}.parquet"
    if cache_path.exists():
        log.info("Constituent universe cache hit: %s", cache_path.name)
        return pd.read_parquet(cache_path)

    if not _CSV_PATH.exists():
        raise FileNotFoundError(f"S&P500.csv not found at {_CSV_PATH}")
    csv = pd.read_csv(_CSV_PATH)
    csv["gvkey"] = csv["gvkey"].astype(str).str.zfill(6)
    csv["entry_date"] = pd.to_datetime(csv["fromdate"], errors="coerce")
    csv = csv.rename(columns={"tic": "ticker", "companyname": "company"})
    csv = csv[["gvkey", "ticker", "company", "entry_date"]].dropna(subset=["gvkey"])
    log.info("Loaded %d constituents from S&P500.csv", len(csv))

    gvkeys = csv["gvkey"].tolist()
    quoted = ", ".join(f"'{g}'" for g in gvkeys)
    sql = f"""
        SELECT DISTINCT ON (gvkey)
            gvkey,
            lpermno AS permno,
            linkdt,
            linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE gvkey IN ({quoted})
          AND linktype IN ('LU','LC')
          AND linkprim IN ('P','C')
        ORDER BY gvkey, linkenddt DESC NULLS FIRST
    """
    try:
        link = db.raw_sql(sql)
        link["gvkey"]  = link["gvkey"].astype(str).str.zfill(6)
        link["permno"] = link["permno"].astype("Int64")
        log.info("CCM: matched %d / %d gvkeys -> permno", len(link), len(gvkeys))
    except Exception as exc:
        log.error("CCM link failed: %s", exc)
        return pd.DataFrame(columns=["gvkey", "ticker", "company", "entry_date", "permno"])

    universe = csv.merge(link[["gvkey", "permno"]], on="gvkey", how="inner")
    universe = universe.dropna(subset=["permno"])
    universe["permno"] = universe["permno"].astype(int)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(cache_path, index=False)
    log.info("Universe saved: %d stocks", len(universe))
    return universe


# ---------------------------------------------------------------------------
# Step 2 - fetch daily returns from crsp.dsf
# ---------------------------------------------------------------------------

def fetch_sp500_prices(
    db,
    permnos: list[int],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch wide daily return panel from crsp.dsf.

    Returns DataFrame: index=date, columns=permno (int), values=decimal return.
    """
    cache_path = _CACHE_DIR / f"sp500_prices_{start.replace('-','')}_{end.replace('-','')}.parquet"
    if use_cache and cache_path.exists():
        log.info("Price panel cache hit: %s", cache_path.name)
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df

    permno_list = ", ".join(str(p) for p in permnos)
    sql = f"""
        SELECT permno, date, ret
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date >= '{start}'
          AND date <= '{end}'
        ORDER BY date, permno
    """
    try:
        long = db.raw_sql(sql)
        long["date"]   = pd.to_datetime(long["date"])
        long["permno"] = long["permno"].astype(int)
        long["ret"]    = pd.to_numeric(long["ret"], errors="coerce")
        wide = long.pivot(index="date", columns="permno", values="ret")
        wide.index.name   = "date"
        wide.columns.name = "permno"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(cache_path)
        log.info("Price panel: %d days x %d stocks", len(wide), wide.shape[1])
        return wide
    except Exception as exc:
        log.error("fetch_sp500_prices failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 3 - apply point-in-time filter (entry + exit dates)
# ---------------------------------------------------------------------------

def apply_point_in_time_filter(
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    pit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Set returns to NaN outside each stock's S&P 500 membership window.

    Applies two complementary filters:
    1. Pre-entry mask: zeroes returns before each stock's fromdate (prevents
       look-ahead bias from including pre-membership alpha).
    2. Post-exit mask (optional): when pit_df is provided, zeroes returns after
       each ticker's confirmed index-exit date (prevents survivorship bias).

    Parameters
    ----------
    returns:
        Wide DataFrame: index=date, columns=permno, values=daily return.
    universe:
        Constituent DataFrame with columns [permno, ticker, entry_date].
    pit_df:
        Point-in-time membership from download_pit_membership().  Columns:
        [ticker, start_date, end_date].  If None, only entry dates are used.
    """
    entry_map = (
        universe[["permno", "entry_date"]]
        .dropna(subset=["entry_date"])
        .groupby("permno")["entry_date"]
        .min()          # earliest entry date wins; also deduplicates index
    )
    common = [c for c in returns.columns if c in entry_map.index]
    filtered = returns.copy()
    n_entry_zeroed = 0
    for permno in common:
        cutoff = entry_map[permno]
        if not isinstance(cutoff, pd.Timestamp):
            cutoff = pd.Timestamp(cutoff)
        # Lag by 1 business day: S&P reconstitutions are known at end-of-day,
        # so trading starts the next business day (PIT discipline).
        cutoff = cutoff + pd.offsets.BDay(1)
        mask = filtered.index < cutoff
        n_entry_zeroed += int(mask.sum())
        filtered.loc[mask, permno] = np.nan
    log.info(
        "Point-in-time entry filter: zeroed %d pre-entry observations across %d stocks",
        n_entry_zeroed, len(common),
    )

    # ── Post-exit masking via fja05680 PIT data ───────────────────────────────
    if pit_df is not None and not pit_df.empty:
        # Build ticker → (last exit date) lookup; stocks with NaT end_date are
        # still in the index — no post-exit mask applied.
        # Build ticker → permno lookup; drop_duplicates guards against
        # multiple PERMNOs sharing a ticker in historical data.
        ticker_map = (
            universe[["permno", "ticker"]]
            .dropna()
            .drop_duplicates(subset=["ticker"], keep="last")
            .set_index("ticker")["permno"]
        )

        n_exit_zeroed = 0
        n_pit_matched = 0
        for _, row in pit_df.iterrows():
            ticker    = row["ticker"]
            exit_date = row["end_date"]   # NaT = still active
            if pd.isna(exit_date):
                continue
            if ticker not in ticker_map.index:
                continue
            permno = ticker_map[ticker]
            if permno not in filtered.columns:
                continue
            mask = filtered.index > exit_date
            n_exit_zeroed += int(mask.sum())
            n_pit_matched += 1
            filtered.loc[mask, permno] = np.nan

        log.info(
            "Point-in-time exit filter: zeroed %d post-exit observations "
            "across %d exited tickers",
            n_exit_zeroed, n_pit_matched,
        )

    return filtered


# ---------------------------------------------------------------------------
# Combined loader — local CSV path (no WRDS required)
# ---------------------------------------------------------------------------

def _build_universe_from_csv(raw: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal universe DataFrame from S&P500.csv joined to the CRSP CSV.

    Parameters
    ----------
    raw:
        Long-format CRSP DataFrame with columns [PERMNO, TICKER, ...].

    Returns
    -------
    DataFrame with columns [ticker, entry_date, permno].
    """
    # PERMNO → most-recent TICKER mapping from the raw data
    permno_ticker = (
        raw[["PERMNO", "TICKER"]]
        .drop_duplicates()
        .rename(columns={"PERMNO": "permno", "TICKER": "ticker"})
        .sort_values("ticker")
        .drop_duplicates(subset=["permno"], keep="last")
    )

    if not _CSV_PATH.exists():
        log.warning("S&P500.csv not found; universe will lack entry_date")
        permno_ticker["entry_date"] = pd.NaT
        return permno_ticker

    sp500 = pd.read_csv(_CSV_PATH)
    sp500["entry_date"] = pd.to_datetime(sp500["fromdate"], errors="coerce")
    sp500 = sp500.rename(columns={"tic": "ticker"})
    sp500 = sp500[["ticker", "entry_date"]].dropna(subset=["ticker"])

    universe = sp500.merge(permno_ticker, on="ticker", how="inner")
    universe = universe.dropna(subset=["permno"])
    universe["permno"] = universe["permno"].astype(int)
    log.info("Universe from CSV: %d ticker-permno pairs", len(universe))
    return universe


def load_sp500_from_local_csv(
    start: str = "2003-01-01",
    end: str = "2024-12-31",
    use_cache: bool = True,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Load S&P 500 constituent returns from the locally downloaded WRDS CSV.

    No WRDS connection required.  Reads
    ``data/sp500_prices_2003_2024_all_constituents.csv`` (crsp.dsf export),
    applies the standard CRSP delisting-return adjustment, applies PIT entry
    and exit filters, and caches the result to
    ``data/wrds_cache/sp500_prices_<start>_<end>.parquet``.

    Parameters
    ----------
    start, end:
        Date range to slice.  Must fall within 2003-01-01 → 2024-12-31.
    use_cache:
        If True and the parquet cache exists, skip all processing.

    Returns
    -------
    (returns, universe)
        returns  — wide DataFrame index=date, columns=permno, values=daily ret
        universe — DataFrame with ticker/entry_date/permno columns
    """
    cache_path = _CACHE_DIR / f"sp500_prices_{start.replace('-','')}_{end.replace('-','')}.parquet"
    if use_cache and cache_path.exists():
        log.info("Price panel cache hit: %s", cache_path.name)
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        # Return a minimal universe (no entry_date needed — PIT already applied)
        return df, pd.DataFrame(columns=["ticker", "entry_date", "permno"])

    if not _LOCAL_CRSP_CSV.exists():
        raise FileNotFoundError(
            f"Local CRSP CSV not found at {_LOCAL_CRSP_CSV}.  "
            "Download crsp.dsf for the required date range from WRDS."
        )

    log.info("Reading local CRSP CSV: %s  (%s → %s)", _LOCAL_CRSP_CSV.name, start, end)
    raw = pd.read_csv(
        _LOCAL_CRSP_CSV,
        usecols=["PERMNO", "date", "RET", "DLSTCD", "DLRET", "TICKER"],
        dtype={"PERMNO": "int32", "TICKER": str},
        low_memory=False,
    )
    raw["date"]   = pd.to_datetime(raw["date"])
    # CRSP uses 'B' (no trade) and 'C' (censored/price unavailable) → NaN
    raw["RET"]    = pd.to_numeric(raw["RET"],    errors="coerce")
    raw["DLRET"]  = pd.to_numeric(raw["DLRET"],  errors="coerce")
    raw["DLSTCD"] = pd.to_numeric(raw["DLSTCD"], errors="coerce")

    # Delisting-return adjustment: if RET is missing on the delisting date
    # but DLRET is available (and the code is not "still active"), use DLRET.
    _normal_codes = {100.0, 101.0, 102.0}
    delist_fill = (
        raw["RET"].isna()
        & raw["DLRET"].notna()
        & ~raw["DLSTCD"].isin(_normal_codes)
    )
    raw.loc[delist_fill, "RET"] = raw.loc[delist_fill, "DLRET"]
    log.info("Delisting-return fill applied to %d rows", int(delist_fill.sum()))

    # Slice to requested date range
    raw = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()

    # Pivot to wide format: index=date, columns=PERMNO (int)
    wide = raw.pivot_table(index="date", columns="PERMNO", values="RET", aggfunc="first")
    wide.index.name   = "date"
    wide.columns.name = "permno"
    wide.columns = wide.columns.astype(int)
    log.info("Raw wide panel: %d days × %d stocks", len(wide), wide.shape[1])

    # Build universe (permno + entry_date) for PIT filtering
    universe = _build_universe_from_csv(raw)

    # PIT entry filter
    returns = apply_point_in_time_filter(wide, universe)

    # PIT exit filter via fja05680/sp500
    try:
        pit_df = download_pit_membership()
        if not pit_df.empty:
            returns = apply_point_in_time_filter(returns, universe, pit_df=pit_df)
    except Exception as exc:
        log.warning("fja05680 PIT exit filter skipped: %s", exc)

    # Delisting mask: NaN all returns after confirmed delisting date
    delist_dates = (
        raw[raw["DLSTCD"].notna() & ~raw["DLSTCD"].isin(_normal_codes)]
        .groupby("PERMNO")["date"]
        .max()
    )
    n_masked = 0
    for permno, dlst_date in delist_dates.items():
        if permno in returns.columns:
            mask = returns.index > dlst_date
            n_masked += int(mask.sum())
            returns.loc[mask, permno] = np.nan
    log.info(
        "Delisting mask: %d post-delisting obs zeroed across %d stocks",
        n_masked, len(delist_dates),
    )

    # Drop columns with fewer than 50% non-null observations
    threshold = int(len(returns) * 0.5)
    returns = returns.dropna(axis=1, thresh=threshold)
    log.info("Final panel: %d days × %d stocks", len(returns), returns.shape[1])

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(cache_path)
    log.info("Cached → %s", cache_path.name)

    return returns, universe


# ---------------------------------------------------------------------------
# Bid-ask spread and ADV loader
# ---------------------------------------------------------------------------

def load_spread_adv_from_local_csv(
    start: str = "2003-01-01",
    end: str = "2024-12-31",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-stock quoted half-spread and 20-day rolling dollar ADV from CRSP.

    Reads BID, ASK, VOL, PRC from sp500_prices_2003_2024_all_constituents.csv
    and returns two wide DataFrames (index=date, columns=PERMNO int):

      half_spread_wide : (ASK - BID) / (2 * |PRC|)  — quoted half-spread fraction.
                         Rows where BID ≤ 0, ASK ≤ 0, or |PRC| ≤ 0.01 are NaN.
      adv_wide         : 20-day rolling mean of (VOL × |PRC|) in USD.
                         min_periods=5 so early rows are not fully NaN.

    Results are cached at data/wrds_cache/sp500_spread_adv_{start}_{end}.parquet
    with MultiIndex columns (level-0: half_spread | adv, level-1: PERMNO).

    The Glosten-Harris effective/quoted ratio (~0.6) is NOT applied here; callers
    should multiply half_spread_wide by cfg.training.spread_eff_factor themselves
    to separate raw data from modelling assumptions.
    """
    cache_path = _CACHE_DIR / (
        f"sp500_spread_adv_{start.replace('-','')}_{end.replace('-','')}.parquet"
    )
    if use_cache and cache_path.exists():
        log.info("Spread/ADV cache hit: %s", cache_path.name)
        combined = pd.read_parquet(cache_path)
        return combined["half_spread"], combined["adv"]

    if not _LOCAL_CRSP_CSV.exists():
        raise FileNotFoundError(
            f"Local CRSP CSV not found at {_LOCAL_CRSP_CSV}. "
            "Download crsp.dsf for the required date range from WRDS."
        )

    log.info(
        "Loading spread/ADV from CRSP CSV: %s (%s → %s) …",
        _LOCAL_CRSP_CSV.name, start, end,
    )
    raw = pd.read_csv(
        _LOCAL_CRSP_CSV,
        usecols=["PERMNO", "date", "BID", "ASK", "PRC", "VOL", "RET"],
        dtype={"PERMNO": "int32"},
        low_memory=False,
    )
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()

    raw["PRC"] = pd.to_numeric(raw["PRC"], errors="coerce").abs()
    raw["BID"] = pd.to_numeric(raw["BID"], errors="coerce")
    raw["ASK"] = pd.to_numeric(raw["ASK"], errors="coerce")
    raw["VOL"] = pd.to_numeric(raw["VOL"], errors="coerce")

    # Invalidate rows with non-positive or missing price/quote data
    valid = (raw["BID"] > 0) & (raw["ASK"] > 0) & (raw["PRC"] > 0.01)
    raw.loc[~valid, ["BID", "ASK", "VOL"]] = np.nan

    raw["half_spread"] = (raw["ASK"] - raw["BID"]) / (2.0 * raw["PRC"])
    raw["dollar_vol"]  = raw["VOL"] * raw["PRC"]

    # Pivot to wide: index=date, columns=PERMNO
    hs_wide = raw.pivot_table(
        index="date", columns="PERMNO", values="half_spread", aggfunc="first"
    )
    dv_wide = raw.pivot_table(
        index="date", columns="PERMNO", values="dollar_vol", aggfunc="first"
    )
    hs_wide.index.name = "date"
    dv_wide.index.name = "date"
    hs_wide.columns = hs_wide.columns.astype(int)
    dv_wide.columns = dv_wide.columns.astype(int)

    # 20-day rolling mean ADV (min_periods=5 to handle gaps at start of panel)
    adv_wide = dv_wide.rolling(window=20, min_periods=5).mean()

    # ── Gap 2: Roll (1984) estimator — validation diagnostic only ────────────
    # S_Roll = 2 * sqrt(max(0, -Cov(Δret_t, Δret_{t-1}))) on 20-day window.
    # Compares to CRSP quoted half-spread as a microstructure sanity check.
    # High Roll/quoted ratio (>0.8) flags stocks with stale or wide closing quotes.
    try:
        raw["RET"] = pd.to_numeric(raw["RET"], errors="coerce")
        ret_wide = raw.pivot_table(
            index="date", columns="PERMNO", values="RET", aggfunc="first"
        )
        ret_wide.index.name = "date"
        ret_wide.columns = ret_wide.columns.astype(int)

        def _roll_spread_series(r: pd.Series) -> pd.Series:
            """Per-stock 20-day rolling Roll (1984) spread estimate."""
            r_vals   = r.values
            lag_vals = r.shift(1).values
            n = 20
            cov_vals = np.full(len(r_vals), np.nan)
            for i in range(n - 1, len(r_vals)):
                w_r   = r_vals[i - n + 1 : i + 1]
                w_lag = lag_vals[i - n + 1 : i + 1]
                mask  = ~(np.isnan(w_r) | np.isnan(w_lag))
                if mask.sum() >= 10:
                    cov_vals[i] = float(np.cov(w_r[mask], w_lag[mask])[0, 1])
            return pd.Series(
                2.0 * np.sqrt(np.maximum(0.0, -cov_vals)),
                index=r.index,
            )

        roll_wide = ret_wide.apply(_roll_spread_series, axis=0)

        # Cross-stock median ratio over the full panel (where both are valid)
        _roll_med   = roll_wide.stack().median()
        _quoted_med = hs_wide.stack().median()
        if _quoted_med > 0:
            _ratio = _roll_med / _quoted_med
            log.info(
                "  Roll/quoted validation: Roll_med=%.2f bps  "
                "quoted_med=%.2f bps  ratio=%.2f%s",
                _roll_med * 1e4,
                _quoted_med * 1e4,
                _ratio,
                "  ✓ consistent" if 0.3 <= _ratio <= 0.9 else "  ⚠ divergent — review quote quality",
            )
    except Exception as _roll_exc:
        log.debug("Roll estimator validation skipped: %s", _roll_exc)

    # Cache with MultiIndex columns for a single parquet file
    combined = pd.concat({"half_spread": hs_wide, "adv": adv_wide}, axis=1)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    log.info(
        "Spread/ADV cached: %d days × %d stocks → %s",
        len(hs_wide), hs_wide.shape[1], cache_path.name,
    )
    return hs_wide, adv_wide


# ---------------------------------------------------------------------------
# Combined loader — WRDS live fetch
# ---------------------------------------------------------------------------

def load_sp500_data(
    start: str = "2003-01-01",
    end: str   = "2024-12-31",
    use_cache: bool = True,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: CSV -> CCM -> prices -> point-in-time filter.

    Tries a live WRDS connection first.  If WRDS is unavailable and the local
    CRSP CSV export exists (data/sp500_prices_2003_2024_all_constituents.csv),
    automatically falls back to ``load_sp500_from_local_csv``.

    Returns
    -------
    (returns, universe)
        returns  -- wide DataFrame index=date, columns=permno, values=daily ret
        universe -- DataFrame with gvkey/ticker/company/entry_date/permno
    """
    # Fast path: if the cache already exists, skip WRDS entirely.
    cache_path = _CACHE_DIR / f"sp500_prices_{start.replace('-','')}_{end.replace('-','')}.parquet"
    if use_cache and not force_refresh and cache_path.exists():
        log.info("Price panel cache hit: %s", cache_path.name)
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df, pd.DataFrame(columns=["ticker", "entry_date", "permno"])

    try:
        import wrds
    except ImportError:
        log.warning("wrds package not installed — falling back to local CSV")
        return load_sp500_from_local_csv(start=start, end=end, use_cache=use_cache)

    if force_refresh:
        if cache_path.exists():
            cache_path.unlink()

    db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME", ""))
    try:
        universe = load_constituent_universe(db)
        if universe.empty:
            raise RuntimeError("Empty constituent universe")
        permnos = universe["permno"].tolist()
        returns = fetch_sp500_prices(db, permnos, start, end, use_cache=use_cache)

        if returns.empty:
            return returns, universe

        returns = apply_point_in_time_filter(returns, universe)

        # Step 3a — apply fja05680 PIT exit-date filter (survivorship bias)
        try:
            pit_df = download_pit_membership()
            if not pit_df.empty:
                returns = apply_point_in_time_filter(returns, universe, pit_df=pit_df)
        except Exception as exc:
            log.warning("fja05680 PIT exit filter skipped: %s", exc)

        # Step 3b — mask returns after confirmed delisting (crsp.dsedelist)
        # Complements the PIT filter for bankruptcies and forced delistings.
        try:
            permno_list = ", ".join(str(p) for p in returns.columns.tolist())
            delist_sql = f"""
                SELECT permno, dlstdt, dlstcd
                FROM crsp.dsedelist
                WHERE permno IN ({permno_list})
                  AND dlstcd NOT IN (100, 101, 102)
            """
            delist = db.raw_sql(delist_sql)
            if not delist.empty:
                delist["permno"] = delist["permno"].astype(int)
                delist["dlstdt"] = pd.to_datetime(delist["dlstdt"])
                delist_map = delist.groupby("permno")["dlstdt"].min()
                n_delist = 0
                for permno, dlst_date in delist_map.items():
                    if permno in returns.columns:
                        mask = returns.index > dlst_date
                        n_delist += int(mask.sum())
                        returns.loc[mask, permno] = np.nan
                log.info(
                    "Delisting filter (dsedelist): %d post-delisting obs zeroed "
                    "across %d stocks", n_delist, len(delist_map),
                )
        except Exception as exc:
            log.warning("dsedelist filter skipped: %s", exc)

        threshold = int(len(returns) * 0.5)
        returns = returns.dropna(axis=1, thresh=threshold)
        log.info("Final panel: %d days x %d stocks", len(returns), returns.shape[1])

    except Exception as wrds_exc:
        db.close()
        if _LOCAL_CRSP_CSV.exists():
            log.warning(
                "WRDS fetch failed (%s) — falling back to local CRSP CSV", wrds_exc
            )
            return load_sp500_from_local_csv(start=start, end=end, use_cache=use_cache)
        raise
    finally:
        try:
            db.close()
        except Exception:
            pass

    if returns.empty:
        return returns, universe
    return returns, universe
