"""
Phase I — WRDS Data Loader.

Fetches 2-year panels of SPX option surface, index prices, VIX, and 10-year
Treasury yield from WRDS.  The returned data is shaped to match the exact
input signatures expected by build_state_vector().

Required WRDS subscriptions / tables used
-----------------------------------------
- OptionMetrics  (optionm)   — vsurfd{YYYY} year-partitioned vol surface
- CRSP           (crsp)      — S&P 500 daily index file (dsi)
- CBOE           (cboe)      — VIX daily closing (cboe.cboe, vix column)
- OptionMetrics  (optionm)   — 1-year zero-coupon rate (zerocd, days=365) [fallback]

External free sources (no auth required)
-----------------------------------------
- FRED DGS10 CSV  — 10-year Treasury CMT yield (primary; falls back to zerocd)

Credentials
-----------
Set WRDS_USERNAME and WRDS_PASSWORD in a .env file at the project root.
python-dotenv is optional: variables may be pre-set in the shell environment.

WRDS table names occasionally vary by institution.  The SQL in each
fetch_* function is clearly labelled so you can swap in the correct
table/column names if a query fails.
"""

from __future__ import annotations

import io
import logging
import os
import time
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on pre-set env vars

log = logging.getLogger(__name__)

def _configure_logging() -> None:
    """Set up a human-readable console handler if none is attached."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)

# OptionMetrics secid for S&P 500 index options
_SPX_SECID = 108105

# Local parquet cache — excluded from git via .gitignore
_CACHE_DIR = Path(__file__).parents[2] / "data" / "wrds_cache"
_SERIES_KEYS = ("prices", "vix", "treasury_10y")  # stored as single-column DataFrames

# Path to the locally provided raw opprcd CSV export (2003-2024)
_RAW_OPPRCD_CSV = Path(__file__).parents[2] / "SPX(2003-2024).csv"

# Local CBOE multi-index volatility CSV (VIX, VXN, VXD, VXO — 2003-2024)
_CBOE_MULTIVOL_CSV = Path(__file__).parents[2] / "data" / "cboe_vix_vox_2003_2024.csv"

# Local OptionMetrics historical (realized) vol CSV for S&P 500 constituents
_OPTIONM_HV_CSV = (
    Path(__file__).parents[2] / "data"
    / "sp500_historical_vol_2003_2024_all_constituents.csv"
)


def _load_spx_prices_from_cache() -> "pd.Series | None":
    """Return cached SPX prices (any available window) for spread normalisation."""
    for parquet in sorted(_CACHE_DIR.glob("prices_*.parquet")):
        try:
            df = pd.read_parquet(parquet)
            return df.iloc[:, 0]
        except Exception:
            continue
    return None


def _process_raw_opprcd_csv(start: str, end: str) -> "pd.Series | None":
    """Compute daily ATM SPX option half-spread (fraction of underlying) from the
    locally provided raw opprcd CSV export.

    Filters to near-ATM calls (|delta - 0.5| < 0.1) with 20-40 days to expiry
    and positive bid/offer.  Dollar half-spread is then divided by the SPX
    closing price on each date so that the result is expressed as a fraction of
    the underlying — the same units used for forward returns in the FWER gating
    test.

    Results are cached to a small parquet so the 38M-row CSV is only scanned
    once; subsequent calls return in under one second.

    Returns None when the CSV is absent or contains no qualifying rows.
    """
    safe_s, safe_e = start.replace("-", ""), end.replace("-", "")
    cache = _CACHE_DIR / f"spx_atm_halfspread_proc_{safe_s}_{safe_e}.parquet"

    if cache.exists():
        df = pd.read_parquet(cache)
        log.info(
            "Half-spread cache hit: %s  (%d obs, median=%.5f / %.1f bps)",
            cache.name, len(df), df["half_spread"].median(), df["half_spread"].median() * 1e4,
        )
        return df["half_spread"]

    if not _RAW_OPPRCD_CSV.exists():
        return None

    log.info(
        "Processing %s → ATM half-spreads %s–%s (one-time scan, ~1-2 min) …",
        _RAW_OPPRCD_CSV.name, start, end,
    )
    t0 = time.perf_counter()

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        _RAW_OPPRCD_CSV,
        usecols=["date", "exdate", "cp_flag", "best_bid", "best_offer", "delta"],
        chunksize=1_000_000,
    ):
        chunk["date"]   = pd.to_datetime(chunk["date"],   errors="coerce")
        chunk["exdate"] = pd.to_datetime(chunk["exdate"], errors="coerce")
        chunk = chunk[chunk["date"].notna() & chunk["exdate"].notna()]
        chunk = chunk[
            (chunk["cp_flag"] == "C")
            & (chunk["best_bid"] > 0)
            & (chunk["best_offer"] > 0)
            & chunk["delta"].between(0.4, 0.6)
        ]
        chunk["days"] = (chunk["exdate"] - chunk["date"]).dt.days
        chunk = chunk[chunk["days"].between(20, 40)]
        if len(chunk) > 0:
            chunk["dollar_hs"] = (chunk["best_offer"] - chunk["best_bid"]) / 2.0
            frames.append(chunk[["date", "dollar_hs"]])

    if not frames:
        log.warning("No ATM 20-40d call rows found in %s", _RAW_OPPRCD_CSV.name)
        return None

    df = pd.concat(frames, ignore_index=True)
    daily_hs = df.groupby("date")["dollar_hs"].median()

    # Normalise by SPX closing price → fraction of underlying (same units as daily returns)
    spx = _load_spx_prices_from_cache()
    if spx is not None:
        spx = spx.astype("float64")  # ensure plain float (Arrow-backed arrays cause issues)
        spx.index = pd.to_datetime(spx.index)  # ensure datetime index for alignment
        aligned = spx.reindex(daily_hs.index)
        valid = aligned.notna() & (aligned > 0)
        daily_hs = (daily_hs[valid] / aligned[valid]).rename("half_spread")
    else:
        log.warning(
            "SPX price cache not found; approximating underlying ≈ 1500 for normalisation"
        )
        daily_hs = (daily_hs / 1500.0).rename("half_spread")

    result = daily_hs.loc[start:end]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_frame().to_parquet(cache)
    log.info(
        "      → %d daily obs  (%.0fs)  median=%.5f (%.1f bps of underlying)",
        len(result), time.perf_counter() - t0, result.median(), result.median() * 1e4,
    )
    return result



def _cache_path(key: str, start: str, end: str) -> Path:
    safe = lambda s: s.replace("-", "")
    return _CACHE_DIR / f"{key}_{safe(start)}_{safe(end)}.parquet"


def _save_cache(data: dict, start: str, end: str) -> None:
    """Persist all four data panels to parquet files."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for key, val in data.items():
        df = val.to_frame() if isinstance(val, pd.Series) else val
        df.to_parquet(_cache_path(key, start, end))
    log.info("Cache written → %s", _CACHE_DIR)


def _load_cache(start: str, end: str) -> dict | None:
    """Return cached data dict if all four parquet files exist, else None."""
    keys = ["prices", "option_panel", "vix", "treasury_10y"]
    paths = {k: _cache_path(k, start, end) for k in keys}
    if not all(p.exists() for p in paths.values()):
        return None
    result = {}
    for key, path in paths.items():
        df = pd.read_parquet(path)
        result[key] = df.iloc[:, 0] if key in _SERIES_KEYS else df
    log.info("Cache hit — loaded from %s", _CACHE_DIR)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect():
    """Return an authenticated wrds.Connection using pgpass credentials."""
    import wrds  # deferred — not required for synthetic-data unit tests

    username = os.environ.get("WRDS_USERNAME")
    if not username:
        raise EnvironmentError("WRDS_USERNAME must be set in .env or environment.")
    log.info("Connecting to WRDS as '%s' …", username)
    conn = wrds.Connection(wrds_username=username)
    log.info("WRDS connection established.")
    return conn


# ---------------------------------------------------------------------------
# Individual fetch functions (each accepts an open db connection)
# ---------------------------------------------------------------------------

def fetch_spx_prices(db, start: str, end: str) -> pd.Series:
    """
    S&P 500 daily closing level from CRSP daily stock indices (crsp.dsi).

    Column used: spindx  — S&P 500 composite index level.
    """
    log.info("[1/4] Fetching SPX prices  (crsp.dsi)  %s → %s …", start, end)
    t0 = time.perf_counter()
    sql = f"""
        SELECT date, spindx AS close
        FROM crsp.dsi
        WHERE date BETWEEN '{start}' AND '{end}'
        ORDER BY date
    """
    df = db.raw_sql(sql, date_cols=["date"])
    df = df.dropna(subset=["close"])
    result = df.set_index("date")["close"].rename("spx")
    log.info("      → %d rows  (%.1fs)", len(result), time.perf_counter() - t0)
    return result


def _vsurfd_union_sql(years: list[int], start: str, end: str) -> str:
    """
    Build a UNION ALL query across year-partitioned optionm.vsurfd{YYYY} tables.

    OptionMetrics partitions the standardised volatility surface by calendar
    year.  The unified VIEW (volatility_surface_view) is inaccessible on this
    subscription due to optionm_europe permission restrictions.

    Delta convention in vsurfd: integer call-delta (0–100).
        delta=25 → OTM call (low IV side)
        delta=50 → ATM
        delta=75 → ITM call = economically equivalent to 25-delta OTM put (high IV side)
    Only cp_flag='C' rows are present for SPX index options in these tables.
    """
    parts = [
        f"SELECT secid, date, days, delta, cp_flag, impl_volatility "
        f"FROM optionm.vsurfd{yr} "
        f"WHERE secid = {_SPX_SECID} "
        f"  AND date BETWEEN '{start}' AND '{end}' "
        f"  AND days IN (30, 91) "
        f"  AND delta IN (10, 25, 50, 75, 90) "
        f"  AND cp_flag = 'C'"
        for yr in years
    ]
    return "\nUNION ALL\n".join(parts) + "\nORDER BY date, days, delta"


def fetch_option_panel(db, start: str, end: str) -> pd.DataFrame:
    """
    SPX standardised vol surface from OptionMetrics year-partitioned tables.

    Constructs three daily series:
        iv_30     — 30-day ATM (delta=50) implied vol, decimal
        iv_91     — 91-day ATM (delta=50) implied vol, decimal
        skew_25d  — 30-day wing spread: IV(delta=75) − IV(delta=25), decimal
                    Positive value = left (put) skew, typical for equity indices.
    """
    start_yr, end_yr = int(start[:4]), int(end[:4])
    years = list(range(start_yr, end_yr + 1))
    log.info(
        "[2/4] Fetching option surface (optionm.vsurfd%d–%d)  %s → %s …",
        years[0], years[-1], start, end,
    )
    t0 = time.perf_counter()

    df: pd.DataFrame | None = None
    while years:
        sql = _vsurfd_union_sql(years, start, end)
        try:
            df = db.raw_sql(sql, date_cols=["date"])
            break
        except Exception as exc:
            if "does not exist" in str(exc):
                dropped = years.pop()
                log.warning(
                    "optionm.vsurfd%d not yet available — retrying without it", dropped
                )
            else:
                raise
    if df is None:
        raise RuntimeError(f"No vsurfd year tables accessible for range {start}–{end}")

    df = df.dropna(subset=["impl_volatility"])
    log.info("      → %d raw rows fetched  (%.1fs)", len(df), time.perf_counter() - t0)

    def _pivot(days: int, delta: int) -> pd.Series:
        mask = (df["days"] == days) & (df["delta"] == delta)
        return df[mask].set_index("date")["impl_volatility"]

    iv_30    = _pivot(30, 50).rename("iv_30")
    iv_91    = _pivot(91, 50).rename("iv_91")
    c10      = _pivot(30, 10)   # 10-delta OTM call → very low IV (far OTM)
    c25      = _pivot(30, 25)   # 25-delta OTM call → low IV
    c75      = _pivot(30, 75)   # 75-delta call ≡ 25-delta OTM put → high IV
    c90      = _pivot(30, 90)   # 90-delta call ≡ 10-delta OTM put → very high IV
    skew_25d = (c75 - c25).rename("skew_25d")   # positive = left (put) skew
    skew_10d = (c90 - c10).rename("skew_10d")   # tail-risk skew premium

    panel = pd.concat([iv_30, iv_91, skew_25d, skew_10d], axis=1).dropna()
    panel.index.name = "date"
    log.info("      → %d clean panel rows after pivot + dropna", len(panel))
    return panel.reset_index()


def _fetch_cboe_csv(code: str, start: str, end: str) -> pd.Series | None:
    """Fetch a CBOE volatility index from their free public CSV endpoint.

    code : 'VIX' or 'VXO'.  Returns decimal series or None on network failure.
    """
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{code}_History.csv"
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read().decode()
        df = pd.read_csv(io.StringIO(raw), parse_dates=["DATE"])
        df = df.rename(columns={"DATE": "date", "CLOSE": "value"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        result = (df.set_index("date")["value"] / 100.0).rename(code.lower())
        log.info(
            "      \u2192 CBOE %s CSV  %d rows  (%.1fs)  mean=%.3f",
            code, len(result), time.perf_counter() - t0, result.mean(),
        )
        return result
    except Exception as exc:
        log.warning("CBOE %s CSV unavailable (%s) \u2014 will use WRDS fallback", code, exc)
        return None


def fetch_vix(db, start: str, end: str) -> pd.Series:
    """CBOE VIX daily closing level.

    Primary source: CBOE free public CSV (no auth, current to today).
    Fallback: WRDS cboe.cboe table.
    Returns decimal form (e.g. 0.20 = 20%).
    """
    log.info("[3/4] Fetching VIX  %s \u2192 %s \u2026", start, end)

    result = _fetch_cboe_csv("VIX", start, end)
    if result is not None and len(result) > 10:
        return result.rename("vix")

    # \u2500\u2500 WRDS fallback \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    log.info("      Falling back to WRDS cboe.cboe \u2026")
    t0 = time.perf_counter()
    sql = f"""
        SELECT date, vix / 100.0 AS vix
        FROM cboe.cboe
        WHERE date BETWEEN '{start}' AND '{end}'
        ORDER BY date
    """
    df = db.raw_sql(sql, date_cols=["date"])
    df = df.dropna(subset=["vix"])
    result = df.set_index("date")["vix"].rename("vix")
    log.info("      \u2192 %d rows  (%.1fs)  mean VIX=%.3f", len(result), time.perf_counter() - t0, result.mean())
    return result


def fetch_vxo(start: str, end: str) -> pd.Series | None:
    """CBOE VXO index from the free public CSV.  Returns None after Sep 2021
    (CBOE discontinued VXO publication).  Required for pre-2003 VIX alignment.

    Results are cached in data/wrds_cache/ to avoid re-downloading each run.
    """
    cached = _load_supp("vxo", start, end)
    if cached is not None:
        s = cached.iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached
        return s.rename("vxo")
    result = _fetch_cboe_csv("VXO", start, end)
    if result is not None:
        s = result.rename("vxo")
        _save_supp("vxo", start, end, s.to_frame())
        return s
    return None


# ---------------------------------------------------------------------------
# Free external supplementary data (no WRDS required)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Supplementary-data cache helpers  (ff_factors, vxo — free downloads)
# ---------------------------------------------------------------------------

def _supp_cache_path(key: str, start: str, end: str) -> Path:
    """Cache file for supplementary (non-WRDS) data."""
    safe = lambda s: s.replace("-", "")
    return _CACHE_DIR / f"{key}_supp_{safe(start)}_{safe(end)}.parquet"


def _load_supp(key: str, start: str, end: str) -> pd.DataFrame | None:
    p = _supp_cache_path(key, start, end)
    if p.exists():
        df = pd.read_parquet(p)
        log.info("Supplementary cache hit: %s (%d rows)", p.name, len(df))
        return df
    return None


def _save_supp(key: str, start: str, end: str, df: pd.DataFrame) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_supp_cache_path(key, start, end))
    log.info("Supplementary cache written: %s", _supp_cache_path(key, start, end).name)


def fetch_ff_factors(start: str, end: str) -> pd.DataFrame:
    """Fama-French 3-factor daily returns from Ken French's data library.

    Source: Dartmouth Tuck public ZIP (no auth required).
    Columns: Mkt-RF, SMB, HML, RF  — all in decimal form (0.01 = 1%).
    Returns an empty DataFrame if the fetch fails (pipeline degrades gracefully).

    Use ``Mkt-RF`` as the market excess return in attribution regression.
    ``RF`` is the daily risk-free rate (1-month T-bill).

    Results are cached in data/wrds_cache/ to avoid re-downloading each run.
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    log.info("Fetching Ken French daily factors  %s \u2192 %s \u2026", start, end)
    t0 = time.perf_counter()

    # Cache check
    cached_ff = _load_supp("ff_factors", start, end)
    if cached_ff is not None:
        return cached_ff

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            content = z.read(z.namelist()[0]).decode(errors="replace")
        rows: list[dict] = []
        for line in content.split("\n"):
            line = line.strip()
            # Data rows: 8-digit YYYYMMDD date in first field
            if len(line) > 8 and line[:8].isdigit():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    try:
                        rows.append({
                            "date":   pd.to_datetime(parts[0], format="%Y%m%d"),
                            "Mkt-RF": float(parts[1]) / 100.0,
                            "SMB":    float(parts[2]) / 100.0,
                            "HML":    float(parts[3]) / 100.0,
                            "RF":     float(parts[4]) / 100.0,
                        })
                    except (ValueError, IndexError):
                        pass
        df = (
            pd.DataFrame(rows)
            .set_index("date")
            .sort_index()
        )
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        log.info(
            "      \u2192 %d rows  (%.1fs)  Mkt-RF \u03bc=%+.4f  RF \u03bc=%.5f",
            len(df), time.perf_counter() - t0, df["Mkt-RF"].mean(), df["RF"].mean(),
        )
        _save_supp("ff_factors", start, end, df)
        return df
    except Exception as exc:
        log.warning("Ken French factors unavailable (%s) \u2014 attribution will use SPX proxy", exc)
        return pd.DataFrame(columns=["Mkt-RF", "SMB", "HML", "RF"])


def fetch_spx_bid_ask_halfspread(
    db, start: str, end: str, atm_delta_range: tuple[int, int] = (40, 60),
) -> pd.Series:
    """Compute the SPX ATM option bid-ask half-spread as a fraction of the mid-price.

    Source: OptionMetrics opprcd{YYYY} year-partitioned tables.
    Filters to near-ATM calls (|delta - 0.5| < 0.1) expiring in 20\u201340 days.

    Returns daily median half-spread (fraction, e.g. 0.0025 = 25 bps).

    When all opprcd year-tables are inaccessible (WRDS subscription limitation),
    falls back to a literature-based estimate of 25 bps.  This is derived from
    Muravyev & Pearson (Review of Financial Studies, 2020) and Broadie, Chernov &
    Johannes (Journal of Finance, 2009), who document SPX ATM option effective
    half-spreads of 20\u201350 bps of option mid-price for short-dated contracts.
    25 bps is the conservative midpoint of this empirical range.
    """
    # ── Try local CSV first (covers 2003–2024, no WRDS subscription needed) ────────
    csv_result = _process_raw_opprcd_csv(start, end)
    if csv_result is not None and len(csv_result) > 0:
        return csv_result

    # ── Fall back to WRDS query ─────────────────────────────────────
    start_yr, end_yr = int(start[:4]), int(end[:4])
    years = list(range(start_yr, end_yr + 1))
    log.info("Fetching SPX ATM bid-ask half-spreads (optionm.opprcd)  %s \u2192 %s \u2026", start, end)
    t0 = time.perf_counter()

    frames: list[pd.DataFrame] = []
    for yr in years:
        sql = (
            f"SELECT date, best_bid, best_offer "
            f"FROM optionm.opprcd{yr} "
            f"WHERE secid = {_SPX_SECID} "
            f"  AND date BETWEEN '{start}' AND '{end}' "
            f"  AND cp_flag = 'C' "
            f"  AND best_bid > 0 AND best_offer > 0 "
            f"  AND days BETWEEN 20 AND 40 "
            f"  AND ABS(delta - 0.5) < 0.1"
        )
        try:
            frames.append(db.raw_sql(sql, date_cols=["date"]))
        except Exception as exc:
            if "does not exist" in str(exc):
                continue
            raise

    if not frames:
        # OptionMetrics opprcd requires a separate WRDS/IvyDB subscription.
        # When inaccessible, use a conservative literature-based estimate:
        # Muravyev & Pearson (RFS 2020) and Broadie, Chernov & Johannes (JF 2009)
        # document SPX ATM effective half-spreads of 20\u201350 bps of option
        # mid-price for 20\u201340-day contracts; 25 bps is the conservative midpoint.
        _LITERATURE_FALLBACK = 25e-4
        log.warning(
            "opprcd tables inaccessible (WRDS subscription). "
            "Using literature-based fallback of %.0f bps "
            "(Muravyev & Pearson, RFS 2020; Broadie, Chernov & Johannes, JF 2009: "
            "SPX ATM effective half-spreads 20\u201350 bps of mid-price). "
            "Gating results with this estimate should be interpreted cautiously.",
            _LITERATURE_FALLBACK * 1e4,
        )
        idx = pd.date_range(start, end, freq="B", name="date")
        return pd.Series(_LITERATURE_FALLBACK, index=idx, name="half_spread")

    df = pd.concat(frames, ignore_index=True)
    df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0
    df["half_spread"] = (df["best_offer"] - df["best_bid"]) / 2.0 / df["mid"]
    result = df.groupby("date")["half_spread"].median().rename("half_spread")
    log.info(
        "      \u2192 %d daily obs  (%.1fs)  median half-spread=%.5f (%.1f bps)",
        len(result), time.perf_counter() - t0, result.median(), result.median() * 1e4,
    )
    return result


def _fetch_fred_dgs10(start: str, end: str) -> pd.Series | None:
    """Fetch 10-year Treasury CMT yield (DGS10) from the FRED public CSV endpoint.

    No API key required.  Values are in percentage points (4.20 = 4.20%);
    converted to decimal before returning.  Returns None on any network error.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read().decode()
        df = pd.read_csv(
            io.StringIO(raw),
            parse_dates=["observation_date"],
        ).rename(columns={"observation_date": "date", "DGS10": "yield_10y"})
        df["yield_10y"] = pd.to_numeric(df["yield_10y"], errors="coerce")
        df = df.dropna(subset=["yield_10y"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        result = df.set_index("date")["yield_10y"] / 100.0
        result = result.rename("yield_10y")
        log.info(
            "[4/4] Treasury 10Y (FRED DGS10)  %s → %s  → %d rows  mean=%.4f  (%.1fs)",
            start, end, len(result), result.mean(), time.perf_counter() - t0,
        )
        return result
    except Exception as exc:
        log.warning("FRED DGS10 fetch failed (%s) — falling back to WRDS zerocd", exc)
        return None


def fetch_treasury_10y(db, start: str, end: str) -> pd.Series:
    """10-year Treasury yield.

    Primary source: FRED DGS10 public CSV (no auth required; daily CMT yield
    in decimal form, e.g. 0.042 = 4.2%).

    Fallback source: OptionMetrics zerocd at days=365 (continuously compounded
    1-year zero-coupon rate — same order of magnitude as DGS10 and available
    for the full OptionMetrics subscription window).
    """
    result = _fetch_fred_dgs10(start, end)
    if result is not None and len(result) > 10:
        return result

    # ── WRDS fallback ─────────────────────────────────────────────────────
    log.info("[4/4] Fetching 1Y zero rate (optionm.zerocd)  %s → %s …", start, end)
    t0 = time.perf_counter()
    sql = f"""
        SELECT date, rate / 100.0 AS yield_10y
        FROM optionm.zerocd
        WHERE days = 365
          AND date BETWEEN '{start}' AND '{end}'
        ORDER BY date
    """
    df = db.raw_sql(sql, date_cols=["date"])
    df = df.dropna(subset=["yield_10y"])
    result = df.set_index("date")["yield_10y"].rename("yield_10y")
    log.info(
        "      → %d rows  (%.1fs)  mean yield=%.4f",
        len(result), time.perf_counter() - t0, result.mean(),
    )
    return result


# ---------------------------------------------------------------------------
# Local CSV loaders — OptionMetrics flat-file exports (no WRDS connection needed)
# ---------------------------------------------------------------------------

_SPX_ZEROCD_CSV = Path(__file__).parents[2] / "SPX_zerocd.csv"
_SPX_OPVOLD_CSV = Path(__file__).parents[2] / "SPX_opvold.csv"


def fetch_zerocd_local(start: str, end: str) -> pd.DataFrame:
    """Compute daily 30-day short rate and yield curve slope from the local
    OptionMetrics zero-coupon yield CSV (SPX_zerocd.csv).

    The CSV has columns: date, days, rate (rate in % p.a., continuously
    compounded).  For each date we linearly interpolate to exactly days=30
    and days=365, then divide by 100 to convert to decimal form.

    Returns
    -------
    pd.DataFrame
        Index: date.  Columns: short_rate (30-day zero rate, decimal),
        curve_slope (365-day minus 30-day zero rate, decimal).
    """
    safe_s, safe_e = start.replace("-", ""), end.replace("-", "")
    cache = _CACHE_DIR / f"zerocd_rates_{safe_s}_{safe_e}.parquet"

    if cache.exists():
        df = pd.read_parquet(cache)
        log.info(
            "Zero curve cache hit: %s  (%d obs  short_rate μ=%.4f  slope μ=%.4f)",
            cache.name, len(df), df["short_rate"].mean(), df["curve_slope"].mean(),
        )
        return df

    if not _SPX_ZEROCD_CSV.exists():
        log.warning("SPX_zerocd.csv not found — returning empty DataFrame")
        return pd.DataFrame(columns=["short_rate", "curve_slope"])

    log.info(
        "Processing SPX_zerocd.csv → short_rate + curve_slope  %s–%s …", start, end
    )
    t0 = time.perf_counter()

    import numpy as np

    raw = pd.read_csv(_SPX_ZEROCD_CSV, parse_dates=["date"])
    raw = raw[
        (raw["date"] >= pd.Timestamp(start)) & (raw["date"] <= pd.Timestamp(end))
    ]
    raw = raw.dropna(subset=["rate", "days"]).sort_values(["date", "days"])

    rows: list[dict] = []
    for dt, grp in raw.groupby("date"):
        d = grp["days"].values.astype(float)
        r = grp["rate"].values.astype(float)
        if len(d) < 2:
            continue
        r30  = float(np.interp(30.0,  d, r)) / 100.0
        r365 = float(np.interp(365.0, d, r)) / 100.0
        rows.append({"date": dt, "short_rate": r30, "curve_slope": r365 - r30})

    if not rows:
        log.warning("No usable zero curve rows found in SPX_zerocd.csv")
        return pd.DataFrame(columns=["short_rate", "curve_slope"])

    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.to_datetime(result.index)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache)
    log.info(
        "      → %d rows  (%.1fs)  short_rate μ=%.4f  curve_slope μ=%.4f",
        len(result), time.perf_counter() - t0,
        result["short_rate"].mean(), result["curve_slope"].mean(),
    )
    return result


def fetch_opvold_local(start: str, end: str) -> pd.Series:
    """Compute daily log put/call volume ratio from the local OptionMetrics
    option volume CSV (SPX_opvold.csv).

    The CSV has columns: date, cp_flag (C/P/NaN for total), volume, ...
    We filter to cp_flag in ('C', 'P'), sum daily volume per side, and
    compute log(put_volume / call_volume).

    Returns
    -------
    pd.Series
        Index: date.  Name: 'log_pcr'.  Positive = more put volume.
    """
    safe_s, safe_e = start.replace("-", ""), end.replace("-", "")
    cache = _CACHE_DIR / f"spx_log_pcr_{safe_s}_{safe_e}.parquet"

    if cache.exists():
        df = pd.read_parquet(cache)
        log.info(
            "Log PCR cache hit: %s  (%d obs  μ=%.4f)",
            cache.name, len(df), df["log_pcr"].mean(),
        )
        return df["log_pcr"]

    if not _SPX_OPVOLD_CSV.exists():
        log.warning("SPX_opvold.csv not found — returning empty Series")
        return pd.Series(dtype=float, name="log_pcr")

    log.info("Processing SPX_opvold.csv → log_pcr  %s–%s …", start, end)
    t0 = time.perf_counter()

    import numpy as np

    raw = pd.read_csv(_SPX_OPVOLD_CSV, parse_dates=["date"])
    raw = raw[
        (raw["date"] >= pd.Timestamp(start)) & (raw["date"] <= pd.Timestamp(end))
    ]
    raw = raw[raw["cp_flag"].isin(["C", "P"])]
    raw = raw.dropna(subset=["volume"])
    raw = raw[raw["volume"] > 0]

    agg = (
        raw.groupby(["date", "cp_flag"])["volume"]
        .sum()
        .unstack("cp_flag")
    )
    for col in ("C", "P"):
        if col not in agg.columns:
            agg[col] = float("nan")

    agg = agg.dropna(subset=["C", "P"])
    agg = agg[(agg["C"] > 0) & (agg["P"] > 0)]
    log_pcr = np.log(agg["P"] / agg["C"]).rename("log_pcr")

    result_df = log_pcr.to_frame()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(cache)
    log.info(
        "      → %d rows  (%.1fs)  log_pcr μ=%.4f",
        len(log_pcr), time.perf_counter() - t0, log_pcr.mean(),
    )
    return log_pcr


def fetch_cboe_multivol_local(start: str, end: str) -> pd.DataFrame:
    """Load VXN (NASDAQ-100) and VXD (DJIA) volatility indices from the local
    CBOE multi-index CSV export (data/cboe_vix_vox_2003_2024.csv).

    Returns
    -------
    pd.DataFrame
        Index: date.  Columns: ['vxn', 'vxd'].  Values in decimal form
        (e.g. 0.20 = 20 VXN).  Sparse nulls are forward-filled.

    Economic rationale
    ------------------
    - ``vxn_vix_spread = log(vxn / vix)`` — tech risk premium vs broad market.
      Positive when NASDAQ options are more expensive than SPX options; signals
      elevated sector-specific uncertainty beyond systematic risk.
    - ``vxd_vix_spread = log(vxd / vix)`` — DJIA blue-chip concentration.
      Usually near zero; spikes during idiosyncratic large-cap stress episodes.
    """
    cached = _load_supp("cboe_multivol", start, end)
    if cached is not None:
        return cached

    if not _CBOE_MULTIVOL_CSV.exists():
        log.warning(
            "CBOE multi-vol CSV not found (%s) — returning empty DataFrame",
            _CBOE_MULTIVOL_CSV.name,
        )
        return pd.DataFrame(columns=["vxn", "vxd"])

    log.info("Reading CBOE multi-vol CSV → VXN, VXD  %s–%s …", start, end)
    raw = pd.read_csv(
        _CBOE_MULTIVOL_CSV,
        usecols=["Date", "vxn", "vxd"],
        parse_dates=["Date"],
    )
    raw = raw.rename(columns={"Date": "date"}).set_index("date")
    raw = raw.sort_index()

    # CBOE publishes as percentage points (20.0 = 20%); convert to decimal
    raw["vxn"] = pd.to_numeric(raw["vxn"], errors="coerce") / 100.0
    raw["vxd"] = pd.to_numeric(raw["vxd"], errors="coerce") / 100.0

    # Forward-fill the sparse nulls (~23 for VXN, ~24 for VXD)
    raw = raw.ffill()

    result = raw.loc[start:end]
    _save_supp("cboe_multivol", start, end, result)
    log.info(
        "      → %d rows  vxn μ=%.4f  vxd μ=%.4f",
        len(result), result["vxn"].mean(), result["vxd"].mean(),
    )
    return result


def fetch_optionm_rv_dispersion_local(
    start: str,
    end: str,
    tenor: int = 30,
) -> pd.Series:
    """Compute cross-sectional realized-vol dispersion from the OptionMetrics
    constituent historical-vol CSV (data/sp500_historical_vol_2003_2024_all_constituents.csv).

    For each date, computes the coefficient of variation (std / mean) of the
    ``tenor``-day historical volatility across all S&P 500 constituent secids
    present on that date.

    Parameters
    ----------
    start, end:
        Date range.
    tenor:
        Calendar days for the historical-vol tenor to use.  Default 30 matches
        the ``rv_21`` feature already in the state vector.

    Returns
    -------
    pd.Series
        Index: date.  Name: 'rv_dispersion'.

    Economic rationale
    ------------------
    Complements the existing ``iv_dispersion`` (cross-sectional IV CV) with
    a realized-vol counterpart.  The difference ``iv_dispersion - rv_dispersion``
    approximates a cross-sectional dispersion variance risk premium — the market
    excess price for dispersion insurance beyond realised dispersion.
    """
    safe_s, safe_e = start.replace("-", ""), end.replace("-", "")
    cache = _CACHE_DIR / f"optionm_rv_dispersion_{safe_s}_{safe_e}.parquet"

    if cache.exists():
        df = pd.read_parquet(cache)
        s = df["rv_dispersion"]
        log.info(
            "RV dispersion cache hit: %s  (%d obs  μ=%.4f)",
            cache.name, len(s), float(s.mean()),
        )
        return s

    if not _OPTIONM_HV_CSV.exists():
        log.warning(
            "OptionMetrics HV CSV not found (%s) — returning empty Series",
            _OPTIONM_HV_CSV.name,
        )
        return pd.Series(dtype=float, name="rv_dispersion")

    log.info(
        "Computing constituent RV dispersion (tenor=%dd)  %s–%s …", tenor, start, end
    )
    t0 = time.perf_counter()

    chunks = []
    for chunk in pd.read_csv(
        _OPTIONM_HV_CSV,
        usecols=["secid", "date", "days", "volatility"],
        chunksize=500_000,
        low_memory=False,
    ):
        sub = chunk[chunk["days"] == tenor].copy()
        sub = sub[
            (sub["date"] >= start) & (sub["date"] <= end)
        ]
        sub["volatility"] = pd.to_numeric(sub["volatility"], errors="coerce")
        sub = sub.dropna(subset=["volatility"])
        if not sub.empty:
            chunks.append(sub[["secid", "date", "volatility"]])

    if not chunks:
        log.warning("No %d-day HV rows found in %s", tenor, _OPTIONM_HV_CSV.name)
        return pd.Series(dtype=float, name="rv_dispersion")

    long = pd.concat(chunks, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])

    # Per-date cross-sectional CV
    grp = long.groupby("date")["volatility"]
    rv_disp = (grp.std() / grp.mean()).rename("rv_dispersion")
    rv_disp = rv_disp.loc[start:end].sort_index()

    result_df = rv_disp.to_frame()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(cache)
    log.info(
        "      → %d rows  (%.1fs)  rv_dispersion μ=%.4f",
        len(rv_disp), time.perf_counter() - t0, float(rv_disp.mean()),
    )
    return rv_disp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_wrds_data(
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch a 2-year panel of SPX data from WRDS, with local parquet caching.

    Parameters
    ----------
    start : str, optional
        ISO date string, e.g. ``"2022-01-01"``.  Defaults to 730 days before
        *end*.
    end : str, optional
        ISO date string.  Defaults to today.
    use_cache : bool
        When True (default), return data from ``data/wrds_cache/`` if available
        rather than making a live WRDS query.  Set to False to always fetch live.
    force_refresh : bool
        When True, fetch fresh data from WRDS even if a cache file exists, then
        overwrite the cache.  Implies ``use_cache=True`` for the write step.

    Returns
    -------
    dict with keys:
        ``"prices"``       — pd.Series  (SPX daily closing level)
        ``"option_panel"`` — pd.DataFrame (date, iv_30, iv_91, skew_25d)
        ``"vix"``          — pd.Series  (VIX in decimal form)
        ``"treasury_10y"`` — pd.Series  (10-year yield in decimal form)

    Cache
    -----
    Parquet files are written to ``data/wrds_cache/`` in the project root.
    Each file is named ``<key>_<YYYYMMDD>_<YYYYMMDD>.parquet`` where the dates
    are the requested *start* and *end*.  Delete the files manually to force a
    fresh pull, or pass ``force_refresh=True``.
    """
    _configure_logging()

    if end is None:
        end = date.today().isoformat()
    if start is None:
        start = (date.today() - timedelta(days=730)).isoformat()

    # ── Cache read ────────────────────────────────────────────────────────────
    if use_cache and not force_refresh:
        cached = _load_cache(start, end)
        if cached is not None:
            log.info("=" * 60)
            log.info("Using cached data  |  window: %s → %s", start, end)
            log.info(
                "Summary  |  prices=%d rows  panel=%d rows  vix=%d rows  treasury=%d rows",
                len(cached["prices"]),
                len(cached["option_panel"]),
                len(cached["vix"]),
                len(cached["treasury_10y"]),
            )
            log.info("=" * 60)
            return cached

    # ── Live WRDS fetch ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("WRDS data pull  |  window: %s → %s", start, end)
    log.info("=" * 60)
    t_total = time.perf_counter()

    db = _connect()
    try:
        result = {
            "prices":       fetch_spx_prices(db, start, end),
            "option_panel": fetch_option_panel(db, start, end),
            "vix":          fetch_vix(db, start, end),
            "treasury_10y": fetch_treasury_10y(db, start, end),
        }
    finally:
        db.close()
        log.info("WRDS connection closed.")

    elapsed = time.perf_counter() - t_total
    log.info("=" * 60)
    log.info("All data fetched in %.1fs", elapsed)
    log.info(
        "Summary  |  prices=%d rows  panel=%d rows  vix=%d rows  treasury=%d rows",
        len(result["prices"]),
        len(result["option_panel"]),
        len(result["vix"]),
        len(result["treasury_10y"]),
    )
    log.info("=" * 60)

    # ── Cache write ───────────────────────────────────────────────────────────
    if use_cache or force_refresh:
        try:
            _save_cache(result, start, end)
        except Exception as exc:
            log.warning("Cache write failed (non-fatal): %s", exc)

    return result
