"""
Phase I — Constituent Volatility Surface Dispersion Index.

Streams the large OptionMetrics constituent vsurfd CSVs (2003–2021 and
2022–2024) and computes a daily cross-sectional implied-volatility dispersion
index (coefficient of variation of 30-day ATM constituent IVs).  Results are
cached to a small parquet file; the large CSVs are only read once.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_CSVS = [
    Path(r"D:\Downloads\constituent_volsurfd(2003-2021).csv"),
    Path(r"D:\Downloads\constituent_volsurfd(2022-2024).csv"),
]
_CACHE_DIR   = Path(__file__).parents[2] / "data" / "wrds_cache"
_CACHE_FILE  = _CACHE_DIR / "constituent_dispersion_20030101_20241231.parquet"
_CHUNK_SIZE  = 2_000_000   # rows per read chunk; ~800 MB RAM per chunk


def compute_dispersion_index(
    csv_paths: list[Path | str] | Path | str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Stream-compute daily cross-sectional IV dispersion index.

    Filters each constituent vol surface CSV to the 30-day ATM call slice
    (days=30, delta=50, cp_flag='C'), then computes per-date:

        mean_iv       — cross-sectional mean of constituent ATM IVs (decimal)
        iv_dispersion — coefficient of variation: std / mean (dimensionless)
        n_stocks      — number of unique constituents contributing each day

    Results are cached; set force_refresh=True to re-process.

    Parameters
    ----------
    csv_paths:
        Path or list of paths to the constituent vsurfd CSVs.  Defaults to
        [2003-2021 CSV, 2022-2024 CSV].  Missing files are skipped.
    force_refresh:
        Re-process even if a cache file already exists.

    Returns
    -------
    pd.DataFrame
        Index: date (daily).  Columns: mean_iv, iv_dispersion, n_stocks.
        Empty DataFrame when no CSV is found.
    """
    if csv_paths is None:
        paths: list[Path] = _DEFAULT_CSVS
    elif isinstance(csv_paths, (str, Path)):
        paths = [Path(csv_paths)]
    else:
        paths = [Path(p) for p in csv_paths]

    available = [p for p in paths if p.exists()]

    if _CACHE_FILE.exists() and not force_refresh:
        df = pd.read_parquet(_CACHE_FILE)
        log.info(
            "Constituent dispersion cache hit: %s  (%d obs  iv_dispersion μ=%.4f)",
            _CACHE_FILE.name, len(df), df["iv_dispersion"].mean(),
        )
        return df

    if not available:
        log.warning(
            "No constituent vsurfd CSV found at: %s — returning empty DataFrame",
            [str(p) for p in paths],
        )
        return pd.DataFrame(columns=["mean_iv", "iv_dispersion", "n_stocks"])

    log.info(
        "Streaming %d constituent vsurfd CSV(s) → iv_dispersion  (one-time, ~5-30 min) …",
        len(available),
    )
    t0 = time.perf_counter()

    # Accumulator: date-string → list of impl_volatility values
    accum: dict[str, list[float]] = {}
    total_rows_read = 0
    total_rows_kept = 0

    for csv_path in available:
        rows_read = 0
        rows_kept = 0
        log.info("  Processing %s …", csv_path.name)

        for chunk in pd.read_csv(
            csv_path,
            usecols=["date", "days", "delta", "cp_flag", "impl_volatility"],
            dtype={"days": float, "delta": float, "impl_volatility": float, "cp_flag": str},
            chunksize=_CHUNK_SIZE,
        ):
            rows_read += len(chunk)

            # Filter to 30-day ATM calls with valid IV
            mask = (
                (chunk["days"] == 30)
                & (chunk["delta"] == 50)
                & (chunk["cp_flag"] == "C")
                & chunk["impl_volatility"].notna()
                & (chunk["impl_volatility"] > 0.0)
            )
            sub = chunk.loc[mask, ["date", "impl_volatility"]]
            rows_kept += len(sub)

            # Accumulate IV values per date — later CSV overwrites earlier for
            # any overlapping dates so the more recent source takes precedence.
            for dt, grp in sub.groupby("date"):
                key = str(dt)
                if key not in accum:
                    accum[key] = []
                accum[key].extend(grp["impl_volatility"].tolist())

            if rows_read % (10 * _CHUNK_SIZE) == 0:
                elapsed = time.perf_counter() - t0
                log.info(
                    "      %.0fM rows processed  |  %d dates so far  |  %.0fs elapsed",
                    (total_rows_read + rows_read) / 1e6, len(accum), elapsed,
                )

        log.info(
            "      %s done: %dM total rows  |  %d qualifying rows",
            csv_path.name, rows_read // 1_000_000, rows_kept,
        )
        total_rows_read += rows_read
        total_rows_kept += rows_kept

    log.info(
        "      All CSVs done: %dM total rows  |  %d qualifying rows  |  %d dates",
        total_rows_read // 1_000_000, total_rows_kept, len(accum),
    )

    # Aggregate per date
    records: list[dict] = []
    for dt_str, ivs in sorted(accum.items()):
        if len(ivs) < 5:   # require at least 5 constituents for a meaningful CV
            continue
        arr = np.array(ivs, dtype=np.float64)
        mean_iv = float(arr.mean())
        records.append({
            "date": pd.Timestamp(dt_str),
            "mean_iv": mean_iv,
            "iv_dispersion": float(arr.std() / mean_iv) if mean_iv > 0 else np.nan,
            "n_stocks": len(arr),
        })

    if not records:
        log.warning("No valid ATM constituent IV data found in %s", csv_path.name)
        return pd.DataFrame(columns=["mean_iv", "iv_dispersion", "n_stocks"])

    result = pd.DataFrame(records).set_index("date")
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(_CACHE_FILE)
    log.info(
        "      → %d days cached  (%.0f s)  iv_dispersion μ=%.4f  n_stocks median=%.0f",
        len(result), time.perf_counter() - t0,
        result["iv_dispersion"].mean(), result["n_stocks"].median(),
    )
    return result
