"""
Tests for Phase I — constituent_vsurfd.py (dispersion index computation).

All tests use synthetic in-memory data; no large CSV files are required.
The compute_dispersion_index function is tested via its internal helper
logic, and the caching/fallback paths are validated.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.econometrics.constituent_vsurfd import compute_dispersion_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_csv(path: Path, n_dates: int = 10, n_stocks: int = 30,
                          seed: int = 0) -> None:
    """Write a minimal constituent vsurfd CSV for testing.

    Columns required: date, days, delta, cp_flag, impl_volatility.
    Only rows with days=30, delta=50, cp_flag='C' are used by the function.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    rows = []
    for dt in dates:
        for _ in range(n_stocks):
            rows.append({
                "date":            dt.strftime("%Y-%m-%d"),
                "days":            30,
                "delta":           50,
                "cp_flag":         "C",
                "impl_volatility": rng.uniform(0.10, 0.50),
            })
        # Add some noise rows that should be filtered out
        for _ in range(5):
            rows.append({
                "date":            dt.strftime("%Y-%m-%d"),
                "days":            91,        # wrong days
                "delta":           25,
                "cp_flag":         "C",
                "impl_volatility": rng.uniform(0.10, 0.50),
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeDispersionIndex:
    def test_returns_empty_when_no_csv(self, tmp_path):
        """Non-existent CSV paths → empty DataFrame."""
        result = compute_dispersion_index(
            csv_paths=[tmp_path / "nonexistent.csv"],
            force_refresh=True,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_dataframe_with_required_columns(self, tmp_path):
        csv = tmp_path / "vsurfd_test.csv"
        _write_synthetic_csv(csv, n_dates=5, n_stocks=20)

        # Redirect cache to tmp_path to avoid polluting real cache
        import src.econometrics.constituent_vsurfd as mod
        old_cache = mod._CACHE_FILE
        mod._CACHE_FILE = tmp_path / "test_dispersion_cache.parquet"
        try:
            result = compute_dispersion_index(csv_paths=[csv], force_refresh=True)
        finally:
            mod._CACHE_FILE = old_cache

        assert set(result.columns) == {"mean_iv", "iv_dispersion", "n_stocks"}

    def test_iv_dispersion_is_coefficient_of_variation(self, tmp_path):
        """iv_dispersion = std / mean, always > 0 for heterogeneous IVs."""
        csv = tmp_path / "vsurfd_test.csv"
        _write_synthetic_csv(csv, n_dates=5, n_stocks=30)

        import src.econometrics.constituent_vsurfd as mod
        old_cache = mod._CACHE_FILE
        mod._CACHE_FILE = tmp_path / "test_dispersion_cache.parquet"
        try:
            result = compute_dispersion_index(csv_paths=[csv], force_refresh=True)
        finally:
            mod._CACHE_FILE = old_cache

        assert (result["iv_dispersion"] > 0).all()
        assert (result["mean_iv"] > 0).all()
        assert (result["n_stocks"] >= 5).all()

    def test_cache_is_used_on_second_call(self, tmp_path):
        """Second call must hit the cache (not re-process the CSV)."""
        csv = tmp_path / "vsurfd_test.csv"
        _write_synthetic_csv(csv, n_dates=5, n_stocks=20)

        import src.econometrics.constituent_vsurfd as mod
        old_cache = mod._CACHE_FILE
        cache_path = tmp_path / "test_dispersion_cache2.parquet"
        mod._CACHE_FILE = cache_path
        try:
            r1 = compute_dispersion_index(csv_paths=[csv], force_refresh=True)
            assert cache_path.exists()
            # Delete the CSV to prove the second call uses cache
            csv.unlink()
            r2 = compute_dispersion_index(csv_paths=[csv], force_refresh=False)
            pd.testing.assert_frame_equal(r1, r2)
        finally:
            mod._CACHE_FILE = old_cache

    def test_multiple_csv_paths_accumulated(self, tmp_path):
        """Two CSVs covering different dates → all dates in result."""
        csv1 = tmp_path / "vsurfd_2020.csv"
        csv2 = tmp_path / "vsurfd_2021.csv"
        _write_synthetic_csv(csv1, n_dates=5, n_stocks=20, seed=0)
        # Write csv2 with different dates (continue from 2021)
        rng = np.random.default_rng(99)
        dates2 = pd.bdate_range("2021-01-04", periods=5)
        rows = []
        for dt in dates2:
            for _ in range(25):
                rows.append({
                    "date":            dt.strftime("%Y-%m-%d"),
                    "days":            30,
                    "delta":           50,
                    "cp_flag":         "C",
                    "impl_volatility": rng.uniform(0.10, 0.50),
                })
        pd.DataFrame(rows).to_csv(csv2, index=False)

        import src.econometrics.constituent_vsurfd as mod
        old_cache = mod._CACHE_FILE
        mod._CACHE_FILE = tmp_path / "test_dispersion_combined.parquet"
        try:
            result = compute_dispersion_index(
                csv_paths=[csv1, csv2], force_refresh=True,
            )
        finally:
            mod._CACHE_FILE = old_cache

        # Combined result should have dates from both CSVs
        assert len(result) >= 8   # 5 + 5 dates (minus potential weekends)

    def test_filters_out_insufficient_constituents(self, tmp_path):
        """Dates with < 5 constituents must be excluded from the result."""
        csv = tmp_path / "sparse.csv"
        # Write only 3 stocks per date
        rows = [
            {"date": "2020-01-02", "days": 30, "delta": 50, "cp_flag": "C",
             "impl_volatility": 0.20},
            {"date": "2020-01-02", "days": 30, "delta": 50, "cp_flag": "C",
             "impl_volatility": 0.22},
            {"date": "2020-01-02", "days": 30, "delta": 50, "cp_flag": "C",
             "impl_volatility": 0.19},
        ]
        pd.DataFrame(rows).to_csv(csv, index=False)

        import src.econometrics.constituent_vsurfd as mod
        old_cache = mod._CACHE_FILE
        mod._CACHE_FILE = tmp_path / "test_dispersion_sparse.parquet"
        try:
            result = compute_dispersion_index(csv_paths=[csv], force_refresh=True)
        finally:
            mod._CACHE_FILE = old_cache

        # 3 constituents < 5 threshold → date should be excluded
        assert len(result) == 0

    def test_date_index_is_datetimeindex(self, tmp_path):
        csv = tmp_path / "vsurfd_test_idx.csv"
        _write_synthetic_csv(csv, n_dates=3, n_stocks=10)

        import src.econometrics.constituent_vsurfd as mod
        old_cache = mod._CACHE_FILE
        mod._CACHE_FILE = tmp_path / "test_dispersion_idx.parquet"
        try:
            result = compute_dispersion_index(csv_paths=[csv], force_refresh=True)
        finally:
            mod._CACHE_FILE = old_cache

        assert isinstance(result.index, pd.DatetimeIndex)
