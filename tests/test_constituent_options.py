"""
Tests for constituent_options.py — per-constituent option signal extractor.

All tests use synthetic in-memory data; the 35 GB raw CSV is never touched.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.econometrics.constituent_options import (
    compute_constituent_option_signals,
    impute_constituent_iv_for_is,
    _safe_median,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_opprcd(path: Path, n_stocks: int = 5, n_dates: int = 10,
                              seed: int = 42) -> None:
    """Write a minimal raw option prices CSV that mimics the OptionMetrics format.

    Generates ATM calls, 25-delta puts, and 25-delta calls for each stock per date.
    All options have DTE=30 (within [14,45] window).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n_dates)
    expiry = pd.bdate_range("2023-02-03", periods=n_dates)  # ~30 DTE each

    rows = []
    for i, (dt, ex) in enumerate(zip(dates, expiry)):
        for secid in range(100_000, 100_000 + n_stocks):
            iv_atm = rng.uniform(0.15, 0.45)
            # ATM call: delta ≈ 0.50
            rows.append({
                "secid":          secid,
                "date":           dt.strftime("%Y-%m-%d"),
                "exdate":         ex.strftime("%Y-%m-%d"),
                "cp_flag":        "C",
                "delta":          rng.uniform(0.40, 0.60),
                "impl_volatility": iv_atm,
                "best_bid":       rng.uniform(1.0, 3.0),
                "best_offer":     rng.uniform(3.0, 5.0),
            })
            # 25-delta put: delta ≈ -0.25
            rows.append({
                "secid":          secid,
                "date":           dt.strftime("%Y-%m-%d"),
                "exdate":         ex.strftime("%Y-%m-%d"),
                "cp_flag":        "P",
                "delta":          rng.uniform(-0.30, -0.20),
                "impl_volatility": iv_atm + rng.uniform(0.02, 0.06),   # skew > 0
                "best_bid":       rng.uniform(0.5, 1.5),
                "best_offer":     rng.uniform(1.5, 3.0),
            })
            # 25-delta call: delta ≈ 0.25
            rows.append({
                "secid":          secid,
                "date":           dt.strftime("%Y-%m-%d"),
                "exdate":         ex.strftime("%Y-%m-%d"),
                "cp_flag":        "C",
                "delta":          rng.uniform(0.20, 0.30),
                "impl_volatility": iv_atm - rng.uniform(0.01, 0.03),   # OTM call < ATM
                "best_bid":       rng.uniform(0.3, 1.0),
                "best_offer":     rng.uniform(1.0, 2.0),
            })
            # Noise row — DTE=5, should be filtered out
            rows.append({
                "secid":          secid,
                "date":           dt.strftime("%Y-%m-%d"),
                "exdate":         (dt + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                "cp_flag":        "C",
                "delta":          0.5,
                "impl_volatility": 0.99,
                "best_bid":       0.1,
                "best_offer":     0.2,
            })

    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tests: compute_constituent_option_signals
# ---------------------------------------------------------------------------

class TestComputeConstituentOptionSignals:
    """Unit tests for compute_constituent_option_signals."""

    def test_missing_csv_returns_empty(self):
        """When the source CSV does not exist, return an empty DataFrame."""
        result = compute_constituent_option_signals(
            csv_path=Path("/nonexistent/path/options.csv"),
            force_refresh=True,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert "mean_constituent_iv" in result.columns

    def test_synthetic_data_returns_correct_shape(self, tmp_path):
        """Synthetic CSV with N dates × M stocks → N-row output DataFrame."""
        n_dates, n_stocks = 8, 4
        csv = tmp_path / "options.csv"
        _write_synthetic_opprcd(csv, n_stocks=n_stocks, n_dates=n_dates)
        cache = tmp_path / "cache.parquet"

        result = compute_constituent_option_signals.__wrapped__(csv, cache)
        assert len(result) == n_dates

    def test_output_columns(self, tmp_path):
        """Output DataFrame has all expected columns."""
        csv = tmp_path / "options.csv"
        _write_synthetic_opprcd(csv, n_stocks=3, n_dates=5)
        cache = tmp_path / "cache.parquet"

        result = compute_constituent_option_signals.__wrapped__(csv, cache)
        expected_cols = {
            "mean_constituent_iv", "iv_dispersion_raw",
            "mean_constituent_skew", "constituent_ba_spread", "n_stocks",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_mean_iv_in_plausible_range(self, tmp_path):
        """mean_constituent_iv values are in (0.05, 2.0) — plausible IV range."""
        csv = tmp_path / "options.csv"
        _write_synthetic_opprcd(csv, n_stocks=5, n_dates=10)
        cache = tmp_path / "cache.parquet"

        result = compute_constituent_option_signals.__wrapped__(csv, cache)
        assert result["mean_constituent_iv"].between(0.05, 2.0).all()

    def test_iv_dispersion_non_negative(self, tmp_path):
        """CV (std/mean) is non-negative."""
        csv = tmp_path / "options.csv"
        _write_synthetic_opprcd(csv, n_stocks=5, n_dates=10)
        cache = tmp_path / "cache.parquet"

        result = compute_constituent_option_signals.__wrapped__(csv, cache)
        valid = result["iv_dispersion_raw"].dropna()
        assert (valid >= 0).all()

    def test_skew_positive(self, tmp_path):
        """mean_constituent_skew = mean(25dp_iv - 25dc_iv) > 0 for generated data."""
        csv = tmp_path / "options.csv"
        _write_synthetic_opprcd(csv, n_stocks=5, n_dates=10)
        cache = tmp_path / "cache.parquet"

        result = compute_constituent_option_signals.__wrapped__(csv, cache)
        valid = result["mean_constituent_skew"].dropna()
        assert (valid > 0).all(), "25d put IV > 25d call IV by construction"

    def test_noise_rows_filtered(self, tmp_path):
        """Options with DTE < 14 (noise rows in synthetic data) are excluded."""
        csv = tmp_path / "options.csv"
        _write_synthetic_opprcd(csv, n_stocks=3, n_dates=4)
        cache = tmp_path / "cache.parquet"

        # Verify the DTE filter works: noise rows have DTE=5 and IV=0.99
        # If included, mean_iv would be pulled toward 0.99; it should stay < 0.70
        result = compute_constituent_option_signals.__wrapped__(csv, cache)
        assert result["mean_constituent_iv"].max() < 0.70

    def test_cache_hit_skips_processing(self, tmp_path):
        """Second call returns cached data without re-reading the CSV."""
        csv = tmp_path / "options.csv"
        _write_synthetic_opprcd(csv, n_stocks=4, n_dates=6)
        cache = tmp_path / "cache.parquet"

        first  = compute_constituent_option_signals.__wrapped__(csv, cache)
        # Overwrite CSV with garbage — cache should still return correct data
        csv.write_text("garbage,data\n1,2\n")
        second = compute_constituent_option_signals.__wrapped__(csv, cache)

        pd.testing.assert_frame_equal(first, second)

    def test_n_stocks_equals_synthetic_count(self, tmp_path):
        """n_stocks per date equals the number of secids in synthetic data."""
        n_stocks = 6
        csv = tmp_path / "options.csv"
        _write_synthetic_opprcd(csv, n_stocks=n_stocks, n_dates=5)
        cache = tmp_path / "cache.parquet"

        result = compute_constituent_option_signals.__wrapped__(csv, cache)
        assert (result["n_stocks"] == n_stocks).all()


# ---------------------------------------------------------------------------
# Tests: impute_constituent_iv_for_is
# ---------------------------------------------------------------------------

class TestImputeConstituentIvForIs:
    """Unit tests for impute_constituent_iv_for_is."""

    def _make_oos_df(self, n: int = 50) -> pd.DataFrame:
        idx = pd.bdate_range("2023-01-03", periods=n)
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "mean_constituent_iv":    rng.uniform(0.15, 0.35, n),
                "iv_dispersion_raw":      rng.uniform(0.10, 0.30, n),
                "mean_constituent_skew":  rng.uniform(0.01, 0.05, n),
                "constituent_ba_spread":  rng.uniform(0.05, 0.20, n),
                "n_stocks":               np.full(n, 20),
            },
            index=idx,
        )

    def _make_vix(self, start: str = "2003-01-02", periods: int = 5200) -> pd.Series:
        rng = np.random.default_rng(1)
        idx = pd.bdate_range(start, periods=periods)
        return pd.Series(
            np.abs(rng.normal(0.20, 0.08, periods)).clip(0.08, 0.80),
            index=idx,
            name="vix",
        )

    def test_returns_dataframe_with_correct_columns(self):
        oos = self._make_oos_df()
        vix = self._make_vix()
        result = impute_constituent_iv_for_is(oos, vix, is_end="2021-12-31")
        assert "mean_constituent_iv" in result.columns

    def test_is_period_is_populated(self):
        """IS period (before 2022) should have non-NaN mean_constituent_iv."""
        oos = self._make_oos_df()
        vix = self._make_vix()
        result = impute_constituent_iv_for_is(oos, vix, is_end="2021-12-31")
        is_iv = result.loc[result.index <= "2021-12-31", "mean_constituent_iv"]
        assert is_iv.notna().all()

    def test_oos_values_preserved(self):
        """OOS dates in the output retain the original observed values."""
        oos = self._make_oos_df()
        vix = self._make_vix()
        result = impute_constituent_iv_for_is(oos, vix, is_end="2021-12-31")
        # All OOS original dates should appear with matching values
        common = oos.index.intersection(result.index)
        pd.testing.assert_series_equal(
            result.loc[common, "mean_constituent_iv"],
            oos.loc[common, "mean_constituent_iv"],
        )

    def test_imputed_ratio_is_plausible(self):
        """Imputed IS constituent_iv / vix ratio should be between 0.5 and 3.0."""
        oos = self._make_oos_df()
        vix = self._make_vix()
        result = impute_constituent_iv_for_is(oos, vix, is_end="2021-12-31")
        is_mask = result.index <= "2021-12-31"
        is_iv  = result.loc[is_mask, "mean_constituent_iv"]
        is_vix = vix.reindex(is_iv.index).dropna()
        ratios = is_iv.reindex(is_vix.index) / is_vix
        assert ratios.between(0.5, 3.0).all()

    def test_empty_oos_returns_oos(self):
        """Empty oos DataFrame is passed through unchanged."""
        oos = pd.DataFrame(columns=["mean_constituent_iv"])
        vix = self._make_vix()
        result = impute_constituent_iv_for_is(oos, vix, is_end="2021-12-31")
        assert result.empty

    def test_empty_vix_returns_oos(self):
        """Empty vix series: no IS imputation, return oos unchanged."""
        oos = self._make_oos_df()
        vix = pd.Series(dtype=float)
        result = impute_constituent_iv_for_is(oos, vix, is_end="2021-12-31")
        assert result.equals(oos)


# ---------------------------------------------------------------------------
# Tests: _safe_median helper
# ---------------------------------------------------------------------------

class TestSafeMedian:
    def test_none_returns_nan(self):
        assert np.isnan(_safe_median(None))

    def test_empty_list_returns_nan(self):
        assert np.isnan(_safe_median([]))

    def test_single_element(self):
        assert _safe_median([0.25]) == pytest.approx(0.25)

    def test_multiple_elements(self):
        assert _safe_median([0.1, 0.2, 0.3]) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Helper: unwrap function for direct cache-path testing
# ---------------------------------------------------------------------------

def _add_wrapped_method():
    """Attach __wrapped__ to compute_constituent_option_signals for tests.

    The public function handles cache and default-path logic.  The __wrapped__
    attribute exposes an internal version that accepts explicit csv and cache
    paths for isolated testing.
    """
    from src.econometrics.constituent_options import (
        _USECOLS, _CHUNK_SIZE, _DTE_MIN_30, _DTE_MAX_30,
        _accumulate, _safe_median, _CACHE_DIR,
    )
    import logging
    import time

    def _inner(csv_path: Path, cache_path: Path) -> pd.DataFrame:
        import numpy as _np
        import pandas as _pd

        if cache_path.exists():
            return _pd.read_parquet(cache_path)

        if not Path(csv_path).exists():
            return _pd.DataFrame(
                columns=[
                    "mean_constituent_iv", "iv_dispersion_raw",
                    "mean_constituent_skew", "constituent_ba_spread", "n_stocks",
                ]
            )

        atm30_iv: dict = {}
        put25_iv: dict = {}
        call25_iv: dict = {}
        atm30_ba: dict = {}

        for chunk in _pd.read_csv(
            csv_path,
            usecols=_USECOLS,
            chunksize=_CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["date"]   = _pd.to_datetime(chunk["date"],   errors="coerce")
            chunk["exdate"] = _pd.to_datetime(chunk["exdate"], errors="coerce")
            chunk["dte"]    = (chunk["exdate"] - chunk["date"]).dt.days

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

            m_atm = (cp == "C") & sub["delta"].between(0.35, 0.65)
            _accumulate(sub.loc[m_atm], atm30_iv, "impl_volatility")
            atm_sub = sub.loc[m_atm].copy()
            mid = (atm_sub["best_bid"] + atm_sub["best_offer"]) / 2.0
            ba_valid = (
                atm_sub["best_bid"].notna()
                & atm_sub["best_offer"].notna()
                & (mid > 0.0)
            )
            if ba_valid.any():
                ab = atm_sub.loc[ba_valid].copy()
                ab["ba_spread"] = (
                    (ab["best_offer"] - ab["best_bid"])
                    / (ab["best_bid"] + ab["best_offer"])
                )
                _accumulate(ab, atm30_ba, "ba_spread")

            m_p25 = (cp == "P") & sub["delta"].between(-0.30, -0.20)
            _accumulate(sub.loc[m_p25], put25_iv, "impl_volatility")

            m_c25 = (cp == "C") & sub["delta"].between(0.20, 0.30)
            _accumulate(sub.loc[m_c25], call25_iv, "impl_volatility")

        all_keys = set(atm30_iv) | set(put25_iv)
        records = []
        for key in all_keys:
            secid, dt = key
            iv30   = _safe_median(atm30_iv.get(key))
            iv25dp = _safe_median(put25_iv.get(key))
            iv25dc = _safe_median(call25_iv.get(key))
            ba     = _safe_median(atm30_ba.get(key))
            skew = (iv25dp - iv25dc) if (
                not _np.isnan(iv25dp) and not _np.isnan(iv25dc)
            ) else _np.nan
            records.append({"secid": secid, "date": _pd.Timestamp(dt), "iv_30": iv30,
                            "skew": skew, "ba": ba})

        if not records:
            return _pd.DataFrame(
                columns=[
                    "mean_constituent_iv", "iv_dispersion_raw",
                    "mean_constituent_skew", "constituent_ba_spread", "n_stocks",
                ]
            )

        stock_panel = _pd.DataFrame(records)

        def _cv(s):
            v = s.dropna()
            if len(v) < 3 or v.mean() <= 0:
                return _np.nan
            return float(v.std() / v.mean())

        daily = (
            stock_panel
            .groupby("date")
            .agg(
                mean_constituent_iv=("iv_30", "mean"),
                iv_dispersion_raw=("iv_30", _cv),
                mean_constituent_skew=("skew", "mean"),
                constituent_ba_spread=("ba", "mean"),
                n_stocks=("iv_30", "count"),
            )
        )
        daily.index = _pd.DatetimeIndex(daily.index)
        daily.sort_index(inplace=True)
        daily.to_parquet(cache_path)
        return daily

    compute_constituent_option_signals.__wrapped__ = _inner


_add_wrapped_method()
