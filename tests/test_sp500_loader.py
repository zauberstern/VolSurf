"""
Tests for src/econometrics/sp500_loader.py — PIT membership and survivorship bias.

All tests use synthetic data; no network access or WRDS connection is required
(the download_pit_membership function is mocked where needed).

Notes on apply_point_in_time_filter:
  - `returns` columns are permno (int), NOT ticker strings.
  - `universe` must have columns [permno, ticker, entry_date].
  - `pit_df` has columns [ticker, start_date, end_date] (from fja05680 format).
  - Function sets out-of-membership observations to NaN (not 0.0).
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.econometrics.sp500_loader import apply_point_in_time_filter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dates():
    return pd.bdate_range("2020-01-02", periods=50)


@pytest.fixture()
def synthetic_returns(dates) -> pd.DataFrame:
    """Wide returns DataFrame: index=date, columns=permno (int), values=float."""
    rng = np.random.default_rng(7)
    permnos = [10001, 10002, 10003]   # AAPL, MSFT, GOOG
    data = rng.normal(0.001, 0.01, (len(dates), len(permnos)))
    return pd.DataFrame(data, index=dates, columns=permnos)


@pytest.fixture()
def synthetic_universe() -> pd.DataFrame:
    """universe with [permno, ticker, entry_date] — all entries before test window."""
    return pd.DataFrame(
        {
            "permno":     [10001, 10002, 10003],
            "ticker":     ["AAPL", "MSFT", "GOOG"],
            "entry_date": [pd.Timestamp("2010-01-01")] * 3,
        }
    )


# ---------------------------------------------------------------------------
# apply_point_in_time_filter — entry date gating
# ---------------------------------------------------------------------------

class TestApplyPointInTimeFilterEntry:
    def test_nans_pre_entry_rows(
        self, synthetic_returns: pd.DataFrame, synthetic_universe: pd.DataFrame, dates
    ):
        """Returns before a permno's entry date must be NaN."""
        # MSFT (permno=10002) entry pushed to day 10
        universe = synthetic_universe.copy()
        universe.loc[universe["permno"] == 10002, "entry_date"] = dates[10]

        result = apply_point_in_time_filter(
            synthetic_returns.copy(), universe
        )
        # MSFT (10002) rows before dates[10] should all be NaN
        assert result[10002].iloc[:10].isna().all()
        # MSFT rows from dates[10] onward should retain original non-NaN values
        assert not result[10002].iloc[10:].isna().all()

    def test_no_effect_when_all_entry_dates_before_start(
        self, synthetic_returns: pd.DataFrame, synthetic_universe: pd.DataFrame
    ):
        """All entry dates before data start → returns unchanged."""
        result = apply_point_in_time_filter(
            synthetic_returns.copy(), synthetic_universe
        )
        pd.testing.assert_frame_equal(result, synthetic_returns)

    def test_missing_permno_in_returns_is_skipped(
        self, synthetic_universe: pd.DataFrame, dates
    ):
        """Permnos in universe but absent from returns columns → no error."""
        returns_small = pd.DataFrame(
            {10001: np.ones(len(dates)) * 0.001},
            index=dates,
        )
        result = apply_point_in_time_filter(returns_small, synthetic_universe)
        assert 10001 in result.columns
        assert result.shape == returns_small.shape


class TestApplyPointInTimeFilterExit:
    def test_nans_post_exit_rows_via_pit_df(
        self, synthetic_returns: pd.DataFrame, synthetic_universe: pd.DataFrame, dates
    ):
        """Returns after PIT exit date must be NaN when pit_df is provided."""
        # MSFT (permno=10002) exits on day 30
        exit_date = dates[29]
        pit_df = pd.DataFrame(
            {
                "ticker":     ["MSFT"],
                "start_date": [pd.Timestamp("2019-01-01")],
                "end_date":   [exit_date],
            }
        )
        result = apply_point_in_time_filter(
            synthetic_returns.copy(), synthetic_universe, pit_df=pit_df
        )
        # MSFT (10002) returns strictly after exit_date must be NaN
        post_exit_mask = synthetic_returns.index > exit_date
        assert result[10002][post_exit_mask].isna().all()
        # MSFT returns up to exit_date should be unchanged
        pre_exit_mask = synthetic_returns.index <= exit_date
        pd.testing.assert_series_equal(
            result[10002][pre_exit_mask],
            synthetic_returns[10002][pre_exit_mask],
        )

    def test_active_ticker_no_exit_unchanged(
        self, synthetic_returns: pd.DataFrame, synthetic_universe: pd.DataFrame
    ):
        """Ticker with NaT end_date (still active) must not be modified."""
        pit_df = pd.DataFrame(
            {
                "ticker":     ["AAPL"],
                "start_date": [pd.Timestamp("2019-01-01")],
                "end_date":   [pd.NaT],   # still in index
            }
        )
        result = apply_point_in_time_filter(
            synthetic_returns.copy(), synthetic_universe, pit_df=pit_df
        )
        pd.testing.assert_series_equal(result[10001], synthetic_returns[10001])

    def test_none_pit_df_skips_exit_filter(
        self, synthetic_returns: pd.DataFrame, synthetic_universe: pd.DataFrame
    ):
        """pit_df=None must not alter returns (all entry dates are before window)."""
        result = apply_point_in_time_filter(
            synthetic_returns.copy(), synthetic_universe, pit_df=None
        )
        pd.testing.assert_frame_equal(result, synthetic_returns)


# ---------------------------------------------------------------------------
# download_pit_membership — network layer (mocked)
# ---------------------------------------------------------------------------

class TestDownloadPitMembership:
    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        """download_pit_membership must return ticker/start_date/end_date columns."""
        import src.econometrics.sp500_loader as loader_mod

        # Build a minimal CSV that mimics fja05680 format
        fake_csv = "ticker,start_date,end_date\nAAPL,2010-01-01,2024-12-31\nMSFT,2015-06-01,\n"

        old_cache = loader_mod._PIT_CACHE
        loader_mod._PIT_CACHE = tmp_path / "test_pit.parquet"
        try:
            with patch("urllib.request.urlopen") as mock_open:
                mock_resp = MagicMock()
                mock_resp.read.return_value = fake_csv.encode()
                mock_resp.__enter__ = lambda s: mock_resp
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_open.return_value = mock_resp

                result = loader_mod.download_pit_membership(force_refresh=True)
        finally:
            loader_mod._PIT_CACHE = old_cache

        assert set(result.columns) >= {"ticker", "start_date", "end_date"}
        assert len(result) == 2

    def test_nat_end_date_for_active_tickers(self, tmp_path):
        """Blank end_date → NaT (ticker still in index)."""
        import src.econometrics.sp500_loader as loader_mod

        fake_csv = "ticker,start_date,end_date\nMSFT,2015-06-01,\n"

        old_cache = loader_mod._PIT_CACHE
        loader_mod._PIT_CACHE = tmp_path / "test_pit2.parquet"
        try:
            with patch("urllib.request.urlopen") as mock_open:
                mock_resp = MagicMock()
                mock_resp.read.return_value = fake_csv.encode()
                mock_resp.__enter__ = lambda s: mock_resp
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_open.return_value = mock_resp

                result = loader_mod.download_pit_membership(force_refresh=True)
        finally:
            loader_mod._PIT_CACHE = old_cache

        assert pd.isna(result.loc[result["ticker"] == "MSFT", "end_date"].iloc[0])

    def test_cache_used_on_second_call(self, tmp_path):
        """Second call without force_refresh must load from parquet cache."""
        import src.econometrics.sp500_loader as loader_mod

        fake_csv = "ticker,start_date,end_date\nGOOG,2004-08-19,\n"

        old_cache = loader_mod._PIT_CACHE
        cache_path = tmp_path / "test_pit3.parquet"
        loader_mod._PIT_CACHE = cache_path
        try:
            with patch("urllib.request.urlopen") as mock_open:
                mock_resp = MagicMock()
                mock_resp.read.return_value = fake_csv.encode()
                mock_resp.__enter__ = lambda s: mock_resp
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_open.return_value = mock_resp

                r1 = loader_mod.download_pit_membership(force_refresh=True)

            # Second call — urlopen should NOT be called again
            with patch("urllib.request.urlopen") as mock_open2:
                r2 = loader_mod.download_pit_membership(force_refresh=False)
                mock_open2.assert_not_called()

            pd.testing.assert_frame_equal(r1, r2, check_dtype=False)
        finally:
            loader_mod._PIT_CACHE = old_cache
