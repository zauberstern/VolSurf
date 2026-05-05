"""
Tests for Phase I — data_ingestion.py

TDD: tests are written first to specify expected behaviour.
"""

import numpy as np
import pandas as pd
import pytest

from src.econometrics.data_ingestion import (
    _align_vix,
    _rolling_realized_variance,
    _volatility_risk_premium,
    build_state_vector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def price_series() -> pd.Series:
    rng = np.random.default_rng(0)
    prices = 4000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    dates = pd.date_range("2010-01-01", periods=200, freq="B")
    return pd.Series(prices, index=dates)


@pytest.fixture()
def option_panel(price_series: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = price_series.index
    return pd.DataFrame(
        {
            "date": dates,
            "iv_30": 0.15 + rng.normal(0, 0.01, len(dates)),
            "iv_91": 0.18 + rng.normal(0, 0.01, len(dates)),
            "skew_25d": -0.03 + rng.normal(0, 0.005, len(dates)),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRollingRealizedVariance:
    def test_shape(self, price_series):
        rv = _rolling_realized_variance(price_series, window=21)
        assert len(rv) == len(price_series)

    def test_nan_in_first_window(self, price_series):
        rv = _rolling_realized_variance(price_series, window=21)
        assert rv.iloc[:21].isna().all()

    def test_positive_values(self, price_series):
        rv = _rolling_realized_variance(price_series, window=21)
        assert (rv.dropna() > 0).all()

    def test_annualisation_scale(self, price_series):
        """RV should be of similar magnitude to squared vol (annualised)."""
        rv = _rolling_realized_variance(price_series, window=21).dropna()
        # Log-returns ~1% daily -> annualised var ~(0.01)^2*252 ≈ 0.025
        assert rv.mean() < 1.0


class TestAlignVix:
    def test_no_vxo_returns_vix(self):
        dates = pd.date_range("2000-01-01", periods=10, freq="B")
        vix = pd.Series(0.2, index=dates)
        result = _align_vix(vix, vxo=None)
        pd.testing.assert_series_equal(result.rename("vix"), vix.rename("vix"))

    def test_pre_2003_substituted(self):
        dates = pd.date_range("2002-01-01", periods=10, freq="B")
        vix = pd.Series(0.25, index=dates)
        vxo = pd.Series(0.30, index=dates)
        result = _align_vix(vix, vxo=vxo)
        assert (result.values == 0.30).all(), "Pre-2003 data should use VXO"

    def test_post_2003_uses_vix(self):
        dates = pd.date_range("2004-01-01", periods=10, freq="B")
        vix = pd.Series(0.15, index=dates)
        vxo = pd.Series(0.99, index=dates)  # clearly wrong value
        result = _align_vix(vix, vxo=vxo)
        assert (result.values == 0.15).all(), "Post-2003 data should use VIX"


class TestBuildStateVector:
    def test_no_lookahead(self, option_panel, price_series):
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        state, fwd = build_state_vector(option_panel, price_series, vix, treasury)
        # Forward return at time t must be for t+1, so last index of fwd
        # must be strictly earlier than last index of price_series
        assert fwd.index[-1] < price_series.index[-1]

    def test_state_columns(self, option_panel, price_series):
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        state, _ = build_state_vector(option_panel, price_series, vix, treasury)
        expected = {"iv_30", "iv_91", "term_structure", "skew_25d", "rv_21", "rv_63", "vrp", "vix", "yield_10y"}
        assert expected.issubset(set(state.columns))

    def test_aligned_lengths(self, option_panel, price_series):
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        state, fwd = build_state_vector(option_panel, price_series, vix, treasury)
        assert len(state) == len(fwd)

    def test_optional_features_log_pcr(self, option_panel, price_series):
        """log_pcr adds a 10th column when supplied."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        log_pcr = pd.Series(0.05, index=price_series.index)
        state, _ = build_state_vector(
            option_panel, price_series, vix, treasury, log_pcr=log_pcr
        )
        assert "log_pcr" in state.columns
        assert state.shape[1] == 10

    def test_optional_features_zerocd(self, option_panel, price_series):
        """short_rate from zerocd adds another column."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        zerocd = pd.DataFrame(
            {"short_rate": 0.05, "curve_slope": 0.01}, index=price_series.index
        )
        state, _ = build_state_vector(
            option_panel, price_series, vix, treasury, zerocd=zerocd
        )
        assert "short_rate" in state.columns

    def test_optional_features_iv_dispersion(self, option_panel, price_series):
        """iv_dispersion adds a column; forward-fill means no extra NaN rows."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        # Provide dispersion only for first half, rest NaN (simulates 2021 cutoff)
        disp_vals = [0.12] * 100 + [np.nan] * 100
        iv_dispersion = pd.Series(disp_vals, index=price_series.index)
        state_base, _ = build_state_vector(option_panel, price_series, vix, treasury)
        state_disp, _ = build_state_vector(
            option_panel, price_series, vix, treasury, iv_dispersion=iv_dispersion
        )
        assert "iv_dispersion" in state_disp.columns
        # Forward-fill should not shrink the row count vs base
        assert len(state_disp) >= len(state_base) - 5  # allow minor boundary variation

    def test_optional_features_skew_10d(self, price_series):
        """skew_10d is added when present in the option panel."""
        rng = np.random.default_rng(42)
        dates = price_series.index
        # Build option_panel that includes skew_10d column
        panel_with_skew10 = pd.DataFrame(
            {
                "date":     dates,
                "iv_30":    0.15 + rng.normal(0, 0.01, len(dates)),
                "iv_91":    0.18 + rng.normal(0, 0.01, len(dates)),
                "skew_25d": -0.03 + rng.normal(0, 0.005, len(dates)),
                "skew_10d":  0.08 + rng.normal(0, 0.01, len(dates)),  # tail-risk premium
            }
        )
        vix = pd.Series(0.20, index=dates)
        treasury = pd.Series(0.04, index=dates)
        state, _ = build_state_vector(panel_with_skew10, price_series, vix, treasury)
        assert "skew_10d" in state.columns

    def test_skew_10d_not_present_when_column_absent(self, option_panel, price_series):
        """When the panel lacks skew_10d the state must not contain it."""
        assert "skew_10d" not in option_panel.columns
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        state, _ = build_state_vector(option_panel, price_series, vix, treasury)
        assert "skew_10d" not in state.columns

    def test_full_12_features(self, option_panel, price_series):
        """All optional sources together yield 12 features."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        log_pcr = pd.Series(0.05, index=price_series.index)
        zerocd = pd.DataFrame(
            {"short_rate": 0.05, "curve_slope": 0.01}, index=price_series.index
        )
        iv_dispersion = pd.Series(0.12, index=price_series.index)
        state, _ = build_state_vector(
            option_panel, price_series, vix, treasury,
            log_pcr=log_pcr, zerocd=zerocd, iv_dispersion=iv_dispersion,
        )
        assert state.shape[1] == 12

    def test_vxn_vxd_spread_columns_added(self, option_panel, price_series):
        """When vxn and vxd are supplied, vxn_vix_spread and vxd_vix_spread appear."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        vxn = pd.Series(0.28, index=price_series.index)
        vxd = pd.Series(0.18, index=price_series.index)
        state, _ = build_state_vector(
            option_panel, price_series, vix, treasury, vxn=vxn, vxd=vxd
        )
        assert "vxn_vix_spread" in state.columns
        assert "vxd_vix_spread" in state.columns

    def test_vxn_spread_is_log_ratio(self, option_panel, price_series):
        """vxn_vix_spread should be log(vxn / vix_aligned), approximately."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        # VXN = 2× VIX → spread ≈ log(2) ≈ 0.693
        vxn = pd.Series(0.40, index=price_series.index)
        state, _ = build_state_vector(
            option_panel, price_series, vix, treasury, vxn=vxn
        )
        assert "vxn_vix_spread" in state.columns
        spread_mean = float(state["vxn_vix_spread"].mean())
        assert abs(spread_mean - np.log(2)) < 0.05, (
            f"Expected log(2)≈0.693 but got {spread_mean:.4f}"
        )

    def test_vxn_absent_no_column(self, option_panel, price_series):
        """When vxn/vxd are not passed the spread columns must not appear."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        state, _ = build_state_vector(option_panel, price_series, vix, treasury)
        assert "vxn_vix_spread" not in state.columns
        assert "vxd_vix_spread" not in state.columns

    def test_vxn_with_duplicate_index(self, option_panel, price_series):
        """Duplicate index in vxn/vxd must not crash build_state_vector."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        # Introduce a duplicate date
        idx_dup = price_series.index.tolist()
        idx_dup[10] = idx_dup[9]  # duplicate day 9
        vxn = pd.Series(0.28, index=idx_dup)
        state, _ = build_state_vector(
            option_panel, price_series, vix, treasury, vxn=vxn
        )
        assert "vxn_vix_spread" in state.columns

    def test_rv_dispersion_column_added(self, option_panel, price_series):
        """When rv_dispersion is supplied, it appears in the state."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        rv_disp = pd.Series(0.15, index=price_series.index, name="rv_dispersion")
        state, _ = build_state_vector(
            option_panel, price_series, vix, treasury, rv_dispersion=rv_disp
        )
        assert "rv_dispersion" in state.columns

    def test_full_16_features(self, option_panel, price_series):
        """All optional sources (incl. new signals) together yield up to 12+3=15 features."""
        vix = pd.Series(0.20, index=price_series.index)
        treasury = pd.Series(0.04, index=price_series.index)
        log_pcr = pd.Series(0.05, index=price_series.index)
        zerocd = pd.DataFrame(
            {"short_rate": 0.05, "curve_slope": 0.01}, index=price_series.index
        )
        iv_dispersion = pd.Series(0.12, index=price_series.index)
        vxn = pd.Series(0.28, index=price_series.index)
        vxd = pd.Series(0.18, index=price_series.index)
        rv_disp = pd.Series(0.15, index=price_series.index, name="rv_dispersion")
        state, _ = build_state_vector(
            option_panel, price_series, vix, treasury,
            log_pcr=log_pcr, zerocd=zerocd, iv_dispersion=iv_dispersion,
            vxn=vxn, vxd=vxd, rv_dispersion=rv_disp,
        )
        assert state.shape[1] == 15
