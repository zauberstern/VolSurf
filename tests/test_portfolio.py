"""
Tests for SP500 portfolio modules.

Uses synthetic data only — no WRDS connection required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.cross_section import (
    backtest_portfolio,
    build_portfolio_weights,
    compute_cross_sectional_signals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """100-day × 20-stock return panel, seeded for reproducibility."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", periods=100, freq="B")
    permnos = list(range(1001, 1021))
    data = rng.normal(0.0, 0.01, size=(100, 20))
    return pd.DataFrame(data, index=dates, columns=permnos)


@pytest.fixture
def synthetic_vix(synthetic_returns) -> pd.Series:
    return pd.Series(0.20, index=synthetic_returns.index, name="vix")


@pytest.fixture
def benchmark_returns(synthetic_returns) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(
        rng.normal(0.0004, 0.01, len(synthetic_returns)),
        index=synthetic_returns.index,
        name="spx",
    )


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------

class TestComputeCrossSectionalSignals:
    def test_output_columns(self, synthetic_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        assert set(["rv_21", "rv_63", "mom_252_21", "vrp_proxy"]).issubset(signals.columns)

    def test_multiindex(self, synthetic_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        assert signals.index.names == ["date", "permno"]

    def test_rv21_positive(self, synthetic_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        rv = signals["rv_21"].dropna()
        assert (rv >= 0).all(), "Realized variance must be non-negative"

    def test_vrp_proxy_with_vix(self, synthetic_returns, synthetic_vix):
        signals = compute_cross_sectional_signals(synthetic_returns, vix=synthetic_vix)
        vrp = signals["vrp_proxy"].dropna()
        assert len(vrp) > 0, "VRP proxy should be non-empty when VIX supplied"

    def test_vrp_nan_without_vix(self, synthetic_returns):
        signals = compute_cross_sectional_signals(synthetic_returns, vix=None)
        assert signals["vrp_proxy"].isna().all(), "VRP proxy should be all-NaN without VIX"

    def test_no_lookahead(self, synthetic_returns):
        """RV_21 at date t should only use returns up to and including t.
        With min_periods=10, the first 9 rows must be NaN."""
        signals = compute_cross_sectional_signals(synthetic_returns)
        rv = signals["rv_21"].unstack("permno")
        # First min_periods-1 = 9 rows have fewer than min_periods observations → NaN
        assert rv.iloc[:9].isna().all().all()


# ---------------------------------------------------------------------------
# Weight tests
# ---------------------------------------------------------------------------

class TestBuildPortfolioWeights:
    def test_equal_weight_sums_to_one(self, synthetic_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        w = build_portfolio_weights(signals, method="equal_weight")
        row_sums = w.sum(axis=1)
        valid = row_sums[row_sums > 0]
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-10)

    def test_equal_weight_nonnegative(self, synthetic_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        w = build_portfolio_weights(signals, method="equal_weight")
        assert (w >= 0).all().all()

    def test_vrp_quartile_sums_to_one(self, synthetic_returns, synthetic_vix):
        signals = compute_cross_sectional_signals(synthetic_returns, vix=synthetic_vix)
        w = build_portfolio_weights(signals, method="vrp_quartile")
        row_sums = w.sum(axis=1)
        valid = row_sums[row_sums > 0]
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-10)

    def test_momentum_weight_nonnegative(self, synthetic_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        w = build_portfolio_weights(signals, method="momentum")
        assert (w >= 0).all().all()

    def test_vrp_quartile_selects_top_25pct(self, synthetic_returns, synthetic_vix):
        """Number of selected stocks each day should be ~25% of 20 = 5."""
        signals = compute_cross_sectional_signals(synthetic_returns, vix=synthetic_vix)
        w = build_portfolio_weights(signals, method="vrp_quartile")
        n_selected = (w > 0).sum(axis=1)
        valid = n_selected[n_selected > 0]
        # With 20 stocks, top 25% = 5 stocks
        assert valid.median() <= 6, f"Expected ~5 stocks selected, got median={valid.median()}"


# ---------------------------------------------------------------------------
# Backtest tests
# ---------------------------------------------------------------------------

class TestBacktestPortfolio:
    def test_returns_dict_keys(self, synthetic_returns, benchmark_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        w = build_portfolio_weights(signals, method="equal_weight")
        result = backtest_portfolio(w, synthetic_returns, benchmark_returns)
        assert "portfolio_returns" in result
        assert "benchmark_returns" in result
        assert "metrics" in result

    def test_portfolio_returns_length(self, synthetic_returns, benchmark_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        w = build_portfolio_weights(signals, method="equal_weight")
        result = backtest_portfolio(w, synthetic_returns, benchmark_returns)
        assert len(result["portfolio_returns"]) == len(synthetic_returns)

    def test_metrics_structure(self, synthetic_returns, benchmark_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        w = build_portfolio_weights(signals, method="equal_weight")
        result = backtest_portfolio(w, synthetic_returns, benchmark_returns)
        m = result["metrics"]
        assert "portfolio" in m
        assert "benchmark" in m
        assert "relative" in m
        assert "sharpe" in m["portfolio"]
        assert "information_ratio" in m["relative"]

    def test_annual_vol_positive(self, synthetic_returns, benchmark_returns):
        signals = compute_cross_sectional_signals(synthetic_returns)
        w = build_portfolio_weights(signals, method="equal_weight")
        result = backtest_portfolio(w, synthetic_returns, benchmark_returns)
        assert result["metrics"]["portfolio"]["annual_vol"] > 0

    def test_tc_reduces_returns(self, synthetic_returns, benchmark_returns):
        """Higher transaction costs should reduce net portfolio return."""
        signals = compute_cross_sectional_signals(synthetic_returns)
        w = build_portfolio_weights(signals, method="equal_weight")
        r_no_tc = backtest_portfolio(w, synthetic_returns, benchmark_returns, transaction_cost_bps=0)
        r_with_tc = backtest_portfolio(w, synthetic_returns, benchmark_returns, transaction_cost_bps=50)
        cr_no_tc = r_no_tc["metrics"]["portfolio"]["cum_return"]
        cr_with_tc = r_with_tc["metrics"]["portfolio"]["cum_return"]
        assert cr_no_tc >= cr_with_tc, "Higher TC should not improve returns"
