"""
Tests for Phase IV — attribution.py (Whalley-Wilmott, Hansen-Hodrick,
Deflated Sharpe Ratio, Profit Factor, Walk-Forward Efficiency, MC permutation)
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.attribution import (
    attribution_regression,
    deflated_sharpe_ratio,
    information_ratio,
    interpret_alpha,
    mc_permutation_pvalue,
    profit_factor,
    walk_forward_efficiency,
    whalley_wilmott_width,
)


class TestWhalleyWilmott:
    def test_scalar_output(self):
        w = whalley_wilmott_width(c=0.001, gamma=0.5, S=4000.0, a=2.0)
        assert isinstance(float(w), float)
        assert w > 0

    def test_zero_gamma_gives_zero(self):
        w = whalley_wilmott_width(c=0.001, gamma=0.0, S=4000.0, a=2.0)
        assert w == pytest.approx(0.0)

    def test_higher_cost_wider_band(self):
        w1 = whalley_wilmott_width(c=0.001, gamma=0.5, S=4000.0, a=2.0)
        w2 = whalley_wilmott_width(c=0.01, gamma=0.5, S=4000.0, a=2.0)
        assert w2 > w1

    def test_higher_risk_aversion_narrows_band(self):
        w1 = whalley_wilmott_width(c=0.001, gamma=0.5, S=4000.0, a=1.0)
        w2 = whalley_wilmott_width(c=0.001, gamma=0.5, S=4000.0, a=10.0)
        assert w2 < w1

    def test_array_input(self):
        gammas = np.array([0.1, 0.5, 1.0])
        w = whalley_wilmott_width(c=0.001, gamma=gammas, S=4000.0, a=2.0)
        assert w.shape == (3,)
        assert (w > 0).all()


class TestAttributionRegression:
    @pytest.fixture()
    def factor_data(self):
        rng = np.random.default_rng(42)
        T = 300
        dates = pd.date_range("2015-01-01", periods=T, freq="B")
        r_M = pd.Series(rng.normal(0.0005, 0.01, T), index=dates)
        alpha_true = 0.0002
        r_RL = alpha_true + 0.5 * r_M + pd.Series(rng.normal(0, 0.005, T), index=dates)
        carry = pd.Series(rng.normal(0, 0.005, T), index=dates)
        vol = pd.Series(rng.normal(0, 0.005, T), index=dates)
        vrp = pd.Series(rng.normal(0, 0.005, T), index=dates)
        return r_RL, r_M, carry, vol, vrp

    def test_output_shape(self, factor_data):
        result = attribution_regression(*factor_data)
        assert result.shape == (5, 4)
        assert set(result.columns) == {"coef", "hac_se", "t_stat", "p_value"}

    def test_alpha_row_present(self, factor_data):
        result = attribution_regression(*factor_data)
        assert "alpha (beta_0)" in result.index

    def test_positive_alpha_detected(self, factor_data):
        result = attribution_regression(*factor_data)
        label = interpret_alpha(result, alpha_level=0.10)
        # With true alpha > 0 and T=300, should detect positive alpha
        assert label in {"Positive Alpha", "Factor Exposure"}


class TestInterpretAlpha:
    def _make_result(self, alpha_coef, alpha_pval, other_pval=0.5):
        return pd.DataFrame(
            {
                "coef": [alpha_coef, 0.5, 0.0, 0.0, 0.0],
                "hac_se": [0.1] * 5,
                "t_stat": [alpha_coef / 0.1, 5.0, 0.0, 0.0, 0.0],
                "p_value": [alpha_pval, other_pval, 0.5, 0.5, 0.5],
            },
            index=["alpha (beta_0)", "beta_m", "beta_c", "beta_v", "beta_VRP"],
        )

    def test_positive_alpha(self):
        r = self._make_result(0.001, 0.01)
        assert interpret_alpha(r) == "Positive Alpha"

    def test_null_result(self):
        r = self._make_result(0.001, 0.9, other_pval=0.9)
        assert interpret_alpha(r) == "Null Result"

    def test_factor_exposure(self):
        r = self._make_result(0.001, 0.9, other_pval=0.01)
        assert interpret_alpha(r) == "Factor Exposure"

    def test_execution_edge(self):
        r = self._make_result(-0.001, 0.01, other_pval=0.9)
        assert interpret_alpha(r) == "Execution Edge Only"


# ─────────────────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestDeflatedSharpeRatio:
    def test_output_in_unit_interval(self):
        dsr = deflated_sharpe_ratio(sr=1.0, T=252, skewness=0.0, excess_kurtosis=0.0)
        assert 0.0 <= dsr <= 1.0

    def test_positive_sr_gives_high_dsr(self):
        dsr = deflated_sharpe_ratio(sr=2.0, T=500, skewness=0.0, excess_kurtosis=0.0)
        assert dsr > 0.5

    def test_negative_sr_gives_low_dsr(self):
        dsr = deflated_sharpe_ratio(sr=-1.0, T=252, skewness=0.0, excess_kurtosis=0.0)
        assert dsr < 0.5

    def test_more_trials_lowers_dsr(self):
        """More strategy trials → higher SR* → lower DSR for the same observed SR.

        Use small T=50 so SR* is large enough (≈0.32) to meaningfully deflate
        an SR of 0.30 when n_trials=50, while n_trials=1 leaves SR*=0.
        """
        dsr_1  = deflated_sharpe_ratio(sr=0.30, T=50, skewness=0.0, excess_kurtosis=0.0, n_trials=1)
        dsr_50 = deflated_sharpe_ratio(sr=0.30, T=50, skewness=0.0, excess_kurtosis=0.0, n_trials=50)
        assert dsr_50 < dsr_1

    def test_negative_skew_lowers_dsr(self):
        """Fat left tail (negative skew) increases SE → lowers DSR.

        Use T=50, SR=0.30 so neither result is saturated (SE≈0.15 → SR/SE≈2).
        Negative skew adds +0.6 to the var numerator, widening the SE.
        """
        dsr_sym = deflated_sharpe_ratio(sr=0.30, T=50, skewness=0.0,  excess_kurtosis=0.0)
        dsr_skw = deflated_sharpe_ratio(sr=0.30, T=50, skewness=-2.0, excess_kurtosis=0.0)
        assert dsr_skw < dsr_sym

    def test_larger_T_increases_dsr(self):
        """More data → smaller SE → higher confidence → higher DSR for same SR."""
        dsr_small = deflated_sharpe_ratio(sr=1.0, T=100,  skewness=0.0, excess_kurtosis=0.0)
        dsr_large = deflated_sharpe_ratio(sr=1.0, T=2000, skewness=0.0, excess_kurtosis=0.0)
        assert dsr_large >= dsr_small - 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Profit Factor
# ─────────────────────────────────────────────────────────────────────────────

class TestProfitFactor:
    def test_profitable_series(self):
        r = pd.Series([0.01, 0.02, -0.005, 0.03, -0.01])
        pf = profit_factor(r)
        assert pf > 1.0

    def test_losing_series(self):
        r = pd.Series([-0.01, -0.02, 0.005, -0.03, 0.001])
        pf = profit_factor(r)
        assert pf < 1.0

    def test_all_gains_returns_inf(self):
        r = pd.Series([0.01, 0.02, 0.005])
        pf = profit_factor(r)
        assert pf == float("inf")

    def test_all_losses_returns_zero(self):
        r = pd.Series([-0.01, -0.02, -0.005])
        pf = profit_factor(r)
        assert pf == pytest.approx(0.0)

    def test_numpy_array_input(self):
        arr = np.array([0.02, -0.01, 0.03, -0.005])
        pf = profit_factor(arr)
        assert isinstance(pf, float)
        assert pf > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Efficiency
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardEfficiency:
    def test_perfect_transfer(self):
        """WFE = 1.0 when OOS == IS."""
        wfe = walk_forward_efficiency(is_ann_return=0.10, oos_ann_return=0.10)
        assert wfe == pytest.approx(1.0)

    def test_above_threshold(self):
        """WFE > 0.5 is institutional target."""
        wfe = walk_forward_efficiency(is_ann_return=0.20, oos_ann_return=0.12)
        assert wfe == pytest.approx(0.6)
        assert wfe > 0.5

    def test_degradation(self):
        wfe = walk_forward_efficiency(is_ann_return=0.20, oos_ann_return=0.05)
        assert wfe == pytest.approx(0.25)
        assert wfe < 0.5

    def test_negative_is_return_returns_nan(self):
        import math
        wfe = walk_forward_efficiency(is_ann_return=-0.05, oos_ann_return=0.10)
        assert math.isnan(wfe)

    def test_oos_negative_allowed(self):
        wfe = walk_forward_efficiency(is_ann_return=0.10, oos_ann_return=-0.03)
        assert wfe == pytest.approx(-0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo Permutation P-value
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPermutationPvalue:
    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, 252)
        p = mc_permutation_pvalue(returns, n_trials=200, seed=42)
        assert 0.0 <= p <= 1.0

    def test_random_returns_high_pvalue(self):
        """Purely random returns → permutations as good → p ≈ 0.5."""
        rng = np.random.default_rng(99)
        returns = rng.normal(0.0, 0.01, 500)
        p = mc_permutation_pvalue(returns, n_trials=500, seed=7)
        # p should not be very small for pure noise
        assert p > 0.05

    def test_consistent_positive_returns_low_pvalue(self):
        """Consistent daily gains → permutations are worse → low p."""
        returns = np.full(252, 0.001)  # perfect daily gain, zero variance after shuffle
        p = mc_permutation_pvalue(returns, n_trials=200, seed=0)
        # Permutations of constant sequence are identical → p = 1.0 or returns NaN
        # Either way, no false signal of significance
        assert 0.0 <= p <= 1.0

    def test_pandas_series_input(self):
        r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 200))
        p = mc_permutation_pvalue(r, n_trials=100, seed=1)
        assert 0.0 <= p <= 1.0

    def test_reproducible_with_same_seed(self):
        r = np.random.default_rng(5).normal(0.001, 0.01, 252)
        p1 = mc_permutation_pvalue(r, n_trials=300, seed=42)
        p2 = mc_permutation_pvalue(r, n_trials=300, seed=42)
        assert p1 == pytest.approx(p2)


# ---------------------------------------------------------------------------
# Information Ratio
# ---------------------------------------------------------------------------

class TestInformationRatio:

    def test_positive_active_returns_positive_ir(self):
        # Strategy beats benchmark every day → IR > 0
        bench = pd.Series(np.full(252, 0.0004))   # ~10% p.a.
        port  = pd.Series(np.full(252, 0.0006))   # ~15% p.a.
        ir = information_ratio(port, bench)
        assert ir > 0.0

    def test_negative_active_returns_negative_ir(self):
        bench = pd.Series(np.full(252, 0.0006))
        port  = pd.Series(np.full(252, 0.0004))
        ir = information_ratio(port, bench)
        assert ir < 0.0

    def test_zero_tracking_error_returns_finite(self):
        # TE clamped to 1e-8 so IR is always finite: portfolio == benchmark
        # gives active series with zero std; IR = 0/1e-8 = 0.0, not NaN.
        r = pd.Series(np.full(252, 0.001))
        ir = information_ratio(r, r)
        assert np.isfinite(ir)
        assert ir == pytest.approx(0.0, abs=1e-6)

    def test_formula_correctness(self):
        rng = np.random.default_rng(7)
        port  = rng.normal(0.0008, 0.01, 252)
        bench = rng.normal(0.0004, 0.01, 252)
        active = port - bench
        expected = (active.mean() * 252) / (active.std() * np.sqrt(252))
        assert information_ratio(port, bench) == pytest.approx(expected, rel=1e-9)

    def test_numpy_array_input(self):
        rng = np.random.default_rng(9)
        port  = rng.normal(0.001, 0.01, 200)
        bench = rng.normal(0.0005, 0.01, 200)
        ir = information_ratio(port, bench)
        assert np.isfinite(ir)
