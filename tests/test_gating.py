"""
Tests for Phase I — gating.py (HAC + FWER)

TDD: tests specify mathematical bounds before implementation is verified.
"""

import numpy as np
import pandas as pd
import pytest

from src.econometrics.gating import (
    _holm_bonferroni,
    _holm_sidak,
    _holm_sidak_fwer,
    _newey_west_variance,
    gate_signals,
)


class TestNeweyWestVariance:
    def test_positive(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        var = _newey_west_variance(x, L=4)
        assert var > 0

    def test_iid_approaches_sample_variance(self):
        """For IID data with L=0, NW variance should equal sample variance."""
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, 500)
        nw = _newey_west_variance(x, L=0)
        sv = float(np.var(x))
        np.testing.assert_allclose(nw, sv, rtol=1e-10)

    def test_weights_exact(self):
        """Verify weight formula w_l = 1 - l/(L+1) is applied."""
        x = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        # With L=1: w_1 = 1 - 1/2 = 0.5
        # Manual: gamma_0 = mean(x^2), gamma_1 = mean(x[1:]*x[:-1])
        x_dm = x - x.mean()
        T = len(x)
        g0 = float(x_dm @ x_dm) / T
        g1 = float(x_dm[1:] @ x_dm[:-1]) / T
        expected = g0 + 2 * 0.5 * g1
        result = _newey_west_variance(x, L=1)
        np.testing.assert_allclose(result, max(expected, 1e-16), rtol=1e-10)


class TestHolmSidak:
    def test_all_small_pvalues_rejected(self):
        p = pd.Series({"a": 0.001, "b": 0.002, "c": 0.003})
        result = _holm_sidak(p, alpha=0.05)
        assert result.all()

    def test_all_large_pvalues_accepted(self):
        p = pd.Series({"a": 0.5, "b": 0.6, "c": 0.7})
        result = _holm_sidak(p, alpha=0.05)
        assert not result.any()

    def test_step_down_stops_at_first_failure(self):
        # Sorted order: a=0.001 (k=0), b=0.04 (k=1), c=0.5 (k=2)
        # k=0 threshold = 1-(0.95)^(1/3) ≈ 0.01695 → a REJECTED
        # k=1 threshold = 1-(0.95)^(1/2) ≈ 0.02532 → b=0.04 > threshold → STOP
        # => b and c not rejected
        p = pd.Series({"a": 0.001, "b": 0.04, "c": 0.5})
        result = _holm_sidak(p, alpha=0.05)
        assert result["a"] is np.bool_(True)
        assert result["b"] is np.bool_(False)
        assert result["c"] is np.bool_(False)


class TestHolmBonferroni:
    def test_fallback_rejects_smallest(self):
        p = pd.Series({"a": 0.001, "b": 0.5})
        result = _holm_bonferroni(p, alpha=0.05)
        assert result["a"] is np.bool_(True)
        assert result["b"] is np.bool_(False)


class TestHolmSidakFwer:
    def test_returns_boolean_series(self):
        p = pd.Series({"x": 0.01, "y": 0.9})
        result = _holm_sidak_fwer(p, alpha=0.05)
        assert result.dtype == bool


class TestGateSignals:
    def test_output_is_boolean_series(self):
        rng = np.random.default_rng(0)
        T = 200
        dates = pd.date_range("2010-01-01", periods=T, freq="B")
        state = pd.DataFrame(
            {"f1": rng.normal(0, 1, T), "f2": rng.normal(0, 1, T)},
            index=dates,
        )
        fwd = pd.Series(rng.normal(0.001, 0.01, T), index=dates)
        spreads = pd.Series({"f1": 0.0002, "f2": 0.0002})
        result = gate_signals(state, fwd, spreads)
        assert result.dtype == bool
        assert set(result.index) == set(state.columns)

    def test_high_return_signals_pass(self):
        """Signals with returns >> spread should consistently pass."""
        rng = np.random.default_rng(1)
        T = 500
        dates = pd.date_range("2010-01-01", periods=T, freq="B")
        state = pd.DataFrame({"strong": rng.normal(0, 1, T)}, index=dates)
        fwd = pd.Series(0.05 + rng.normal(0, 0.001, T), index=dates)  # large positive drift
        spreads = pd.Series({"strong": 0.0001})
        result = gate_signals(state, fwd, spreads)
        assert result["strong"]
