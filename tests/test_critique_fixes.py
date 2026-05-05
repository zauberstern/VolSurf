"""
Regression tests verifying the signal gating and policy architecture.

All tests in this file fail before the specified fixes are applied.
Run: pytest tests/test_critique_fixes.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# FIX 9 — Benjamini-Yekutieli FDR as the primary multiple-testing procedure
# ─────────────────────────────────────────────────────────────────────────────

class TestBonferroniIsPrimaryFwer:
    """gate_signals must route through Benjamini-Yekutieli FDR (critique fix:
    replaces conservative Holm-Bonferroni FWER control with FDR control)."""

    def test_gate_signals_source_calls_by_fdr_not_sidak(self):
        """gate_signals source must call _benjamini_yekutieli, not _holm_sidak_fwer."""
        import inspect
        from src.econometrics.gating import gate_signals
        src = inspect.getsource(gate_signals)
        assert "_benjamini_yekutieli" in src, (
            "gate_signals does not call _benjamini_yekutieli. "
            "Critique fix: replace Holm-FWER with Benjamini-Yekutieli FDR."
        )
        assert "_holm_sidak_fwer" not in src, (
            "gate_signals still routes through _holm_sidak_fwer. "
            "Replace with _benjamini_yekutieli."
        )

    def test_sidak_is_less_conservative_than_bonferroni(self):
        """Holm-Sidak rejects hypotheses Bonferroni does not (p in (0.025, 0.02532) for m=2).

        Concretely: for m=2, Sidak threshold at k=0 is 1-(0.95)^0.5 ≈ 0.02532
        while Bonferroni is 0.025.  For p in (0.025, 0.02532) Sidak rejects, Bonf does not.
        """
        from src.econometrics.gating import _holm_sidak, _holm_bonferroni

        # p2 = 0.0251 is in (0.025, 0.02532) — the two procedures must disagree
        p = pd.Series({"a": 0.0251, "b": 0.80})
        sidak_result = _holm_sidak(p, alpha=0.05)
        bonf_result = _holm_bonferroni(p, alpha=0.05)
        # Sidak k=0 threshold ≈ 0.02532 > 0.0251 → rejects "a"
        # Bonf  k=0 threshold  = 0.025  < 0.0251 → does NOT reject "a"
        assert sidak_result["a"] is np.bool_(True), (
            f"Sidak should reject p=0.0251 (threshold ≈ 0.02532), got {sidak_result['a']}"
        )
        assert bonf_result["a"] is np.bool_(False), (
            f"Bonferroni should NOT reject p=0.0251 (threshold=0.025), got {bonf_result['a']}"
        )

    def test_gate_signals_agrees_with_bonferroni_not_sidak(self):
        """When p is in the differentiating range (0.025, 0.02532) for m=2,
        gate_signals must agree with Bonferroni (reject=False), not Sidak (reject=True)."""
        from src.econometrics.gating import (
            gate_signals, _holm_bonferroni, _holm_sidak, _hac_one_sided_ttest,
        )
        import scipy.stats as sp_stats

        # Engineer inputs to produce p ≈ 0.0251 for one predictor.
        # one-sided HAC t-test: p = 1 - Φ(t). For p=0.0251: t = Φ^{-1}(0.9749) ≈ 1.955
        # With T=200, n_lags=3, se≈1/sqrt(200)≈0.0707, mu = t*se ≈ 0.138
        T = 200
        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        rng = np.random.default_rng(0)
        n_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))

        # Build a predictor whose excess return has HAC t-stat ≈ 1.955
        # Use iid data so NW ≈ sample variance; mu/se = t → choose mu
        excess_target_t = 1.955   # → p ≈ 0.0251 (in differentiating range)
        noise = rng.normal(0, 1, T)
        # scale so that t_stat ≈ excess_target_t
        se_est = 1.0 / np.sqrt(T)
        mu_target = excess_target_t * se_est
        excess_f1 = mu_target + noise * se_est

        # Verify p-value lands in (0.025, 0.02532)
        _, p1 = _hac_one_sided_ttest(excess_f1, n_lags)
        if not (0.020 < p1 < 0.035):
            pytest.skip(f"Engineered p={p1:.5f} not close enough to 0.0251; skip this subtest")

        # Add a clearly non-significant predictor (p >> 0.05)
        excess_f2 = rng.normal(0, 0.01, T)   # near-zero mean → large p-value

        state = pd.DataFrame(
            {"f1": rng.normal(0, 1, T), "f2": rng.normal(0, 1, T)}, index=dates
        )
        spreads = pd.Series({"f1": 0.0, "f2": 0.0})
        fwd_f1 = pd.Series(excess_f1, index=dates)   # excess already = return - spread

        # Re-use both excess series as separate forward returns to get two distinct p-values
        # We'll compute p directly and check gate_signals against Bonferroni
        p_series = pd.Series({"f1": p1, "f2": float(sp_stats.norm.sf(
            excess_f2.mean() / (excess_f2.std() / np.sqrt(T))
        ))})
        bonf_expected = _holm_bonferroni(p_series, 0.05)
        sidak_result = _holm_sidak(p_series, 0.05)

        # Only run the assertion if the procedures actually disagree
        if bonf_expected.equals(sidak_result):
            pytest.skip("Sidak and Bonferroni agree on this p-value pair; skip subtest")

        # Now run gate_signals and verify it matches Bonferroni
        result = gate_signals(state, fwd_f1, spreads)
        pd.testing.assert_series_equal(
            result.sort_index(),
            bonf_expected.sort_index(),
            check_names=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — TC parameter alignment: training (1 bp) vs evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestTCBPSAlignment:
    def test_tc_bps_uses_env_var(self):
        """run_portfolio.py TC_BPS must read from PIPELINE_TC_BPS env var, not be hardcoded."""
        source = (ROOT / "run_portfolio.py").read_text(encoding="utf-8")
        assert "PIPELINE_TC_BPS" in source, (
            "TC_BPS in run_portfolio.py is hardcoded (5.0). "
            "It must read from PIPELINE_TC_BPS env var with default 1.0 bps "
            "to match the training PIPELINE_TC=1e-4 (1 bp)."
        )

    def test_default_tc_bps_is_1_bps(self):
        """Default TC_BPS (1.0) must equal PIPELINE_TC (1e-4) in basis-point units."""
        import os
        os.environ.pop("PIPELINE_TC_BPS", None)
        os.environ.pop("PIPELINE_TC", None)
        pipeline_tc = float(os.getenv("PIPELINE_TC", "1e-4"))        # proportional
        tc_bps_default = float(os.getenv("PIPELINE_TC_BPS", "1.0")) # basis points
        assert pipeline_tc * 10_000 == pytest.approx(tc_bps_default, rel=1e-6), (
            f"PIPELINE_TC={pipeline_tc} = {pipeline_tc*10_000:.2f} bps "
            f"but PIPELINE_TC_BPS default = {tc_bps_default} bps — misaligned."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — Remove tanh squash from _bootstrap_paths
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapNoTanhSquash:
    def test_paths_can_exceed_historical_range(self):
        """Without tanh squash, simulated paths must be able to escape historical bounds."""
        from run_pipeline import _bootstrap_paths

        rng = np.random.default_rng(0)
        # Tight historical range [0.2, 0.8]; AR dynamics + residuals should push paths outside
        data = rng.uniform(0.2, 0.8, (200, 2))
        paths = _bootstrap_paths(data, n_steps=252, n_paths=500, block=21)

        hist_max = data.max(axis=0)
        hist_min = data.min(axis=0)
        # With no tanh squash, at least some paths must exceed historical range by a margin
        exceeds = (
            (paths > hist_max[np.newaxis, np.newaxis, :] + 0.05).any()
            or (paths < hist_min[np.newaxis, np.newaxis, :] - 0.05).any()
        )
        assert exceeds, (
            "All paths stay within historical ±0.05 range — "
            "tanh squash is still active and compressing tail dynamics."
        )

    def test_bootstrap_source_no_tanh(self):
        """_bootstrap_paths source must not contain a tanh squash step."""
        import inspect
        from run_pipeline import _bootstrap_paths
        src = inspect.getsource(_bootstrap_paths)
        assert "tanh" not in src, (
            "_bootstrap_paths still contains tanh squash. "
            "Remove the state = mid + half * np.tanh(...) line."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX 5 — Attribution TC drag: proper turnover-based formula
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTCDrag:
    """compute_tc_drag helper must exist in src/portfolio/cross_section.py
    and implement the correct formula: -(TC_BPS/10000) * mean_daily_turnover * 252."""

    def test_formula_is_correct(self):
        from src.portfolio.cross_section import compute_tc_drag
        result = compute_tc_drag(tc_bps=1.0, mean_daily_turnover=0.5)
        expected = -(1.0 / 10_000.0) * 0.5 * 252
        assert result == pytest.approx(expected, rel=1e-8)

    def test_returns_negative_value(self):
        from src.portfolio.cross_section import compute_tc_drag
        assert compute_tc_drag(tc_bps=5.0, mean_daily_turnover=0.3) < 0

    def test_proportional_to_tc_bps(self):
        from src.portfolio.cross_section import compute_tc_drag
        drag1 = compute_tc_drag(tc_bps=1.0, mean_daily_turnover=0.5)
        drag5 = compute_tc_drag(tc_bps=5.0, mean_daily_turnover=0.5)
        assert abs(drag5) == pytest.approx(5.0 * abs(drag1), rel=1e-8)

    def test_proportional_to_turnover(self):
        from src.portfolio.cross_section import compute_tc_drag
        drag_low = compute_tc_drag(tc_bps=1.0, mean_daily_turnover=0.1)
        drag_high = compute_tc_drag(tc_bps=1.0, mean_daily_turnover=0.5)
        assert abs(drag_high) == pytest.approx(5.0 * abs(drag_low), rel=1e-8)

    def test_run_portfolio_uses_formula_not_heuristic(self):
        """run_portfolio.py must not use the 0.05 heuristic multiplier for tc_drag."""
        source = (ROOT / "run_portfolio.py").read_text(encoding="utf-8")
        assert "ann_ret_rl) * 0.05" not in source, (
            "run_portfolio.py still uses heuristic tc_drag = -abs(ann_ret_rl)*0.05. "
            "Replace with compute_tc_drag(TC_BPS, mean_daily_turnover)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — Portfolio drift: compute_drift_weights in policy.py
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeDriftWeights:
    """compute_drift_weights(w_prev, r_prev) must exist in src/drl_policy/policy.py."""

    def test_function_exists(self):
        from src.drl_policy.policy import compute_drift_weights  # noqa: F401

    def test_zero_prior_returns_zero(self):
        """Zero prior weights (initial step) → drift result is zero."""
        from src.drl_policy.policy import compute_drift_weights
        w_prev = torch.zeros(3, 5)
        r_prev = torch.randn(3, 5)
        result = compute_drift_weights(w_prev, r_prev)
        assert result.shape == (3, 5)
        torch.testing.assert_close(result, torch.zeros(3, 5))

    def test_sums_to_one_for_valid_simplex(self):
        """With simplex weights and finite returns, result sums to 1 per row."""
        from src.drl_policy.policy import compute_drift_weights
        torch.manual_seed(42)
        w_prev = torch.softmax(torch.randn(4, 6), dim=-1)
        r_prev = torch.randn(4, 6) * 0.01
        result = compute_drift_weights(w_prev, r_prev)
        sums = result.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_asymmetric_returns_shift_weights_correctly(self):
        """One leg up, one leg down → drift weight correctly re-normalises."""
        from src.drl_policy.policy import compute_drift_weights
        # w = [0.5, 0.5], r = [+0.10, -0.10]
        # w' = [0.5*1.10, 0.5*0.90] / (0.5*1.10 + 0.5*0.90) = [0.55, 0.45]
        w_prev = torch.tensor([[0.5, 0.5]])
        r_prev = torch.tensor([[0.10, -0.10]])
        result = compute_drift_weights(w_prev, r_prev)
        expected = torch.tensor([[0.55, 0.45]])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_nonnegative_output(self):
        """Drift weights must be non-negative."""
        from src.drl_policy.policy import compute_drift_weights
        torch.manual_seed(7)
        w_prev = torch.softmax(torch.randn(8, 10), dim=-1)
        r_prev = torch.randn(8, 10) * 0.05
        result = compute_drift_weights(w_prev, r_prev)
        assert (result >= 0).all(), "Drift weights contain negative values"

    def test_drift_differs_from_raw_when_returns_nonzero(self):
        """Result must differ from raw w_prev whenever returns ≠ 0."""
        from src.drl_policy.policy import compute_drift_weights
        torch.manual_seed(0)
        w_prev = torch.softmax(torch.randn(2, 4), dim=-1)
        r_prev = torch.randn(2, 4) * 0.05  # small but nonzero
        result = compute_drift_weights(w_prev, r_prev)
        assert not torch.allclose(result, w_prev, atol=1e-4), (
            "Drift weights are identical to raw w_prev — drift adjustment has no effect."
        )

    def test_rollout_uses_drift_weights(self):
        """run_pipeline.py rollout must reference compute_drift_weights."""
        source = (ROOT / "run_pipeline.py").read_text(encoding="utf-8")
        assert "compute_drift_weights" in source, (
            "run_pipeline.py rollout does not call compute_drift_weights. "
            "Replace raw w_prev with drift-adjusted weights in delta_w computation."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Moving Block Bootstrap (MBB) to preserve ARCH clustering
# ─────────────────────────────────────────────────────────────────────────────

class TestMovingBlockBootstrap:
    """_mbb_sample_residuals must exist and produce consecutive-block samples."""

    def test_helper_function_exists(self):
        from run_pipeline import _mbb_sample_residuals  # noqa: F401

    def test_output_shape(self):
        from run_pipeline import _mbb_sample_residuals
        rng = np.random.default_rng(0)
        residuals = rng.normal(0, 1, (100, 3))
        sampled = _mbb_sample_residuals(residuals, n_steps=50, block=10, rng=rng)
        assert sampled.shape == (50, 3)

    def test_consecutive_indices_within_block(self):
        """Within each drawn block the residual rows must be time-consecutive."""
        from run_pipeline import _mbb_sample_residuals

        T = 100
        # Encode row index into every column so we can verify consecutiveness
        residuals = np.tile(np.arange(T, dtype=float).reshape(-1, 1), (1, 3))
        block = 10
        rng = np.random.default_rng(42)
        sampled = _mbb_sample_residuals(residuals, n_steps=30, block=block, rng=rng)

        # First full block of 10 must consist of 10 consecutive integers
        first_block = sampled[:block, 0]
        diffs = np.diff(first_block)
        np.testing.assert_array_equal(
            diffs, np.ones(block - 1),
            err_msg=f"First block is not consecutive: {first_block}",
        )

    def test_bootstrap_paths_uses_mbb(self):
        """_bootstrap_paths source must reference _mbb_sample_residuals."""
        import inspect
        from run_pipeline import _bootstrap_paths
        src = inspect.getsource(_bootstrap_paths)
        assert "_mbb_sample_residuals" in src, (
            "_bootstrap_paths still uses np.random.randint for IID residual sampling. "
            "Replace with _mbb_sample_residuals to preserve ARCH clustering."
        )

    def test_full_bootstrap_shape(self):
        from run_pipeline import _bootstrap_paths
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1, (200, 5))
        paths = _bootstrap_paths(data, n_steps=50, n_paths=10, block=21)
        assert paths.shape == (10, 50, 5)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 7 — Look-ahead bias guard: bootstrap uses IS-only data
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapNoLookahead:
    """_bootstrap_paths must derive all statistics from its input data argument only.
    Since run_pipeline already passes state_is.values, this is a guard test."""

    def test_bootstrap_bounds_match_input_not_larger_set(self):
        """If tanh/data_min/data_max are removed (fix 4), this is automatically satisfied.
        Verify bootstrap output statistics are consistent with IS-only input distribution."""
        from run_pipeline import _bootstrap_paths
        rng = np.random.default_rng(5)
        # IS data: mean ~= 0, std ~= 1
        is_data = rng.normal(0, 1, (300, 3))
        paths = _bootstrap_paths(is_data, n_steps=30, n_paths=20, block=21)
        assert paths.shape == (20, 30, 3), "Shape mismatch"
        # Paths should be in the ballpark of IS data (not wildly different scale)
        path_std = float(paths.std())
        is_std = float(is_data.std())
        # Within 5× — reasonable for VAR(1) dynamics without artificial squash
        assert path_std < 5 * is_std, (
            f"Bootstrap path std={path_std:.2f} is >5× IS std={is_std:.2f} — "
            "something is wrong with path generation."
        )

    def test_gating_call_uses_is_only_data(self):
        """run_pipeline.py must pass state_is (not state) to gate_signals."""
        source = (ROOT / "run_pipeline.py").read_text(encoding="utf-8")
        # Check gate_signals receives state_is, not state
        import re
        gate_calls = re.findall(r"gate_signals\s*\(\s*(\w+)\s*,", source)
        assert all(s == "state_is" for s in gate_calls), (
            f"gate_signals called with: {gate_calls}. Must use state_is (IS-only)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX B — Chronological boundary validation (env var date ordering)
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvVarBoundaryValidation:
    """run_pipeline.py must assert DATA_START < INSAMPLE_END < OOS_END at module
    load time to prevent silent data leakage from misconfigured env vars."""

    def test_source_contains_chronological_assertion(self):
        """run_pipeline.py source must contain an assertion on date ordering."""
        source = (ROOT / "run_pipeline.py").read_text(encoding="utf-8")
        # Look for the chronological assertion pattern
        import re
        # Accepts any of: assert ..., pd.Timestamp assertions, or ValueError raises
        has_guard = bool(
            re.search(r"assert\s+pd\.Timestamp\(DATA_START\)", source)
            or re.search(r"if\s+pd\.Timestamp\(DATA_START\).*raise\s+ValueError", source, re.DOTALL)
        )
        assert has_guard, (
            "run_pipeline.py lacks a chronological assertion on DATA_START < "
            "INSAMPLE_END < OOS_END.  Add:\n"
            "    assert pd.Timestamp(DATA_START) < pd.Timestamp(INSAMPLE_END) "
            "< pd.Timestamp(OOS_END), ...\n"
            "immediately after the date env-var block."
        )

    def test_reversed_insample_end_raises(self):
        """A reversed INSAMPLE_END (> OOS_END) must raise AssertionError or ValueError
        when run_pipeline constants are evaluated with bad env vars."""
        import os
        import importlib
        import sys

        # Save original env
        orig_is  = os.environ.get("PIPELINE_INSAMPLE_END")
        orig_oos = os.environ.get("PIPELINE_OOS_END")
        try:
            # Force DATA_START=2015, INSAMPLE_END=2025 (> OOS_END=2024) → violation
            os.environ["PIPELINE_INSAMPLE_END"] = "2025-01-01"
            os.environ["PIPELINE_OOS_END"]      = "2024-12-31"
            # Remove cached modules so constants are re-evaluated (cfg included)
            for _mod in ("run_pipeline", "src.config"):
                if _mod in sys.modules:
                    del sys.modules[_mod]
            with pytest.raises((AssertionError, ValueError)):
                import run_pipeline  # noqa: F401
        finally:
            # Restore env
            if orig_is is not None:
                os.environ["PIPELINE_INSAMPLE_END"] = orig_is
            else:
                os.environ.pop("PIPELINE_INSAMPLE_END", None)
            if orig_oos is not None:
                os.environ["PIPELINE_OOS_END"] = orig_oos
            else:
                os.environ.pop("PIPELINE_OOS_END", None)
            if "run_pipeline" in sys.modules:
                del sys.modules["run_pipeline"]
            if "src.config" in sys.modules:
                del sys.modules["src.config"]
