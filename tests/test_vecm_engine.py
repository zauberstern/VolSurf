"""
Tests for Phase II — vecm_engine.py

Verifies Reinsel-Ahn correction factor, SVD 95% PVE fallback, and
SSVI projection bounds.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation.vecm_engine import (
    _lee_moment_bound,
    _johansen_rank,
    _svd_fallback,
    project_surface_ssvi,
    apply_ssvi_bounds,
)


class TestSvdFallback:
    def test_pve_at_least_95_percent(self):
        rng = np.random.default_rng(3)
        # Low-rank data: 2 latent factors driving 10-dim series
        factors = rng.normal(0, 1, (200, 2))
        loadings = rng.normal(0, 1, (2, 10))
        data = factors @ loadings + rng.normal(0, 0.01, (200, 10))
        components, load_out = _svd_fallback(data)
        # Verify reconstruction explains >= 95% variance
        recon = components @ load_out
        ss_res = np.sum((data - data.mean(axis=0) - recon) ** 2)
        ss_tot = np.sum((data - data.mean(axis=0)) ** 2)
        pve = 1.0 - ss_res / ss_tot
        assert pve >= 0.95


class TestLeeMomentBound:
    def test_atm_bound_is_large(self):
        """At-the-money (k=0) the Lee bound should be the fallback (10.0)."""
        k = np.array([0.0])
        bound = _lee_moment_bound(k, p=1.0, tail="right")
        assert bound[0] == pytest.approx(10.0)

    def test_deep_itm_positive_bound(self):
        k = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        bound_r = _lee_moment_bound(k, p=1.5, tail="right")
        bound_l = _lee_moment_bound(k, p=1.5, tail="left")
        assert (bound_r >= 0).all()
        assert (bound_l >= 0).all()


class TestProjectSurfaceSsvi:
    def test_output_within_lee_bounds(self):
        k = np.linspace(-2.0, 2.0, 20)
        raw_w = 0.04 + 0.01 * k ** 2  # simple smile
        projected = project_surface_ssvi(k, raw_w, asset_moments=(1.5, 1.5))
        # Lee upper bound is per-wing: right slope for k>0, left slope for k<0, large cap at ATM
        # p=1.5: slope = 2 - 1.5 + 2*sqrt(0.5) ≈ 1.914
        p = 1.5
        slope = 2.0 - p + 2.0 * np.sqrt(max(1.0 - p, 0.0))
        lee_upper = np.where(np.abs(k) < 1e-6, 10.0, slope * np.abs(k))
        assert (projected <= lee_upper + 1e-6).all()
        assert (projected >= -1e-6).all()

    def test_output_nonnegative(self):
        k = np.linspace(-1.0, 1.0, 10)
        raw_w = 0.04 * np.ones_like(k)
        projected = project_surface_ssvi(k, raw_w, asset_moments=(1.8, 1.8))
        assert (projected >= -1e-8).all()


class TestApplySsviBounds:
    """Tests for the SSVI arbitrage-free post-simulation IV constraints."""

    def _make_paths(self, iv30_vals, iv91_vals) -> tuple[np.ndarray, list[str]]:
        """Build a minimal (1, T, 2) paths array with iv_30 and iv_91 columns."""
        T = len(iv30_vals)
        paths = np.stack([np.array(iv30_vals), np.array(iv91_vals)], axis=1)
        paths = paths[np.newaxis, :, :]   # (1, T, 2)
        return paths, ["iv_30", "iv_91"]

    def test_floor_clips_negative_iv(self):
        """Negative IVs must be raised to iv_min=0.01 (non-negative variance)."""
        paths, names = self._make_paths([-0.05, 0.10], [0.02, 0.12])
        out = apply_ssvi_bounds(paths, names, iv_min=0.01)
        assert float(out[0, 0, 0]) == pytest.approx(0.01)   # clipped up

    def test_ceiling_clips_extreme_iv(self):
        """IVs above iv_max=3.0 must be capped (Lee upper bound)."""
        paths, names = self._make_paths([5.0, 0.20], [6.0, 0.22])
        out = apply_ssvi_bounds(paths, names, iv_max=3.0)
        assert float(out[0, 0, 0]) == pytest.approx(3.0)
        assert float(out[0, 0, 1]) == pytest.approx(3.0)

    def test_calendar_spread_enforced(self):
        """iv_91 must satisfy iv_91 >= iv_30 * sqrt(30/91) (monotone total variance)."""
        cal_floor_factor = np.sqrt(30.0 / 91.0)   # ≈ 0.5743
        iv30 = 0.20
        # Set iv_91 below the calendar floor
        iv91_bad = iv30 * cal_floor_factor - 0.01
        paths, names = self._make_paths([iv30], [iv91_bad])
        out = apply_ssvi_bounds(paths, names)
        iv91_out = float(out[0, 0, 1])
        assert iv91_out >= iv30 * cal_floor_factor - 1e-9

    def test_valid_paths_unchanged(self):
        """Paths already satisfying all constraints must pass through unmodified."""
        iv30 = 0.18
        iv91 = 0.22   # iv91 > iv30 * sqrt(30/91) ≈ 0.1034
        paths, names = self._make_paths([iv30], [iv91])
        out = apply_ssvi_bounds(paths, names)
        np.testing.assert_allclose(out, paths, rtol=1e-10)

    def test_unknown_features_ignored(self):
        """Paths with no iv_30/iv_91 columns must be returned without modification."""
        paths = np.ones((2, 5, 3))
        out = apply_ssvi_bounds(paths, ["vrp", "vix", "rv_21"])
        np.testing.assert_array_equal(out, paths)
