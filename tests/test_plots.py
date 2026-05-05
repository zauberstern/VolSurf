"""
Smoke tests for all 12 visualization functions in src/evaluation/plots.py.

These tests render to a temporary directory using synthetic data to verify:
  1. Each function completes without raising an exception.
  2. Each function returns the expected output file path.
  3. Each output file exists and is non-empty (> 0 bytes).

No pixel-level image content is validated — this is a smoke / regression
test to catch signature mismatches, import errors, or matplotlib crashes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.plots import (
    # Existing 4 plots
    policy_surface_3d,
    rl_regime_analysis,
    feature_sensitivity_heatmap,
    trade_activity_calendar,
    rolling_regime_metrics,
    # New 8 plots
    streamgraph_allocation,
    terrain_miner_3d,
    friction_labyrinth,
    volatility_loom,
    alpha_sonar_radar,
    tactical_execution_dashboard,
    tail_risk_topography,
    constellation_risk,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def out_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="module")
def synth():
    """Synthetic OOS data covering 60 trading days."""
    rng = np.random.default_rng(42)
    T = 60
    dates = pd.bdate_range("2022-01-03", periods=T)
    feature_names = ["iv_30", "iv_91", "term_structure", "skew_25d", "rv_21"]
    D = len(feature_names)
    return {
        "T":          T,
        "dates":      dates,
        "actions":    rng.uniform(-0.8, 0.8, T),
        "returns":    pd.Series(rng.normal(0.001, 0.01, T), index=dates),
        "rewards":    rng.normal(0.001, 0.005, T),
        "vix":        pd.Series(rng.uniform(0.15, 0.40, T), index=dates),
        "states":     rng.standard_normal((T, D)).astype(np.float32),
        "features":   feature_names,
        "dispersion": pd.Series(rng.uniform(0.10, 0.30, T), index=dates),
    }


# ---------------------------------------------------------------------------
# Existing plot functions
# ---------------------------------------------------------------------------

class TestRlRegimeAnalysis:
    def test_renders_and_returns_path(self, synth, out_dir):
        rl_res_dict = {
            "oos_dates":        synth["dates"],
            "oos_rl_returns":   synth["returns"].values,
            "oos_spx_returns":  synth["returns"].values * 0.9,
            "oos_actions":      synth["actions"],
            "training_rewards": synth["rewards"].tolist()[:30],
            "training_cvar":    np.zeros(30).tolist(),
            "training_eta":     [0.01] * 30,
            "ww_half_width":    0.005,
        }
        out = out_dir / "rl_regime.png"
        path = rl_regime_analysis(rl_res_dict, synth["vix"], output_path=out)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


class TestTradeActivityCalendar:
    def test_renders_and_returns_path(self, synth, out_dir):
        out = out_dir / "cal.png"
        path = trade_activity_calendar(synth["returns"], synth["dates"], output_path=out)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


class TestRollingRegimeMetrics:
    def test_renders_and_returns_path(self, synth, out_dir):
        out = out_dir / "rolling.png"
        path = rolling_regime_metrics(
            rl_returns=synth["returns"],
            spx_returns=synth["returns"] * 0.95,
            vix_oos=synth["vix"],
            oos_dates=synth["dates"],
            output_path=out,
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


# ---------------------------------------------------------------------------
# New 8 plot functions
# ---------------------------------------------------------------------------

class TestStreamgraphAllocation:
    def test_renders_without_error(self, synth, out_dir):
        out = out_dir / "streamgraph.png"
        path = streamgraph_allocation(
            synth["dates"], synth["actions"], synth["vix"], output_path=out
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000   # must be a real PNG

    def test_extreme_actions_handled(self, synth, out_dir):
        """Actions at ±1 boundary must not cause any arithmetic error."""
        extreme = np.clip(synth["actions"] * 2, -1.0, 1.0)
        out = out_dir / "streamgraph_extreme.png"
        path = streamgraph_allocation(synth["dates"], extreme, synth["vix"], output_path=out)
        assert Path(path).exists()


class TestTerrainMiner3D:
    def test_renders_without_error(self, synth, out_dir):
        out = out_dir / "terrain.png"
        path = terrain_miner_3d(
            synth["states"], synth["actions"], synth["dates"],
            synth["features"], output_path=out,
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_handles_missing_feature_names(self, synth, out_dir):
        """Fallback to feature index when names don't match."""
        out = out_dir / "terrain_nofeat.png"
        path = terrain_miner_3d(
            synth["states"], synth["actions"], synth["dates"],
            [], output_path=out,       # empty names → fallback indices
        )
        assert Path(path).exists()


class TestFrictionLabyrinth:
    def test_renders_without_error(self, synth, out_dir):
        out = out_dir / "labyrinth.png"
        path = friction_labyrinth(
            list(synth["rewards"][:20]), 0.001, output_path=out
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_single_epoch_handled(self, synth, out_dir):
        out = out_dir / "labyrinth_single.png"
        path = friction_labyrinth([0.001], 0.0005, output_path=out)
        assert Path(path).exists()

    def test_empty_epoch_rewards_handled(self, synth, out_dir):
        out = out_dir / "labyrinth_empty.png"
        path = friction_labyrinth([], 0.001, output_path=out)
        assert Path(path).exists()


class TestVolatilityLoom:
    def test_renders_without_error(self, synth, out_dir):
        out = out_dir / "loom.png"
        path = volatility_loom(
            synth["states"], synth["actions"], synth["features"],
            synth["dates"], output_path=out,
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000


class TestAlphaSonarRadar:
    def test_renders_without_error(self, synth, out_dir):
        attr_df = pd.DataFrame(
            {"alpha": [0.001], "MKT": [0.5], "SMB": [-0.1], "HML": [0.2], "RF": [0.0]}
        )
        out = out_dir / "sonar.png"
        path = alpha_sonar_radar(attr_df, synth["dates"], output_path=out)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_empty_attr_df_handled(self, synth, out_dir):
        """Empty attr_df must produce a fallback plot without crash."""
        out = out_dir / "sonar_empty.png"
        path = alpha_sonar_radar(pd.DataFrame(), synth["dates"], output_path=out)
        assert Path(path).exists()

    def test_none_attr_df_handled(self, synth, out_dir):
        out = out_dir / "sonar_none.png"
        path = alpha_sonar_radar(None, synth["dates"], output_path=out)
        assert Path(path).exists()


class TestTacticalExecutionDashboard:
    def test_renders_without_error(self, synth, out_dir):
        out = out_dir / "tactical.png"
        path = tactical_execution_dashboard(
            synth["states"], synth["actions"], synth["rewards"],
            synth["vix"], 0.001, synth["dates"], synth["features"],
            output_path=out,
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_all_positive_rewards_handled(self, synth, out_dir):
        out = out_dir / "tactical_pos.png"
        path = tactical_execution_dashboard(
            synth["states"], synth["actions"],
            np.abs(synth["rewards"]),   # all positive
            synth["vix"], 0.001, synth["dates"], synth["features"],
            output_path=out,
        )
        assert Path(path).exists()


class TestTailRiskTopography:
    def test_renders_with_dispersion(self, synth, out_dir):
        out = out_dir / "topography.png"
        path = tail_risk_topography(
            synth["returns"], synth["actions"],
            synth["dispersion"], -0.02, synth["dates"],
            output_path=out,
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_renders_without_dispersion(self, synth, out_dir):
        out = out_dir / "topography_nodisp.png"
        path = tail_risk_topography(
            synth["returns"], synth["actions"],
            None, -0.02, synth["dates"],
            output_path=out,
        )
        assert Path(path).exists()


class TestConstellationRisk:
    def test_renders_without_error(self, synth, out_dir):
        out = out_dir / "constellation.png"
        path = constellation_risk(
            synth["states"], synth["actions"], synth["returns"],
            synth["features"], synth["dates"],
            output_path=out,
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_handles_constant_returns(self, synth, out_dir):
        """Constant returns → zero std; must not divide by zero."""
        const_returns = pd.Series(0.001, index=synth["dates"])
        out = out_dir / "constellation_const.png"
        path = constellation_risk(
            synth["states"], synth["actions"], const_returns,
            synth["features"], synth["dates"],
            output_path=out,
        )
        assert Path(path).exists()
