"""
Integration tests against a live WRDS connection.

Decorated with ``@pytest.mark.wrds`` — automatically skipped when
WRDS_USERNAME / WRDS_PASSWORD are absent from .env or the environment.

These tests validate that the real data pipeline satisfies the same
structural and domain constraints enforced by the synthetic-data unit tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.econometrics.data_ingestion import build_state_vector


# ---------------------------------------------------------------------------
# Raw data integrity
# ---------------------------------------------------------------------------

@pytest.mark.wrds
class TestWRDSDataIntegrity:
    def test_prices_is_series(self, wrds_data):
        assert isinstance(wrds_data["prices"], pd.Series)

    def test_prices_length_approx_2y(self, wrds_data):
        # ~504 business days in 2 years; allow ±10 % for holidays / gaps
        n = len(wrds_data["prices"])
        assert 450 <= n <= 560, f"Expected ~504 trading days, got {n}"

    def test_prices_positive(self, wrds_data):
        assert (wrds_data["prices"] > 0).all()

    def test_option_panel_required_columns(self, wrds_data):
        required = {"date", "iv_30", "iv_91", "skew_25d"}
        assert required.issubset(wrds_data["option_panel"].columns)

    def test_option_panel_has_skew_10d(self, wrds_data):
        """skew_10d must be present after SQL upgrade to delta IN (10,25,50,75,90)."""
        assert "skew_10d" in wrds_data["option_panel"].columns, (
            "skew_10d missing — delete option_panel cache and re-fetch "
            "(delta=10/90 were added to the vsurfd SQL)"
        )
        panel = wrds_data["option_panel"]
        assert panel["skew_10d"].mean() > 0, (
            "skew_10d mean should be positive (IV(delta=90) > IV(delta=10))"
        )

    def test_iv_in_decimal_not_percentage(self, wrds_data):
        panel = wrds_data["option_panel"]
        # Decimal IVs typically 0.05 – 0.80; percentage IVs would be 5 – 80
        assert panel["iv_30"].mean() < 2.0, "iv_30 appears to be in percentage form"
        assert panel["iv_91"].mean() < 2.0, "iv_91 appears to be in percentage form"

    def test_vix_in_decimal_not_percentage(self, wrds_data):
        vix = wrds_data["vix"]
        assert vix.mean() < 1.0, f"VIX mean {vix.mean():.3f} — expected decimal form"

    def test_treasury_in_decimal_not_percentage(self, wrds_data):
        tr = wrds_data["treasury_10y"]
        assert tr.mean() < 0.20, f"10Y yield mean {tr.mean():.4f} — expected decimal form"

    def test_option_panel_no_duplicate_dates(self, wrds_data):
        panel = wrds_data["option_panel"].set_index("date")
        dupes = panel.index.duplicated().sum()
        assert dupes == 0, f"option_panel has {dupes} duplicate date(s)"

    def test_option_panel_no_all_nan_rows(self, wrds_data):
        panel = wrds_data["option_panel"].set_index("date")
        all_nan = panel.isnull().all(axis=1).sum()
        assert all_nan == 0, f"{all_nan} rows are entirely NaN"


# ---------------------------------------------------------------------------
# build_state_vector on real data
# ---------------------------------------------------------------------------

@pytest.mark.wrds
class TestBuildStateVectorOnRealData:
    def test_builds_without_error(self, wrds_data):
        state, fwd = build_state_vector(
            option_panel=wrds_data["option_panel"],
            underlying_prices=wrds_data["prices"],
            vix=wrds_data["vix"],
            treasury_10y=wrds_data["treasury_10y"],
        )
        assert isinstance(state, pd.DataFrame)
        assert isinstance(fwd, pd.Series)

    def test_state_columns_complete(self, wrds_data):
        expected = {
            "iv_30", "iv_91", "term_structure", "skew_25d",
            "rv_21", "rv_63", "vrp", "vix", "yield_10y",
        }
        state, _ = build_state_vector(
            option_panel=wrds_data["option_panel"],
            underlying_prices=wrds_data["prices"],
            vix=wrds_data["vix"],
            treasury_10y=wrds_data["treasury_10y"],
        )
        assert expected.issubset(state.columns)

    def test_no_lookahead_bias(self, wrds_data):
        """Last date in the state must precede the last available price date."""
        state, _ = build_state_vector(
            option_panel=wrds_data["option_panel"],
            underlying_prices=wrds_data["prices"],
            vix=wrds_data["vix"],
            treasury_10y=wrds_data["treasury_10y"],
        )
        assert state.index[-1] < wrds_data["prices"].index[-1]

    def test_state_length_after_dropna(self, wrds_data):
        """rv_21 requires 21 days of burn-in; at least 420 rows must survive."""
        state, _ = build_state_vector(
            option_panel=wrds_data["option_panel"],
            underlying_prices=wrds_data["prices"],
            vix=wrds_data["vix"],
            treasury_10y=wrds_data["treasury_10y"],
        )
        assert len(state) >= 420, f"Only {len(state)} rows survived dropna"

    def test_vrp_sign_convention(self, wrds_data):
        """
        Variance risk premium (IV_30 − RV_21) is positive on average in
        equity markets.  A strongly negative mean indicates a sign error.
        """
        state, _ = build_state_vector(
            option_panel=wrds_data["option_panel"],
            underlying_prices=wrds_data["prices"],
            vix=wrds_data["vix"],
            treasury_10y=wrds_data["treasury_10y"],
        )
        assert state["vrp"].mean() > -0.05, (
            f"VRP mean {state['vrp'].mean():.4f} — check IV/RV unit conventions"
        )

    def test_forward_returns_not_all_zero(self, wrds_data):
        _, fwd = build_state_vector(
            option_panel=wrds_data["option_panel"],
            underlying_prices=wrds_data["prices"],
            vix=wrds_data["vix"],
            treasury_10y=wrds_data["treasury_10y"],
        )
        assert fwd.abs().sum() > 0
