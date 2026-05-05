"""
Phase I — Data Ingestion: predictor state vector synthesis.

Compiles the deterministic state tensor s_t from SPX option panels,
realized variance, VIX/VXO, and 10-year Treasury yield data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_state_vector(
    option_panel: pd.DataFrame,
    underlying_prices: pd.Series,
    vix: pd.Series,
    treasury_10y: pd.Series,
    vxo: pd.Series | None = None,
    log_pcr: pd.Series | None = None,
    zerocd: "pd.DataFrame | None" = None,
    iv_dispersion: pd.Series | None = None,
    vxn: pd.Series | None = None,
    vxd: pd.Series | None = None,
    rv_dispersion: pd.Series | None = None,
    constituent_iv: pd.Series | None = None,
    constituent_skew: pd.Series | None = None,
) -> "tuple[pd.DataFrame, pd.Series]":
    """Return the aligned state tensor and the forward return series.

    Parameters
    ----------
    option_panel:
        Daily SPX option panel with columns ['date', 'iv_30', 'iv_91',
        'skew_25d'].  All implied volatilities are in decimal form.
    underlying_prices:
        Daily closing prices of the underlying (SPX).  Index is date.
    vix:
        Daily VIX index values (decimal, e.g. 0.20 for 20%).  Index is date.
    treasury_10y:
        Daily 10-year U.S. Treasury yield in decimal form.  Index is date.
    vxo:
        Daily VXO index values.  Required for pre-2003 data where OptionMetrics
        suffers from the 3:59 PM vs 4:00 PM timestamp corruption.
    log_pcr:
        Daily log put/call volume ratio from OptionMetrics opvold.
        Positive = more put than call volume.  Index is date.
    zerocd:
        DataFrame with columns ['short_rate', 'curve_slope'] from
        fetch_zerocd_local().  Both in decimal form.  Index is date.
    iv_dispersion:
        Daily cross-sectional coefficient of variation of constituent 30-day
        ATM implied vols (std / mean across S&P 500 members).  Index is date.
        Covers 2003-2021; NaN for later dates are forward-filled.
    vxn:
        NASDAQ-100 implied volatility index (decimal, e.g. 0.42 = 42%).
        From ``fetch_cboe_multivol_local()``.  Used to compute the VXN-VIX
        log spread — a proxy for the tech-sector risk premium above the
        broad market.
    vxd:
        DJIA implied volatility index (decimal).  Used to compute the VXD-VIX
        log spread — blue-chip idiosyncratic stress signal.
    rv_dispersion:
        Daily cross-sectional CV of constituent 30-day OptionMetrics historical
        vol (std / mean across S&P 500 secids per date).  Complements
        ``iv_dispersion`` — their difference approximates the cross-sectional
        dispersion variance risk premium.
    constituent_iv:
        Daily cross-sectional mean of per-constituent ATM 30d call IV (decimal).
        Derived from raw OptionMetrics opprcd data (2023-2024); IS values are
        back-filled by ``impute_constituent_iv_for_is`` using vix × OOS ratio.
        When present, adds the ``constituent_iv`` state column.
    constituent_skew:
        Daily cross-sectional mean of per-constituent 25-delta put-call IV skew.
        Available for 2023-2024 only; forward-filled in the OOS window.
        When present, adds the ``constituent_skew`` state column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (state_df, forward_returns) — state_df contains the predictor matrix
        s_t; forward_returns is R_{t+1} = (S_{t+1} - S_t) / S_t, strictly
        separated from the state tensor to prevent look-ahead bias.

    State vector composition
    ------------------------
    The 9 core features specified in the thesis proposal are:
        iv_30, iv_91, term_structure, skew_25d, rv_21, rv_63, vrp, vix, yield_10y

    When optional data sources are available, up to 9 extension features are
    added beyond the proposal specification (explicit extensions, not deviations):
        skew_10d           — 10-delta OTM put/call skew; tail-risk premium proxy
        log_pcr            — log(put/call volume ratio); market sentiment signal
        short_rate         — OptionMetrics 30-day zero-coupon rate; short-end term structure
        iv_dispersion      — cross-sectional CV of constituent ATM IVs; fear index proxy
        vxn_vix_spread     — log(VXN/VIX); NASDAQ tech risk premium vs broad market
        vxd_vix_spread     — log(VXD/VIX); DJIA blue-chip concentration risk
        rv_dispersion      — cross-sectional CV of constituent 30d realized vol
        constituent_iv     — cross-sectional mean constituent ATM 30d IV (2023-2024 actual,
                             IS imputed via VIX × OOS ratio)
        constituent_skew   — cross-sectional mean constituent 25d put-call skew (2023-2024)

    When none of the optional sources are provided the function returns the
    original 9-feature state consistent with the proposal specification.
    When all sources are available the state has up to 18 features.
    """
    panel = option_panel.set_index("date") if "date" in option_panel.columns else option_panel.copy()

    iv_30: pd.Series = panel["iv_30"]
    iv_91: pd.Series = panel["iv_91"]
    term_structure: pd.Series = iv_91 - iv_30
    skew_25d: pd.Series = panel["skew_25d"]

    rv_21 = _rolling_realized_variance(underlying_prices, window=21)
    rv_63 = _rolling_realized_variance(underlying_prices, window=63)

    vrp = _volatility_risk_premium(iv_30, rv_21)

    vix_aligned = _align_vix(vix, vxo)

    state_dict: dict[str, pd.Series] = {
        "iv_30":          iv_30,
        "iv_91":          iv_91,
        "term_structure": term_structure,
        "skew_25d":       skew_25d,
        "rv_21":          rv_21,
        "rv_63":          rv_63,
        "vrp":            vrp,
        "vix":            vix_aligned,
        "yield_10y":      treasury_10y,
    }

    # skew_10d: tail-risk premium (10-delta OTM put skew vs 10-delta OTM call)
    if "skew_10d" in panel.columns:
        state_dict["skew_10d"] = panel["skew_10d"]

    # ── Optional enrichment features ─────────────────────────────────────────
    if log_pcr is not None and len(log_pcr) > 0:
        state_dict["log_pcr"] = log_pcr

    if zerocd is not None and not zerocd.empty:
        if "short_rate" in zerocd.columns:
            state_dict["short_rate"] = zerocd["short_rate"]

    if iv_dispersion is not None and len(iv_dispersion) > 0:
        state_dict["iv_dispersion"] = iv_dispersion

    # ── Cross-index vol spreads (VXN/VIX, VXD/VIX) ───────────────────────────
    if vxn is not None and len(vxn) > 0:
        vxn_dedup = vxn.loc[~vxn.index.duplicated(keep="last")]
        vxn_aligned = vxn_dedup.reindex(vix_aligned.index).ffill()
        vix_safe = vix_aligned.clip(lower=1e-6)
        state_dict["vxn_vix_spread"] = np.log(
            vxn_aligned.clip(lower=1e-6) / vix_safe
        ).rename("vxn_vix_spread")

    if vxd is not None and len(vxd) > 0:
        vxd_dedup = vxd.loc[~vxd.index.duplicated(keep="last")]
        vxd_aligned = vxd_dedup.reindex(vix_aligned.index).ffill()
        vix_safe = vix_aligned.clip(lower=1e-6)
        state_dict["vxd_vix_spread"] = np.log(
            vxd_aligned.clip(lower=1e-6) / vix_safe
        ).rename("vxd_vix_spread")

    # ── Cross-sectional realized-vol dispersion ───────────────────────────────
    if rv_dispersion is not None and len(rv_dispersion) > 0:
        state_dict["rv_dispersion"] = rv_dispersion

    # ── Per-constituent option signals (2023-2024 actual; IS imputed via VIX) ─
    # IMPORTANT: reindex to the primary daily index BEFORE building the DataFrame.
    # If constituent_iv has a different (e.g., monthly/bi-weekly) frequency,
    # pd.DataFrame(state_dict) takes the UNION of all indices, causing the
    # ffill(limit=252) to count positions in the union rather than daily gaps,
    # and dropna() then eliminates all daily rows beyond 252 union-positions of
    # the last constituent observation.
    _primary_idx = iv_30.index  # the canonical daily index
    if constituent_iv is not None and len(constituent_iv) > 0:
        _civ = constituent_iv.reindex(_primary_idx).ffill(limit=252)
        if _civ.notna().sum() > 0:
            state_dict["constituent_iv"] = _civ

    if constituent_skew is not None and len(constituent_skew) > 0:
        _csk = constituent_skew.reindex(_primary_idx).ffill(limit=252)
        if _csk.notna().sum() > 0:
            state_dict["constituent_skew"] = _csk

    state = pd.DataFrame(state_dict)

    # Forward-fill sparsely-observed signals up to 252 trading days so the OOS
    # window retains a valid value where the source data ends.
    # Note: constituent_iv/skew are already ffill'd on the primary index above.
    for _disp_col in ("iv_dispersion", "rv_dispersion"):
        if _disp_col in state.columns:
            state[_disp_col] = state[_disp_col].ffill(limit=252)

    state = state.dropna()

    # Forward return: causal alignment — t+1 return aligned to state at t.
    fwd_returns: pd.Series = underlying_prices.pct_change().shift(-1).rename("R_t1")
    fwd_returns = fwd_returns.reindex(state.index).dropna()
    state = state.reindex(fwd_returns.index)

    # Ensure plain float64 — pandas 2.x nullable FloatingArray breaks numpy ops.
    state = state.astype("float64")
    fwd_returns = fwd_returns.astype("float64")

    return state, fwd_returns


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_realized_variance(prices: pd.Series, window: int) -> pd.Series:
    """Annualised rolling realised variance from daily log-returns."""
    log_returns = np.log(prices / prices.shift(1))
    rv = log_returns.rolling(window).var() * 252  # annualise
    return rv.rename(f"rv_{window}")


def _volatility_risk_premium(iv_30: pd.Series, rv_21: pd.Series) -> pd.Series:
    """VRP = IV_30 - E[RV].  Uses current RV_21 as proxy for expected RV."""
    return (iv_30 - rv_21).rename("vrp")


def _align_vix(vix: pd.Series, vxo: pd.Series | None) -> pd.Series:
    """Return a unified volatility index, substituting VXO pre-2003.

    OptionMetrics suffers from a known timestamp corruption (3:59 PM vs
    4:00 PM) for data before 2003.  VXO is used as the substitute index
    for all observations with date < 2003-01-01.
    """
    cutoff = pd.Timestamp("2003-01-01")

    if vxo is None:
        return vix.rename("vix")

    combined = vix.copy().rename("vix")
    pre_2003_mask = combined.index < cutoff
    combined.loc[pre_2003_mask] = vxo.reindex(combined.index[pre_2003_mask])
    return combined
