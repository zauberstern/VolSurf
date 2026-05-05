"""
Phase I — Econometric Signal Gating.

Implements Newey-West HAC covariance estimation and Benjamini-Yekutieli FDR
control.  Holm-Sidak FWER and Holm-Bonferroni are included as comparison
procedures and are exercised by the test suite.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as nla
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gate_signals(
    state: pd.DataFrame,
    forward_returns: pd.Series,
    half_spreads: pd.Series,
    alpha: float = 0.05,
    n_lags: int | None = None,
) -> pd.Series:
    """Return a boolean Series indicating which signals pass the FDR gate.

    For each predictor column i in *state*, tests H_0: mu_i <= c_bar_i
    (expected return does not exceed the bid-ask half-spread) using a
    Newey-West HAC t-statistic.  Multiple-testing correction is applied
    via the Benjamini-Yekutieli step-up FDR procedure, which is valid
    under arbitrary dependence between test statistics.

    Parameters
    ----------
    state:
        T x m predictor matrix s_t.
    forward_returns:
        T-length series R_{t+1}.
    half_spreads:
        m-length series of bid-ask half-spreads c_bar_i (one per predictor).
    alpha:
        False discovery rate level (default 0.05).
    n_lags:
        Newey-West lag truncation L.  Defaults to floor(0.75·T^(1/3)).

    Returns
    -------
    pd.Series[bool]
        Index = predictor names; True means the signal passed (reject H_0).
    """
    T, m = state.shape
    if n_lags is None:
        # Andrews (1991) data-adaptive lag truncation: floor(0.75·T^(1/3))
        n_lags = max(1, int(np.floor(0.75 * T ** (1 / 3))))

    p_values: dict[str, float] = {}
    for col in state.columns:
        excess = forward_returns.values - half_spreads[col]
        t_stat, p_val = _hac_one_sided_ttest(excess, n_lags)
        p_values[col] = p_val

    p_series = pd.Series(p_values)
    rejected = _benjamini_yekutieli(p_series, alpha)
    return rejected


# ---------------------------------------------------------------------------
# HAC covariance / t-statistic
# ---------------------------------------------------------------------------

def _hac_one_sided_ttest(x: np.ndarray, n_lags: int) -> tuple[float, float]:
    """One-sided t-test (H_0: mu <= 0) with Newey-West HAC standard error.

    Returns (t_statistic, p_value) where p_value is for the right tail.
    """
    T = len(x)
    x = np.asarray(x, dtype=np.float64)  # ensure numpy — handles pandas FloatingArray
    mu_hat = x.mean()
    hac_var = _newey_west_variance(x, n_lags)
    se = np.sqrt(hac_var / T)
    t_stat = mu_hat / se if se > 0 else 0.0
    p_val = 1.0 - stats.norm.cdf(t_stat)  # one-sided right tail
    return float(t_stat), float(p_val)


def _newey_west_variance(x: np.ndarray, L: int) -> float:
    """Scalar Newey-West HAC variance estimate.

    Weights: w_l = 1 - l / (L + 1)  for l = 0, ..., L.
    """
    T = len(x)
    # Ensure plain numpy array — pandas FloatingArray doesn't support @
    x = np.asarray(x, dtype=np.float64)
    x_demeaned = x - x.mean()
    gamma_0 = float(x_demeaned @ x_demeaned) / T
    nw_var = gamma_0
    for l in range(1, L + 1):
        w_l = 1.0 - l / (L + 1)
        gamma_l = float(x_demeaned[l:] @ x_demeaned[:-l]) / T
        nw_var += 2.0 * w_l * gamma_l
    # Clamp at 1e-8 to prevent t-stat explosion in low-volatility regimes.
    return max(nw_var, 1e-8)


# ---------------------------------------------------------------------------
# Multiple testing: Holm-Sidak with Holm-Bonferroni fallback
# ---------------------------------------------------------------------------

def _holm_sidak_fwer(p_values: pd.Series, alpha: float) -> pd.Series:
    """Holm-Sidak step-down FWER procedure.

    Falls back to Holm-Bonferroni if covariance structure is ill-conditioned
    (caught via numpy.linalg.LinAlgError or non-positive-definite check).

    Returns a boolean Series — True means the null is rejected (signal passes).
    """
    try:
        return _holm_sidak(p_values, alpha)
    except (nla.LinAlgError, ValueError):
        return _holm_bonferroni(p_values, alpha)


def _holm_sidak(p_values: pd.Series, alpha: float) -> pd.Series:
    """Holm-Sidak step-down procedure.

    Raises ValueError if any threshold computation fails (signals non-positive
    definite dependency structure, triggering Bonferroni fallback).
    """
    m = len(p_values)
    sorted_idx = p_values.argsort(kind='stable')
    sorted_p = p_values.iloc[sorted_idx].values

    rejected = np.zeros(m, dtype=bool)
    for k in range(m):
        threshold = 1.0 - (1.0 - alpha) ** (1.0 / (m - k))
        if threshold <= 0 or not np.isfinite(threshold):
            raise ValueError("Non-finite Holm-Sidak threshold; switching to Bonferroni.")
        if sorted_p[k] <= threshold:
            rejected[k] = True
        else:
            break  # step-down: stop at first non-rejection

    result = pd.Series(False, index=p_values.index)
    result.iloc[sorted_idx[rejected]] = True
    return result


def _holm_bonferroni(p_values: pd.Series, alpha: float) -> pd.Series:
    """Holm-Bonferroni step-down fallback."""
    m = len(p_values)
    sorted_idx = p_values.argsort(kind='stable')
    sorted_p = p_values.iloc[sorted_idx].values

    rejected = np.zeros(m, dtype=bool)
    for k in range(m):
        threshold = alpha / (m - k)
        if sorted_p[k] <= threshold:
            rejected[k] = True
        else:
            break

    result = pd.Series(False, index=p_values.index)
    result.iloc[sorted_idx[rejected]] = True
    return result


# ---------------------------------------------------------------------------
# Benjamini-Yekutieli FDR: valid under arbitrary dependence (BY 2001)
# ---------------------------------------------------------------------------

def _benjamini_yekutieli(p_values: pd.Series, alpha: float) -> pd.Series:
    """Benjamini-Yekutieli step-up FDR procedure.

    Controls the false discovery rate at level *alpha* under arbitrary
    (including positive/negative) dependence between test statistics.
    The BY procedure is strictly more powerful than FWER control (Holm,
    Bonferroni) at the cost of allowing up to alpha×m expected false
    discoveries rather than zero.

    Threshold for the k-th smallest p-value:
        p_{(k)} <= alpha * k / (m * c_m)
    where c_m = sum_{i=1}^{m} 1/i  (harmonic number, BY 2001 eq. 3).

    Parameters
    ----------
    p_values:
        pd.Series of p-values with predictor names as index.
    alpha:
        Target FDR level.

    Returns
    -------
    pd.Series[bool] — True where the null is rejected.
    """
    m = len(p_values)
    if m == 0:
        return pd.Series(dtype=bool)
    # Harmonic number c_m = 1 + 1/2 + ... + 1/m
    c_m = sum(1.0 / i for i in range(1, m + 1))

    sorted_idx = p_values.argsort(kind='stable')
    sorted_p = p_values.iloc[sorted_idx].values

    # Find the largest k such that p_{(k)} <= alpha * k / (m * c_m)
    last_reject = -1
    for k in range(m):
        threshold = alpha * (k + 1) / (m * c_m)
        if sorted_p[k] <= threshold:
            last_reject = k

    rejected_flags = np.zeros(m, dtype=bool)
    if last_reject >= 0:
        rejected_flags[: last_reject + 1] = True

    result = pd.Series(False, index=p_values.index)
    result.iloc[sorted_idx[rejected_flags]] = True
    return result
