"""
Phase IV — Performance Integrity and Out-of-Sample Validation.

Implements:
  - Whalley-Wilmott asymptotic no-trade zone width
  - Newey-West HAC return attribution regression
  - Deflated Sharpe Ratio (Bailey & López de Prado 2014)
  - Profit Factor
  - Walk-Forward Efficiency
  - Monte Carlo permutation p-value
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.stats as _stats
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Whalley-Wilmott no-trade zone benchmark
# ---------------------------------------------------------------------------

def whalley_wilmott_width(
    c: float | np.ndarray,
    gamma: float | np.ndarray,
    S: float | np.ndarray,
    a: float,
) -> float | np.ndarray:
    """Compute the Whalley-Wilmott optimal no-trade zone half-width.

    Formula:  w = ( 3 * c * Gamma^2 * S / (2 * a) )^(1/3)

    Parameters
    ----------
    c:     Transaction cost rate (proportional).
    gamma: Option gamma (second derivative of option price w.r.t. S).
    S:     Current spot price of the underlying.
    a:     Risk aversion coefficient.

    Returns
    -------
    Scalar or array no-trade half-width w.
    """
    return (3.0 * c * gamma ** 2 * S / (2.0 * a)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Hansen-Hodrick attribution regression
# ---------------------------------------------------------------------------

def attribution_regression(
    rl_returns: pd.Series,
    market_returns: pd.Series,
    carry_returns: pd.Series,
    vol_returns: pd.Series,
    vrp_returns: pd.Series,
    n_lags: int | None = None,
) -> pd.DataFrame:
    """Return attribution regression results with Hansen-Hodrick SE.

    Model:
        r_RL = beta_0 + beta_m*r_M + beta_c*r_carry + beta_v*r_v
               + beta_VRP*r_VRP + u_t

    HAC standard errors are computed using the Hansen-Hodrick kernel
    (uniform lag window) to correct for overlapping data horizons and
    serial correlation.

    Parameters
    ----------
    rl_returns:     Out-of-sample policy returns.
    market_returns: Broad market factor returns.
    carry_returns:  Carry factor returns.
    vol_returns:    Volatility factor returns.
    vrp_returns:    VRP factor returns.
    n_lags:         Lag window for Hansen-Hodrick correction.
                    Defaults to floor(T^(1/3)).

    Returns
    -------
    pd.DataFrame with columns [coef, hac_se, t_stat, p_value] for each
    regressor plus the intercept (beta_0 = true alpha).
    """
    aligned = pd.concat(
        [rl_returns, market_returns, carry_returns, vol_returns, vrp_returns],
        axis=1,
    ).dropna()
    aligned.columns = ["r_RL", "r_M", "r_carry", "r_v", "r_VRP"]

    y = aligned["r_RL"].values
    X = sm.add_constant(aligned[["r_M", "r_carry", "r_v", "r_VRP"]].values)

    T = len(y)
    if n_lags is None:
        # Daily returns are non-overlapping; NW(L=1) corrects for day-1 serial correlation.
        n_lags = 1

    ols = sm.OLS(y, X).fit()
    hac = ols.get_robustcov_results(cov_type="hac", maxlags=n_lags, use_correction=True)

    labels = ["alpha (beta_0)", "beta_m", "beta_c", "beta_v", "beta_VRP"]
    result = pd.DataFrame(
        {
            "coef": hac.params,
            "hac_se": hac.bse,
            "t_stat": hac.tvalues,
            "p_value": hac.pvalues,
        },
        index=labels,
    )
    return result


def interpret_alpha(result: pd.DataFrame, alpha_level: float = 0.05) -> str:
    """Map alpha t-statistic to qualitative outcome label."""
    row = result.loc["alpha (beta_0)"]
    if row["p_value"] < alpha_level and row["coef"] > 0:
        return "Positive Alpha"
    if row["p_value"] < alpha_level and row["coef"] <= 0:
        return "Execution Edge Only"
    other_sig = (result.drop("alpha (beta_0)")["p_value"] < alpha_level).any()
    if other_sig:
        return "Factor Exposure"
    return "Null Result"


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio  (Bailey & López de Prado 2014)
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    sr: float,
    T: int,
    skewness: float,
    excess_kurtosis: float,
    n_trials: int = 1,
) -> float:
    """Deflated Sharpe Ratio.

    Adjusts the observed Sharpe ratio for non-normality (skewness, fat tails)
    and multiple-testing inflation (n_trials strategy candidates evaluated).

    Returns the probability P(SR > SR*) where SR* is the expected maximum
    Sharpe ratio achieved by random chance across n_trials independent trials.

    Parameters
    ----------
    sr:               Observed annualised Sharpe ratio.
    T:                Number of return observations used to estimate SR.
    skewness:         Sample skewness of the return series.
    excess_kurtosis:  Excess kurtosis (kurtosis − 3) of the return series.
    n_trials:         Number of independent strategy trials evaluated.

    Returns
    -------
    float in [0, 1].  Values > 0.95 pass the 5% significance threshold.

    References
    ----------
    Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
    Journal of Portfolio Management, 40(5), 94-107.
    """
    # Expected maximum SR under n_trials iid Normal draws
    # (Euler-Mascheroni approximation, López de Prado eq. 12)
    gamma_e = 0.5772156649  # Euler-Mascheroni constant
    if n_trials > 1:
        sr_star = (
            (1.0 - gamma_e) * _stats.norm.ppf(1.0 - 1.0 / n_trials)
            + gamma_e * _stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        ) / math.sqrt(T)
    else:
        sr_star = 0.0

    # Standard error of the SR estimator under non-normality (Mertens 2002)
    # Var(SR) = (1 − γ₃·SR + ((γ₄−1)/4)·SR²) / (T−1)
    kurtosis = excess_kurtosis + 3.0
    var_sr = (1.0 - skewness * sr + ((kurtosis - 1.0) / 4.0) * sr ** 2) / max(T - 1, 1)
    var_sr = max(var_sr, 1e-6)  # floor: prevents numerical blow-up for near-zero-vol portfolios

    z = (sr - sr_star) / math.sqrt(var_sr)
    z = max(min(z, 8.0), -8.0)  # clip to avoid cdf returning exactly 0 or 1
    return float(_stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# Profit Factor
# ---------------------------------------------------------------------------

def profit_factor(returns: pd.Series | np.ndarray) -> float:
    """Gross gains divided by gross losses.

    Institutional target: 1.75 – 3.0.
    Values > 4.0 are heavily scrutinised for look-ahead bias.

    Parameters
    ----------
    returns: Daily (or periodic) return series.

    Returns
    -------
    float.  Returns ``inf`` if there are no losing periods.
    Returns 0.0 if there are no winning periods.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0 or float(np.abs(r).sum()) == 0.0:
        return float("nan")  # no trades or flat-return period
    gross_gains  = float(r[r > 0].sum())
    gross_losses = float(abs(r[r < 0].sum()))
    if gross_losses == 0.0:
        return math.inf
    if gross_gains == 0.0:
        return 0.0
    return gross_gains / gross_losses


# ---------------------------------------------------------------------------
# Walk-Forward Efficiency
# ---------------------------------------------------------------------------

def walk_forward_efficiency(is_ann_return: float, oos_ann_return: float) -> float:
    """Walk-Forward Efficiency (WFE).

    WFE = OOS annualised return / IS annualised return.

    Institutional target: WFE > 0.50 (OOS retains at least half of IS edge).

    Parameters
    ----------
    is_ann_return:  In-sample annualised return.
    oos_ann_return: Out-of-sample annualised return.

    Returns
    -------
    float.  Returns ``nan`` when IS return is non-positive (undefined ratio).
    """
    if is_ann_return <= 0.0:
        return float("nan")
    return oos_ann_return / is_ann_return


# ---------------------------------------------------------------------------
# Information Ratio
# ---------------------------------------------------------------------------

def information_ratio(
    portfolio_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Annualised Information Ratio.

    IR = (annualised active return) / (annualised tracking error)
       = (mean(r_p − r_b) × T) / (std(r_p − r_b) × √T)

    Institutional target: > 0.38 (satisfactory), > 0.80 (highly consistent skill).

    Parameters
    ----------
    portfolio_returns:  Strategy daily returns.
    benchmark_returns:  Benchmark daily returns aligned to the same dates.
    periods_per_year:   Trading days per year (default 252).

    Returns
    -------
    float.  Returns ``nan`` when tracking error is zero.
    """
    rp = np.asarray(portfolio_returns, dtype=float)
    rb = np.asarray(benchmark_returns, dtype=float)
    active = rp - rb
    # Clamp tracking error to 1e-8 so IR is always finite for constant-alpha strategies.
    te = max(float(active.std() * np.sqrt(periods_per_year)), 1e-8)
    active_return = float(active.mean() * periods_per_year)
    return active_return / te


# ---------------------------------------------------------------------------
# Monte Carlo Permutation P-value
# ---------------------------------------------------------------------------

def mc_permutation_pvalue(
    returns: pd.Series | np.ndarray,
    n_trials: int = 1000,
    seed: int = 42,
    block_size: int | None = None,
) -> float:
    """Moving-block bootstrap p-value for the strategy's observed Sharpe ratio.

    Null hypothesis H₀: the observed Sharpe ratio is no better than what
    can be obtained by resampling the same return series in contiguous blocks.

    Uses a moving block bootstrap (MBB) instead of simple permutation to
    preserve the temporal autocorrelation structure of the return series,
    consistent with the Newey-West HAC assumptions in Phase I.  Simple
    permutation destroys serial dependence and deflates the null variance,
    inflating false-positive rates when returns exhibit autocorrelation.

    Parameters
    ----------
    returns:    Daily (or periodic) return series.
    n_trials:   Number of bootstrap replications.
    seed:       Random seed for reproducibility.
    block_size: Block length for MBB.  Defaults to floor(T^(1/3)).

    Returns
    -------
    float in [0, 1].
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    std = r.std()
    obs_sharpe = r.mean() / std if std > 0.0 else 0.0

    if block_size is None:
        block_size = max(1, int(np.floor(T ** (1.0 / 3.0))))
    n_blocks = int(np.ceil(T / block_size))

    rng = np.random.default_rng(seed)
    count_geq = 0
    for _ in range(n_trials):
        starts = rng.integers(0, max(1, T - block_size + 1), size=n_blocks)
        resampled = np.concatenate([r[s: s + block_size] for s in starts])[:T]
        s = resampled.std()
        perm_sr = resampled.mean() / s if s > 0.0 else 0.0
        if perm_sr >= obs_sharpe:
            count_geq += 1

    return count_geq / n_trials
