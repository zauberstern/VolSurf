"""
Phase II — Generative Simulation Engine.

Implements:
  - Johansen cointegration rank detection
  - VECM with Reinsel-Ahn finite-sample bias correction
  - SVD fallback (95% PVE) for ill-conditioned windows
  - SVI/SSVI arbitrage-free boundary projection with Lee moment constraints
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as nla
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_paths(
    state_history: pd.DataFrame,
    n_steps: int,
    n_paths: int,
    alpha_sig: float = 0.05,
    max_lags: int = 8,
) -> np.ndarray:
    """Generate n_paths × n_steps simulated state trajectories.

    Fits a VECM (or SVD fallback) to *state_history*, then draws forward
    paths under the physical measure P.  IV columns are post-processed with
    Lee (2004) SSVI arbitrage-free bounds before returning, preventing the RL
    agent from exploiting synthetic arbitrage in the simulated surfaces.

    Parameters
    ----------
    state_history:
        T × n historical state matrix.
    n_steps:
        Forecast horizon (number of simulation steps).
    n_paths:
        Number of Monte-Carlo paths to generate.
    alpha_sig:
        Significance level for Johansen trace test.
    max_lags:
        Maximum VAR lag order to evaluate during BIC selection (default 8).

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps, n_features)
    """
    data = state_history.values.astype(float)
    T, n = data.shape

    try:
        # BIC-optimal lag selection avoids AIC over-selection in long samples.
        bic_lags = _select_vecm_lags(data, max_lags=max_lags)
        rank = _johansen_rank(data, alpha_sig, k_ar_diff=bic_lags)
        residuals, innovation_cov = _fit_vecm(data, rank, k_ar_diff=bic_lags)
        sim_fn = _vecm_simulator(data, rank, innovation_cov, k_ar_diff=bic_lags)
        paths = sim_fn(n_steps, n_paths=n_paths)   # (n_paths, n_steps, n)
    except (nla.LinAlgError, np.linalg.LinAlgError, ValueError):
        # Fallback: SVD capturing 95% PVE
        components, loadings = _svd_fallback(data)
        sim_fn = _svd_simulator(data, components, loadings)
        paths = np.stack([sim_fn(n_steps) for _ in range(n_paths)], axis=0)

    # Apply SSVI arbitrage-free bounds to IV columns (Gatheral-Jacquier SVI framework).
    paths = apply_ssvi_bounds(paths, list(state_history.columns))
    return paths


def apply_ssvi_bounds(
    paths: np.ndarray,
    feature_names: list[str],
    iv_min: float = 0.01,
    iv_max: float = 3.0,
) -> np.ndarray:
    """Apply Lee (2004) SSVI arbitrage-free bounds to IV columns of simulated paths.

    Enforces three constraints derived from the Gatheral-Jacquier SVI/SSVI
    framework and Lee's moment formula, preventing the RL agent from
    exploiting synthetic arbitrage in the simulated volatility surfaces:

    1. **Non-negativity / floor**: iv_30, iv_91 ≥ iv_min (0.01 = 1% ATM IV).
       Implied variance is non-negative by construction; a small floor guards
       degenerate near-zero values that can cause numerical instability.

    2. **Lee upper bound**: iv_30, iv_91 ≤ iv_max.
       Lee (2004) moment formula: for p_plus → 0 (fat tails), the slope of
       total variance in the right wing approaches 4.  With T=30/365 and
       |k|=0.75 log-moneyness, w_max ≈ 3, giving σ_max ≈ 3.0 in IV units.

    3. **Calendar spread no-arbitrage**: w_91 ≥ w_30 (monotone total variance).
       Total variance w = σ²·T must be non-decreasing in maturity T.
       For iv_30 at T=30/365 and iv_91 at T=91/365:
           iv_91² · 91 ≥ iv_30² · 30   ⟹   iv_91 ≥ iv_30 · √(30/91).

    Parameters
    ----------
    paths:
        Simulated paths array of shape (n_paths, n_steps, n_features).
    feature_names:
        Column names corresponding to the last dimension of paths.
    iv_min:
        Minimum permissible implied volatility (default 0.01 = 1%).
    iv_max:
        Maximum permissible implied volatility (default 3.0 = 300%).

    Returns
    -------
    np.ndarray of same shape as paths, with IV columns constrained.
    """
    paths = paths.copy()
    names = list(feature_names)

    def _idx(col: str) -> int | None:
        return names.index(col) if col in names else None

    i30 = _idx("iv_30")
    i91 = _idx("iv_91")

    # Constraint 1 & 2: floor and ceiling on each IV column independently.
    # Replace NaN (simulation divergence) with iv_min before clip to avoid
    # silent NaN propagation through the state tensor.
    for i in (i30, i91):
        if i is not None:
            paths[:, :, i] = np.clip(
                np.nan_to_num(paths[:, :, i], nan=iv_min), iv_min, iv_max
            )

    # Constraint 3: calendar spread — iv_91 ≥ iv_30 · √(30/91)
    if i30 is not None and i91 is not None:
        cal_floor = paths[:, :, i30] * np.sqrt(30.0 / 91.0)
        paths[:, :, i91] = np.maximum(paths[:, :, i91], cal_floor)

    return paths


def project_surface_ssvi(
    log_moneyness: np.ndarray,
    raw_total_variance: np.ndarray,
    asset_moments: tuple[float, float],
) -> np.ndarray:
    """Project raw simulated implied variances onto SSVI arbitrage-free bounds.

    Parameters
    ----------
    log_moneyness:
        Array of log-moneyness values k = log(K / F).
    raw_total_variance:
        Raw total variance w(k) output from the simulation.
    asset_moments:
        (p_plus, p_minus) — right and left tail moment exponents from
        Lee's moment formula.  Constrain wing slopes accordingly.

    Returns
    -------
    np.ndarray of SSVI-constrained total implied variance.
    """
    p_plus, p_minus = asset_moments

    # Lee (2004): upper variance bound per wing
    # Right tail (k > 0): w(k) <= slope_r * k
    # Left tail  (k < 0): w(k) <= slope_l * |k|
    p_r = np.clip(p_plus, 0.0, 2.0)
    p_l = np.clip(p_minus, 0.0, 2.0)
    slope_r = 2.0 - p_r + 2.0 * np.sqrt(max(1.0 - p_r, 0.0))
    slope_l = 2.0 - p_l + 2.0 * np.sqrt(max(1.0 - p_l, 0.0))

    k = log_moneyness
    lee_upper = np.where(
        np.abs(k) < 1e-6,
        10.0,  # ATM: no wing constraint, use large cap
        np.where(k > 0, slope_r * k, slope_l * np.abs(k)),
    )
    lee_lower = np.zeros_like(k)  # variance is non-negative

    # Ensure bounds are valid (lower strictly < upper); guard degenerate inputs
    lee_upper = np.maximum(lee_upper, lee_lower + 1e-8)

    w_init = np.clip(raw_total_variance.copy(), lee_lower + 1e-9, lee_upper - 1e-9)
    result = minimize(
        _ssvi_objective,
        w_init,
        args=(k,),
        method="L-BFGS-B",
        bounds=list(zip(lee_lower, lee_upper)),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    # Clamp to strict positive floor so downstream sqrt(w) is safe.
    return np.maximum(result.x, 1e-8)


# ---------------------------------------------------------------------------
# BIC lag-order selection
# ---------------------------------------------------------------------------

def _select_vecm_lags(data: np.ndarray, max_lags: int = 8) -> int:
    """Select VAR lag order via Hannan-Quinn Information Criterion (HQIC).

    HQIC provides a stronger-than-AIC but weaker-than-BIC penalty, which
    better captures long-memory IV dynamics without exhausting Reinsel-Ahn
    degrees of freedom:

        HQIC(p) = log|Σ̂_p| + 2*(p·n²)*log(log(T_eff)) / T_eff

    Returns p ≥ 1.  Falls back to 1 on any numerical error.
    """
    T, n = data.shape
    best_hqic = np.inf
    best_p = 1
    # Maximum testable lag: need at least n+1 effective obs after subsetting
    p_max = min(max_lags, max(1, (T - 1) // n - 1))
    for p in range(1, p_max + 1):
        try:
            res = VAR(data).fit(maxlags=p, ic=None, trend="c")
            sigma = res.sigma_u                      # (n, n)
            log_det = np.log(np.linalg.det(sigma + 1e-12 * np.eye(n)))
            T_eff = T - p
            k_params = p * n * n                     # AR parameters (excl. intercept)
            # HQIC penalty: 2 * k * log(log(T)) / T  (Hannan-Quinn 1979)
            log_log_T = np.log(np.log(max(T_eff, 3)))  # guard log(log(T)) for tiny T
            hqic = log_det + 2.0 * k_params * log_log_T / T_eff
            if hqic < best_hqic:
                best_hqic = hqic
                best_p = p
        except Exception:
            break
    return best_p


# ---------------------------------------------------------------------------
# Johansen rank detection
# ---------------------------------------------------------------------------

def _johansen_rank(data: np.ndarray, alpha: float, k_ar_diff: int = 1) -> int:
    """Return cointegration rank via Johansen trace test (Reinsel-Ahn corrected).

    Guard: if DoF = n*p >= T_block, the Reinsel-Ahn scalar is negative, which
    flips eigenvalue signs and produces divergent paths.  In that case a
    ValueError is raised so the caller falls through to the SVD fallback.
    """
    result = coint_johansen(data, det_order=0, k_ar_diff=k_ar_diff)
    T, n = data.shape
    p = k_ar_diff

    # Reinsel-Ahn guard: if DoF >= T the correction scalar is negative,
    # which flips all eigenvalue signs and produces divergent paths.
    if n * p >= T:
        raise ValueError(
            f"Reinsel-Ahn DoF ({n * p}) >= T_block ({T}): correction scalar "
            f"= ({T - n * p}) / {T} < 0; bypassing to SVD fallback."
        )

    # Reinsel-Ahn correction: scale trace statistics by (T - n*p) / T
    correction = (T - n * p) / T
    trace_stats = result.lr1 * correction
    critical_values = result.cvt  # shape (n_ranks, 3) — cols: 10%, 5%, 1%

    col = {0.10: 0, 0.05: 1, 0.01: 2}.get(alpha, 1)
    rank = 0
    for i in range(n):
        if trace_stats[i] > critical_values[i, col]:
            rank += 1
        else:
            break
    return rank


# ---------------------------------------------------------------------------
# VECM fit and simulation
# ---------------------------------------------------------------------------

def _fit_vecm(data: np.ndarray, rank: int, k_ar_diff: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fit VECM and return (residuals, innovation_covariance).

    For rank=0 (no cointegration), falls back to a VAR(1) on differences
    because statsmodels VECM has a shape bug when coint_rank=0.
    """
    if rank == 0:
        # No cointegration: model Δy as VAR(1) in differences
        diffs = np.diff(data, axis=0)             # (T-1, n)
        # OLS: Δy_t = Γ·Δy_{t-1} + ε
        X = diffs[:-1]                            # (T-2, n)
        Y = diffs[1:]                             # (T-2, n)
        gamma, _, _, _ = nla.lstsq(X, Y, rcond=None)   # (n, n)
        resid = Y - X @ gamma
        innovation_cov = np.cov(resid.T)
        return resid, innovation_cov

    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic="ci")
    fitted = model.fit()
    residuals = fitted.resid
    innovation_cov = np.cov(residuals.T)
    return residuals, innovation_cov


def _vecm_simulator(
    data: np.ndarray, rank: int, innovation_cov: np.ndarray, k_ar_diff: int = 1
):
    """Return a closure that generates n_paths paths of length n_steps.

    Implements proper VECM dynamics with state feedback at every step:

        Δy_t = α(β'y_{t-1} + μ) + Γ Δy_{t-1} + ε_t
        y_t  = y_{t-1} + Δy_t

    For rank=0, reduces to a random-walk VAR(1) on differences.
    Vectorised over all paths simultaneously for speed.
    """
    if rank == 0:
        # No cointegration — simulate as VAR(1) in differences (random walk in levels)
        diffs = np.diff(data, axis=0)
        X = diffs[:-1]; Y = diffs[1:]
        gamma_rw, _, _, _ = nla.lstsq(X, Y, rcond=None)
        L = nla.cholesky(innovation_cov + 1e-8 * np.eye(data.shape[1]))
        LT = L.T
        gT = gamma_rw.T

        def simulate_rw(n_steps: int, n_paths: int = 1) -> np.ndarray:
            y  = np.broadcast_to(data[-1], (n_paths, data.shape[1])).copy()
            dy = np.broadcast_to(diffs[-1], (n_paths, data.shape[1])).copy()
            out = np.empty((n_paths, n_steps, data.shape[1]))
            for t in range(n_steps):
                eps = np.random.randn(n_paths, data.shape[1]) @ LT
                dy  = dy @ gT + eps
                y   = y + dy
                out[:, t, :] = y
            return out

        return simulate_rw

    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=rank, deterministic="ci")
    fitted = model.fit()

    alpha = np.asarray(fitted.alpha)                       # (n, r)
    beta  = np.asarray(fitted.beta)                        # (n, r)
    gamma_raw = getattr(fitted, "gamma", None)
    gamma = (
        np.asarray(gamma_raw)
        if gamma_raw is not None and np.asarray(gamma_raw).size > 0
        else np.zeros((data.shape[1], data.shape[1]))
    )                                                      # (n, n)
    det_raw = getattr(fitted, "det_coef_coint", None)
    det_coint = (
        np.asarray(det_raw).flatten()[:rank]
        if det_raw is not None
        else np.zeros(max(rank, 1))
    )                                                      # (r,)

    L = nla.cholesky(innovation_cov + 1e-8 * np.eye(innovation_cov.shape[0]))
    n = data.shape[1]
    # Pre-compute transpose for vectorised matmul (paths × n) @ (n × r)
    betaT  = beta.T    # (r, n)
    alphaT = alpha.T   # (r, n) -- used as (n, r) @ (r,) below
    gammaT = gamma.T   # (n, n)
    LT     = L.T       # (n, n)

    def simulate(n_steps: int, n_paths: int = 1) -> np.ndarray:
        """Vectorised VECM simulation. Returns (n_paths, n_steps, n)."""
        # Initial conditions: all paths start at last in-sample state
        y  = np.broadcast_to(data[-1],  (n_paths, n)).copy()   # (P, n)
        dy = np.broadcast_to(data[-1] - data[-2], (n_paths, n)).copy()  # (P, n)
        out = np.empty((n_paths, n_steps, n))

        for t in range(n_steps):
            # Error-correction term: (P, r) = (P, n) @ (n, r)
            ci  = y @ beta + det_coint          # (P, r)
            ec  = ci @ alpha.T                  # (P, n)
            # Short-run adjustment
            sr  = dy @ gammaT                   # (P, n)
            # Gaussian innovations
            eps = np.random.randn(n_paths, n) @ LT  # (P, n)

            dy  = ec + sr + eps                 # Δy_t
            y   = y + dy                        # y_t = y_{t-1} + Δy_t
            out[:, t, :] = y

        return out

    return simulate


# ---------------------------------------------------------------------------
# SVD fallback (95% PVE)
# ---------------------------------------------------------------------------

def _svd_fallback(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose data via SVD and retain components explaining 95% PVE."""
    data_centered = data - data.mean(axis=0)
    U, s, Vt = nla.svd(data_centered, full_matrices=False)
    pve = np.cumsum(s ** 2) / np.sum(s ** 2)
    n_components = int(np.searchsorted(pve, 0.95)) + 1
    components = U[:, :n_components] * s[:n_components]  # T × k scores
    loadings = Vt[:n_components]                          # k × n loadings
    return components, loadings


def _svd_simulator(
    data: np.ndarray, components: np.ndarray, loadings: np.ndarray
):
    """Return a closure that generates one path via SVD score simulation."""
    score_diffs = np.diff(components, axis=0)
    diff_cov = np.cov(score_diffs.T) if score_diffs.shape[1] > 1 else np.array([[score_diffs.var()]])
    last_score = components[-1]
    data_mean = data.mean(axis=0)   # add back after projecting from centred scores
    L = nla.cholesky(diff_cov + 1e-8 * np.eye(diff_cov.shape[0]))

    def simulate(n_steps: int) -> np.ndarray:
        path = np.zeros((n_steps, loadings.shape[1]))
        score = last_score.copy()
        for t in range(n_steps):
            score = score + L @ np.random.randn(len(score))
            path[t] = score @ loadings + data_mean
        return path

    return simulate


# ---------------------------------------------------------------------------
# SSVI / Lee moment constraints
# ---------------------------------------------------------------------------

def _ssvi_objective(w: np.ndarray, k: np.ndarray) -> float:
    """Penalise calendar-spread and butterfly arbitrage violations."""
    # Butterfly: d^2w/dk^2 >= 0 enforced via finite-difference penalty
    d2w = np.diff(w, 2)
    butterfly_penalty = np.sum(np.maximum(0.0, -d2w) ** 2)
    # Smoothness regulariser
    smoothness = np.sum(np.diff(w) ** 2)
    return butterfly_penalty + 1e-4 * smoothness


def _lee_moment_bound(k: np.ndarray, p: float, tail: str) -> np.ndarray:
    """Lee (2004) moment formula asymptotic upper/lower bound on total variance.

    For right tail: w(k) <= (2 - p + 2*sqrt(1 - p)) * |k| as k -> +inf.
    For left tail:  w(k) <= (2 - p + 2*sqrt(1 - p)) * |k| as k -> -inf.
    """
    p_clipped = np.clip(p, 0.0, 2.0)
    slope = 2.0 - p_clipped + 2.0 * np.sqrt(max(1.0 - p_clipped, 0.0))
    if tail == "right":
        bound = slope * np.abs(np.maximum(k, 0.0))
    else:
        bound = slope * np.abs(np.minimum(k, 0.0))
    # Minimum total variance is 0; max is the Lee bound (or large value for ATM)
    bound = np.where(np.abs(k) < 1e-6, 10.0, bound)
    return bound
