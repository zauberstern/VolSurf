"""
Cross-Sectional Portfolio Construction and Backtesting.

Builds signal-weighted long-only portfolios from a panel of individual stock
returns, benchmarks against buy-and-hold SPX, and reports performance metrics.

Design
------
Signals computed per stock:
  - rv_21:      21-day rolling realized variance (annualised)
  - rv_63:      63-day rolling realized variance (annualised)
  - mom_252_21: 12-1 month price momentum (252-day return excluding last 21)
  - vrp_proxy:  market_vix − stock_rv_21  (implied−realised vol proxy)

Portfolio methods:
  - equal_weight:  uniform 1/N across all stocks each day
  - vrp_quartile:  long top-quartile by vrp_proxy (positive VRP = IV > RV)
  - momentum:      long top-quartile by 12-1 month momentum
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_cross_sectional_signals(
    returns: pd.DataFrame,
    vix: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute per-stock daily cross-sectional signals.

    Parameters
    ----------
    returns:
        Wide DataFrame: index=date, columns=permno, values=daily decimal return.
    vix:
        Market VIX as daily decimal series (optional; used for vrp_proxy).

    Returns
    -------
    Panel DataFrame with MultiIndex (date, permno) and columns:
        rv_21, rv_63, mom_252_21, vrp_proxy
    """
    log.info("Computing cross-sectional signals for %d stocks", returns.shape[1])

    # Annualised realized variance (rolling)
    def _ann_rv(rets: pd.DataFrame, window: int) -> pd.DataFrame:
        return rets.rolling(window, min_periods=window // 2).var() * 252

    rv21 = _ann_rv(returns, 21)
    rv63 = _ann_rv(returns, 63)

    # 12-1 month momentum: 252-day return minus last 21 days
    cum_252 = (1 + returns).rolling(252, min_periods=126).apply(
        lambda x: np.prod(x) - 1, raw=True
    )
    cum_21 = (1 + returns).rolling(21, min_periods=10).apply(
        lambda x: np.prod(x) - 1, raw=True
    )
    mom = (1 + cum_252) / (1 + cum_21) - 1

    # VRP proxy: align VIX (market IV²) to stock RV_21
    if vix is not None:
        vix_aligned = vix.reindex(returns.index)
        # Broadcast VIX across all stocks
        vrp_proxy = pd.DataFrame(
            np.outer(vix_aligned.values ** 2, np.ones(returns.shape[1])),
            index=returns.index,
            columns=returns.columns,
        ) - rv21
    else:
        vrp_proxy = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    # Stack to long form
    panel = pd.concat(
        {
            "rv_21": rv21,
            "rv_63": rv63,
            "mom_252_21": mom,
            "vrp_proxy": vrp_proxy,
        },
        axis=1,
    )
    # Reshape: (date, feature, permno) → MultiIndex (date, permno)
    panel = panel.stack(level=1, future_stack=True)
    panel.index.names = ["date", "permno"]
    return panel


# ---------------------------------------------------------------------------
# Portfolio weight construction
# ---------------------------------------------------------------------------

def build_portfolio_weights(
    signals: pd.DataFrame,
    method: str = "equal_weight",
    signal_col: str = "vrp_proxy",
    top_quantile: float = 0.75,
) -> pd.DataFrame:
    """Build daily portfolio weights from cross-sectional signals.

    Parameters
    ----------
    signals:
        MultiIndex (date, permno) DataFrame from compute_cross_sectional_signals.
    method:
        'equal_weight'  — 1/N uniform across all stocks with valid signals.
        'vrp_quartile'  — Long top-quartile by vrp_proxy.
        'momentum'      — Long top-quartile by mom_252_21.
    signal_col:
        Column to rank on (used when method is 'vrp_quartile' or 'momentum').
    top_quantile:
        Quantile cut-off (0.75 = top 25%) for signal-ranked methods.

    Returns
    -------
    Wide DataFrame: index=date, columns=permno, values=weight (sum to 1 per row).
    """
    if method == "equal_weight":
        # 1/N across all stocks that have any valid signal that day
        valid = signals["rv_21"].unstack("permno").notna().astype(float)
        n = valid.sum(axis=1).replace(0, np.nan)
        weights = valid.div(n, axis=0)
        return weights.fillna(0.0)

    col_map = {"vrp_quartile": "vrp_proxy", "momentum": "mom_252_21"}
    sig_col = col_map.get(method, signal_col)

    # Unstack chosen signal
    sig_wide = signals[sig_col].unstack("permno")

    # Cross-sectional rank → top-quantile selection
    def _top_q_weights(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < 10:
            return pd.Series(0.0, index=row.index)
        cutoff = valid.quantile(top_quantile)
        selected = valid[valid >= cutoff]
        w = pd.Series(0.0, index=row.index)
        w[selected.index] = 1.0 / len(selected)
        return w

    weights = sig_wide.apply(_top_q_weights, axis=1)
    return weights.fillna(0.0)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_portfolio(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    transaction_cost_bps: float = 5.0,
    rf: pd.Series | None = None,
) -> dict:
    """Backtest a daily-rebalanced portfolio against a buy-and-hold benchmark.

    Parameters
    ----------
    weights:
        Wide DataFrame from build_portfolio_weights.
    returns:
        Wide DataFrame of daily individual stock returns.
    benchmark_returns:
        Daily benchmark (SPX) returns as a decimal Series.
    transaction_cost_bps:
        Round-trip transaction cost per unit of turnover, in basis points.
    rf:
        Optional daily risk-free rate Series (Ken French RF, decimal).
        Passed through to _compute_metrics for Sharpe computation.

    Returns
    -------
    dict with keys:
        portfolio_returns, benchmark_returns, metrics (nested dict)
    """
    tc = transaction_cost_bps / 10_000.0

    # Align weights and returns on common dates
    common_dates = weights.index.intersection(returns.index)
    w = weights.reindex(common_dates).fillna(0.0)
    r = returns.reindex(common_dates)

    # Use lagged weights (yesterday's weights applied to today's returns)
    w_lagged = w.shift(1).fillna(0.0)

    # Daily portfolio return = sum(w_t-1 * r_t)
    port_ret = (w_lagged * r).sum(axis=1)

    # Turnover = one-way weight change per day (sum |Δw_k|).
    # tc_bps is round-trip; divide by 2 to avoid double-counting
    # (one weight unit change = one buy + one sell; round-trip TC prices the full leg).
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    port_ret_net = port_ret - tc * turnover / 2.0

    # Align benchmark
    bmk = benchmark_returns.reindex(common_dates).fillna(0.0)

    metrics = _compute_metrics(port_ret_net, bmk, rf=rf)
    return {
        "portfolio_returns": port_ret_net,
        "gross_portfolio_returns": port_ret,
        "benchmark_returns": bmk,
        "turnover": turnover,
        "metrics": metrics,
    }


def _compute_metrics(
    portfolio: pd.Series,
    benchmark: pd.Series,
    rf: pd.Series | None = None,
) -> dict:
    """Compute annualised performance metrics for portfolio vs benchmark.

    Parameters
    ----------
    rf : optional daily risk-free rate series (Ken French RF column). If None,
         assumes zero RF (conservative — overstates Sharpe slightly).
    """
    ann = 252

    def _sharpe(r: pd.Series) -> float:
        # Subtract daily RF before computing Sharpe
        rf_aligned = rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0
        excess_r = r - rf_aligned
        if excess_r.std() == 0:
            return np.nan
        return float(excess_r.mean() / excess_r.std() * np.sqrt(ann))

    def _cagr(r: pd.Series) -> float:
        cum = (1 + r).prod()
        n_years = len(r) / ann
        return float(cum ** (1 / n_years) - 1) if n_years > 0 else np.nan

    def _max_dd(r: pd.Series) -> float:
        cum = (1 + r).cumprod()
        rolling_max = cum.cummax()
        dd = (cum - rolling_max) / rolling_max
        return float(dd.min())

    def _hit_rate(r: pd.Series) -> float:
        return float((r > 0).mean())

    excess = portfolio - benchmark
    return {
        "portfolio": {
            "annual_return": _cagr(portfolio),
            "annual_vol": float(portfolio.std() * np.sqrt(ann)),
            "sharpe": _sharpe(portfolio),
            "max_drawdown": _max_dd(portfolio),
            "hit_rate": _hit_rate(portfolio),
            "cum_return": float((1 + portfolio).prod() - 1),
        },
        "benchmark": {
            "annual_return": _cagr(benchmark),
            "annual_vol": float(benchmark.std() * np.sqrt(ann)),
            "sharpe": _sharpe(benchmark),
            "max_drawdown": _max_dd(benchmark),
            "hit_rate": _hit_rate(benchmark),
            "cum_return": float((1 + benchmark).prod() - 1),
        },
        "relative": {
            "information_ratio": float(
                excess.mean() / excess.std() * np.sqrt(ann)
                if excess.std() > 0 else 0.0  # clamp NaN to 0.0 when TE=0
            ),
            "active_return": float(excess.mean() * ann),
            "tracking_error": float(excess.std() * np.sqrt(ann)),
        },
    }


# ---------------------------------------------------------------------------
# Performance report printer
# ---------------------------------------------------------------------------

def print_portfolio_report(result: dict, method: str = "Portfolio") -> None:
    """Print a formatted performance comparison table."""
    m = result["metrics"]
    p = m["portfolio"]
    b = m["benchmark"]
    r = m["relative"]

    print(f"\n{'='*60}")
    print(f"  PORTFOLIO BACKTEST — {method.upper()}")
    print(f"{'='*60}")
    print(f"{'Metric':<28}  {'Portfolio':>12}  {'SPX B&H':>12}")
    print(f"{'-'*56}")
    print(f"{'Annual Return':<28}  {p['annual_return']:>11.2%}  {b['annual_return']:>11.2%}")
    print(f"{'Annual Volatility':<28}  {p['annual_vol']:>11.2%}  {b['annual_vol']:>11.2%}")
    print(f"{'Sharpe Ratio':<28}  {p['sharpe']:>11.4f}  {b['sharpe']:>11.4f}")
    print(f"{'Max Drawdown':<28}  {p['max_drawdown']:>11.2%}  {b['max_drawdown']:>11.2%}")
    print(f"{'Hit Rate':<28}  {p['hit_rate']:>11.2%}  {b['hit_rate']:>11.2%}")
    print(f"{'Cumulative Return':<28}  {p['cum_return']:>11.2%}  {b['cum_return']:>11.2%}")
    print(f"{'-'*56}")
    print(f"{'Information Ratio':<28}  {r['information_ratio']:>11.4f}")
    print(f"{'Active Return (ann)':<28}  {r['active_return']:>11.2%}")
    print(f"{'Tracking Error (ann)':<28}  {r['tracking_error']:>11.2%}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Transaction-cost drag
# ---------------------------------------------------------------------------

def compute_tc_drag(
    tc_bps: float, mean_daily_turnover: float, trading_days: int = 252
) -> float:
    """Annualised transaction-cost drag from observed portfolio turnover.

    Parameters
    ----------
    tc_bps : round-trip cost per unit turnover, in basis points
    mean_daily_turnover : mean L1 per-step portfolio turnover (0–2 range)
    trading_days : annualisation factor (default 252)

    Returns
    -------
    float (negative) — annualised TC drag
    """
    return -(tc_bps / 10_000.0) * mean_daily_turnover * trading_days
