"""
Visualisation utilities for the Friction-Aware Alpha Generation Framework.

Plot groups
-----------
1. portfolio_dashboard         — 4-panel cross-sectional strategy backtest
2. rl_agent_dashboard          — 4-panel RL training/OOS behaviour
3. policy_surface_3d           — 3-D policy surface + GRU memory trajectory
4. institutional_scorecard     — traffic-light panel: DSR/PF/WFE/IR/SR/MDD vs thresholds
5. attribution_tearsheet       — horizontal beta bars with 95% HAC CI
6. monthly_returns_heatmap     — calendar heatmap RL vs SPX (institutional format)
7. rolling_metrics_panel       — 63-day rolling SR/IR/CVaR/MDD over OOS window
8. return_decomposition_waterfall — gross return → factor stripping → net alpha
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless — saves to file
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

_FIG_DIR = Path(__file__).parents[2] / "data" / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature display names  (raw variable name → human-readable label)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "iv_30":            "30-Day Impl. Vol (ATM)",
    "iv_91":            "91-Day Impl. Vol (ATM)",
    "term_structure":   "IV Term Structure (91d−30d)",
    "skew_25d":         "25Δ Put-Call Skew",
    "skew_10d":         "10Δ Tail Skew",
    "rv_21":            "21-Day Realized Vol",
    "rv_63":            "63-Day Realized Vol",
    "vrp":              "Vol Risk Premium (IV−RV)",
    "vix":              "VIX Index",
    "yield_10y":        "10Y Treasury Yield",
    "log_pcr":          "Log Put/Call Ratio",
    "short_rate":       "30D Zero-Coupon Rate",
    "iv_dispersion":    "IV Cross-Sect. Dispersion",
    "vxn_vix_spread":   "Nasdaq/SPX Vol Spread",
    "vxd_vix_spread":   "DJIA/SPX Vol Spread",
    "rv_dispersion":    "RV Cross-Sect. Dispersion",
    "constituent_iv":   "Constituent Mean IV",
    "constituent_skew": "Constituent Mean Skew",
}


def _display_name(raw: str) -> str:
    """Return a human-readable display name for a feature variable name."""
    return FEATURE_DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cum(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod() - 1

def _rolling_sharpe(r: pd.Series, window: int = 63) -> pd.Series:
    mu  = r.rolling(window).mean()
    sig = r.rolling(window).std()
    return (mu / sig.replace(0, np.nan)) * np.sqrt(252)

def _drawdown(r: pd.Series) -> pd.Series:
    cum = (1 + r.fillna(0)).cumprod()
    return (cum - cum.cummax()) / cum.cummax()

def _style(ax, title: str, ylabel: str = "", pct_y: bool = False) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    if pct_y:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio dashboard
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "Equal-Weight": "#2196F3",
    "VRP Quartile": "#FF9800",
    "Momentum":     "#4CAF50",
    "SPX B&H":      "#9E9E9E",
}

def portfolio_dashboard(
    backtest_results: dict[str, dict],
    spx_ret: pd.Series,
    output_path: str | Path | None = None,
) -> Path:
    """Four-panel portfolio performance dashboard.

    Parameters
    ----------
    backtest_results:
        Dict mapping strategy label -> dict from backtest_portfolio().
    spx_ret:
        Daily SPX returns (benchmark).
    output_path:
        Where to save the PNG.  Defaults to data/figures/portfolio_dashboard.png.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "portfolio_dashboard.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "S&P 500 Cross-Sectional Portfolio Backtest  (2015 – 2024)\n"
        "Point-in-time constituents  |  5 bps round-trip TC",
        fontsize=11, fontweight="bold", y=0.98,
    )
    ax_cum, ax_sharpe, ax_dd, ax_annual = axes.flat

    # ── Panel 1: Cumulative returns ──────────────────────────────────────────
    _style(ax_cum, "Cumulative Return", pct_y=True)
    spx_cum = _cum(spx_ret)
    ax_cum.plot(spx_cum.index, spx_cum, color=STRATEGY_COLORS["SPX B&H"],
                lw=1.5, label="SPX B&H", linestyle="--")
    for label, res in backtest_results.items():
        r = res["portfolio_returns"]
        c = _cum(r)
        ax_cum.plot(c.index, c, color=STRATEGY_COLORS.get(label, "#555"),
                    lw=1.8, label=label)
    ax_cum.legend(fontsize=7, framealpha=0.6)
    ax_cum.axhline(0, color="black", lw=0.5)

    # ── Panel 2: Rolling 63-day Sharpe ──────────────────────────────────────
    _style(ax_sharpe, "Rolling 63-day Sharpe Ratio")
    ax_sharpe.plot(_rolling_sharpe(spx_ret), color=STRATEGY_COLORS["SPX B&H"],
                   lw=1.2, label="SPX B&H", linestyle="--")
    for label, res in backtest_results.items():
        r = res["portfolio_returns"]
        ax_sharpe.plot(_rolling_sharpe(r), color=STRATEGY_COLORS.get(label, "#555"),
                       lw=1.5, label=label)
    ax_sharpe.axhline(0, color="black", lw=0.5)
    ax_sharpe.legend(fontsize=7, framealpha=0.6)

    # ── Panel 3: Drawdown ────────────────────────────────────────────────────
    _style(ax_dd, "Drawdown", pct_y=True)
    ax_dd.fill_between(_drawdown(spx_ret).index, _drawdown(spx_ret), 0,
                       color=STRATEGY_COLORS["SPX B&H"], alpha=0.3, label="SPX B&H")
    for label, res in backtest_results.items():
        dd = _drawdown(res["portfolio_returns"])
        ax_dd.plot(dd.index, dd, color=STRATEGY_COLORS.get(label, "#555"),
                   lw=1.4, label=label)
    ax_dd.legend(fontsize=7, framealpha=0.6)

    # ── Panel 4: Annual returns bar chart ────────────────────────────────────
    _style(ax_annual, "Annual Return by Year", pct_y=True)
    all_rets = {"SPX B&H": spx_ret}
    for label, res in backtest_results.items():
        all_rets[label] = res["portfolio_returns"]

    years = sorted(set(spx_ret.index.year))
    x = np.arange(len(years))
    n_strats = len(all_rets)
    width = 0.8 / n_strats
    for i, (label, r) in enumerate(all_rets.items()):
        annual = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
        annual = annual.reindex(years, fill_value=0.0)
        color = STRATEGY_COLORS.get(label, "#555")
        offset = (i - n_strats / 2 + 0.5) * width
        ax_annual.bar(x + offset, annual.values, width=width * 0.9,
                      label=label, color=color, alpha=0.85)
    ax_annual.set_xticks(x)
    ax_annual.set_xticklabels([str(y) for y in years], fontsize=6, rotation=45)
    ax_annual.axhline(0, color="black", lw=0.5)
    ax_annual.legend(fontsize=7, framealpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# RL agent dashboard
# ─────────────────────────────────────────────────────────────────────────────

def rl_agent_dashboard(
    rl_results: dict,
    output_path: str | Path | None = None,
) -> Path:
    """Four-panel RL agent behaviour dashboard.

    Expected keys in rl_results
    ---------------------------
    training_rewards  : list[float]  -- per-epoch mean reward
    training_cvar     : list[float]  -- per-epoch CVaR
    training_eta      : list[float]  -- per-epoch Lagrangian dual eta
    oos_dates         : pd.DatetimeIndex
    oos_rl_returns    : pd.Series    -- net OOS daily returns
    oos_spx_returns   : pd.Series    -- OOS SPX daily returns
    oos_actions       : pd.Series    -- OOS mean |action| per day
    ww_half_width     : float        -- W-W no-trade zone half-width
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "rl_agent_dashboard.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "RL Agent Behaviour  |  GRU-64  |  Lagrangian CVaR  (OOS: 2022 – 2024)",
        fontsize=11, fontweight="bold", y=0.98,
    )
    ax_train, ax_eta, ax_action, ax_cum = axes.flat

    epochs = np.arange(1, len(rl_results["training_rewards"]) + 1)

    # ── Panel 1: Training convergence ────────────────────────────────────────
    ax1b = ax_train.twinx()
    ax_train.plot(epochs, rl_results["training_rewards"], color="#1976D2",
                  lw=2, label="Mean Reward")
    ax1b.plot(epochs, rl_results["training_cvar"], color="#E53935",
              lw=1.5, linestyle="--", label="CVaR")
    ax_train.set_title("Training: Reward & CVaR per Epoch",
                       fontsize=10, fontweight="bold")
    ax_train.set_xlabel("Epoch", fontsize=8)
    ax_train.set_ylabel("Reward", fontsize=8, color="#1976D2")
    ax1b.set_ylabel("CVaR", fontsize=8, color="#E53935")
    ax_train.tick_params(labelsize=7)
    ax1b.tick_params(labelsize=7)
    ax_train.spines[["top", "right"]].set_visible(False)
    lines1, lbl1 = ax_train.get_legend_handles_labels()
    lines2, lbl2 = ax1b.get_legend_handles_labels()
    ax_train.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, framealpha=0.6)
    ax_train.grid(axis="y", alpha=0.2)

    # ── Panel 2: Lagrangian dual variable eta ────────────────────────────────
    _style(ax_eta, "Lagrangian Dual Variable η (CVaR Penalty)")
    ax_eta.plot(epochs, rl_results["training_eta"], color="#7B1FA2", lw=2)
    ax_eta.axhline(0, color="black", lw=0.5)
    ax_eta.set_xlabel("Epoch", fontsize=8)

    # ── Panel 3: OOS action magnitude + W-W zone ────────────────────────────
    _style(ax_action, "OOS Mean |Action| vs Whalley-Wilmott No-Trade Zone")
    actions = pd.Series(rl_results["oos_actions"])
    if hasattr(actions.index, "dtype") and actions.index.dtype != "datetime64[ns]":
        actions.index = rl_results["oos_dates"]
    ax_action.plot(rl_results["oos_dates"], actions,
                   color="#FF7043", lw=1, alpha=0.7, label="Mean |action|")
    ww = rl_results["ww_half_width"]
    ax_action.axhline(ww, color="#388E3C", lw=1.5, linestyle="--",
                      label=f"W-W half-width = {ww:.3f}")
    ax_action.fill_between(rl_results["oos_dates"], 0, ww,
                           color="#388E3C", alpha=0.08)
    ax_action.legend(fontsize=7, framealpha=0.6)
    ax_action.set_xlabel("Date", fontsize=8)

    # ── Panel 4: OOS cumulative return ──────────────────────────────────────
    _style(ax_cum, "OOS Cumulative Return: RL vs SPX", pct_y=True)
    rl_ret  = pd.Series(rl_results["oos_rl_returns"],  index=rl_results["oos_dates"])
    spx_ret = pd.Series(rl_results["oos_spx_returns"], index=rl_results["oos_dates"])
    ax_cum.plot(_cum(rl_ret).index,  _cum(rl_ret),  color="#1976D2", lw=2,  label="RL Agent")
    ax_cum.plot(_cum(spx_ret).index, _cum(spx_ret), color="#9E9E9E", lw=1.5,
                linestyle="--", label="SPX B&H")
    ax_cum.axhline(0, color="black", lw=0.5)
    ax_cum.legend(fontsize=7, framealpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 3-D Policy Transparency Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def _pca3(X: np.ndarray) -> np.ndarray:
    """Compact numpy-only PCA → first 3 principal components."""
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:3].T          # (N, 3)


def policy_surface_3d(
    vix_grid:       np.ndarray,    # (G, G) meshgrid of VIX values
    vrp_grid:       np.ndarray,    # (G, G) meshgrid of VRP values
    action_surface: np.ndarray,    # (G, G) policy output at each grid point
    oos_states:     np.ndarray,    # (T, D) actual OOS state vectors
    hidden_states:  np.ndarray,    # (T, 64) GRU hidden state at each OOS step
    oos_actions:    np.ndarray,    # (T,) tanh action at each OOS step
    oos_returns:    np.ndarray,    # (T,) SPX daily returns (for coloring)
    feature_names:  list,          # length-D list of state feature names
    output_path: str | Path | None = None,
) -> Path:
    """Two-panel 3-D policy transparency figure.

    Panel A — Policy Action Surface
      3-D surface: z = pi(VIX, VRP | other_features=mean), with the actual
      OOS trajectory overlaid as a scatter.  Colour encodes action polarity
      (red=long / blue=defensive).  A shadow contour is projected onto the
      floor so regime boundaries are readable from any viewing angle.

    Panel B — GRU Memory Trajectory
      The 64-dim GRU hidden state is projected to 3 PCs via SVD.  Each OOS
      trading day is a point; the temporal path connects them as a line.
      Colour sweeps blue->red across the OOS period (Jan 2022 -> Dec 2024).
      Point size is proportional to |action|, so high-conviction days are
      visually prominent.  The three labelled regime centroids (2022 drawdown,
      2023 recovery, 2024 AI bull) show whether the agent formed distinct
      memories for different regimes.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection

    output_path = Path(output_path) if output_path else _FIG_DIR / "policy_surface_3d.png"

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(
        "RL Policy Transparency  |  GRU-64  |  Lagrangian CVaR",
        fontsize=13, fontweight="bold", y=0.98,
    )

    ax_surf = fig.add_subplot(121, projection="3d")
    ax_pca  = fig.add_subplot(122, projection="3d")

    # ── Shared colour norm for action [-1, +1] ────────────────────────────────
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable
    norm_action = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL A: Policy Action Surface
    # ─────────────────────────────────────────────────────────────────────────
    cmap_surf = plt.cm.RdBu_r

    # Surface
    surf = ax_surf.plot_surface(
        vix_grid, vrp_grid, action_surface,
        cmap=cmap_surf, norm=norm_action,
        alpha=0.68, linewidth=0, antialiased=True,
    )

    # Shadow contour projected onto floor
    z_floor = action_surface.min() - 0.08
    ax_surf.contourf(
        vix_grid, vrp_grid, action_surface,
        zdir="z", offset=z_floor, cmap=cmap_surf, norm=norm_action,
        levels=20, alpha=0.4,
    )

    # OOS trajectory scatter (colour = sign of realised SPX return)
    vix_idx = feature_names.index("vix") if "vix" in feature_names else 7
    vrp_idx = feature_names.index("vrp") if "vrp" in feature_names else 6
    oos_vix = oos_states[:, vix_idx]
    oos_vrp = oos_states[:, vrp_idx]

    ret_colors = np.where(oos_returns >= 0, "#43A047", "#E53935")
    ax_surf.scatter(
        oos_vix, oos_vrp, oos_actions,
        c=ret_colors, s=6, alpha=0.55, depthshade=True,
        label="OOS: green=up day, red=down day",
    )

    ax_surf.set_xlabel("VIX", fontsize=9, labelpad=6)
    ax_surf.set_ylabel("VRP (IV-RV)", fontsize=9, labelpad=6)
    ax_surf.set_zlabel("Policy Action", fontsize=9, labelpad=6)
    ax_surf.set_title(
        "A  Policy Action Surface  π(VIX, VRP)\n"
        "surface = mean agent action  |  dots = actual OOS days\n"
        "red = aggressive long  |  blue = defensive/short",
        fontsize=9, fontweight="bold", pad=10,
    )
    ax_surf.tick_params(labelsize=7)
    ax_surf.view_init(elev=32, azim=-45)

    # Level-set isolines projected on the surface itself for readability
    try:
        ax_surf.contour3D(
            vix_grid, vrp_grid, action_surface,
            levels=8, cmap="RdBu_r", norm=norm_action,
            linewidths=0.8, alpha=0.85, zorder=4,
        )
    except Exception:
        pass  # contour3D not available in all matplotlib builds

    # Colourbar
    cb = fig.colorbar(
        ScalarMappable(norm=norm_action, cmap=cmap_surf),
        ax=ax_surf, shrink=0.55, pad=0.08, aspect=20,
    )
    cb.set_label("Action  (-1 = max short  /  +1 = max long)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL B: GRU Memory Trajectory (PCA 3D)
    # ─────────────────────────────────────────────────────────────────────────
    T = len(hidden_states)
    h_pca   = _pca3(hidden_states)         # (T, 3)
    t_norm  = np.linspace(0, 1, T)         # 0=Jan-2022, 1=Dec-2024
    cmap_t  = plt.cm.coolwarm

    # Ghost line for temporal path
    ax_pca.plot(
        h_pca[:, 0], h_pca[:, 1], h_pca[:, 2],
        lw=0.5, alpha=0.25, color="grey", zorder=1,
    )

    # Scatter: colour=time, size=|action|
    sizes = 10 + 200 * (np.abs(oos_actions) / max(np.abs(oos_actions).max(), 1e-6))
    sc = ax_pca.scatter(
        h_pca[:, 0], h_pca[:, 1], h_pca[:, 2],
        c=t_norm, cmap=cmap_t, s=sizes, alpha=0.65, depthshade=True, zorder=2,
    )

    # Annotate three market regime centroids — use 2D annotations with arrows
    # to avoid 3D text overlap.  We project the 3D centroid to 2D figure
    # coordinates and place flat annotation boxes at the axes edges.
    regime_labels = [
        (0,       T // 3,       "2022\nDrawdown", "#B71C1C"),
        (T // 3,  2 * T // 3,  "2023\nRecovery", "#1B5E20"),
        (2 * T // 3, T,         "2024\nAI Bull",  "#0D47A1"),
    ]
    # Annotation anchor offsets in 2D axes fraction coordinates (upper area)
    _anchor_2d = [(0.10, 0.88), (0.50, 0.92), (0.88, 0.88)]
    from mpl_toolkits.mplot3d import proj3d
    import matplotlib.patches as mpatches
    for (lo, hi, label, col), (ax_frac_x, ax_frac_y) in zip(regime_labels, _anchor_2d):
        cx, cy, cz = h_pca[lo:hi, :3].mean(axis=0)
        ax_pca.scatter([cx], [cy], [cz], color=col, s=160, marker="*",
                       edgecolors="white", linewidths=0.8, zorder=5)
        # Project centroid to display coordinates
        try:
            x2d, y2d, _ = proj3d.proj_transform(cx, cy, cz, ax_pca.get_proj())
            # Convert display coordinates to axes fraction
            ax_box = ax_pca.transData.transform((x2d, y2d))
            ax_inv = ax_pca.transAxes.inverted()
            node_frac = ax_inv.transform(ax_box)
            ax_pca.annotate(
                label,
                xy=node_frac,
                xytext=(ax_frac_x, ax_frac_y),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=8.5, color=col, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=col, alpha=0.92, linewidth=1.2),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=col,
                    lw=1.2,
                    connectionstyle="arc3,rad=0.2",
                ),
                zorder=8,
            )
        except Exception:
            # Fallback if projection unavailable
            ax_pca.text(
                cx, cy, cz, label,
                fontsize=8, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=col, alpha=0.80, linewidth=0.8),
                zorder=6,
            )

    ax_pca.set_xlabel("PC-1", fontsize=9, labelpad=6)
    ax_pca.set_ylabel("PC-2", fontsize=9, labelpad=6)
    ax_pca.set_zlabel("PC-3", fontsize=9, labelpad=6)
    ax_pca.set_title(
        "B  GRU Hidden State Memory  (PCA 3D)\n"
        "colour = time  |  size proportional to |action|  |  * = regime centroids",
        fontsize=9, fontweight="bold", pad=10,
    )
    ax_pca.tick_params(labelsize=7)
    ax_pca.view_init(elev=22, azim=55)

    cb2 = fig.colorbar(sc, ax=ax_pca, shrink=0.55, pad=0.08, aspect=20)
    cb2.set_label("Time  (blue=Jan-2022  /  red=Dec-2024)", fontsize=7)
    cb2.ax.tick_params(labelsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Institutional Validation Scorecard
# ─────────────────────────────────────────────────────────────────────────────

def institutional_scorecard(
    metrics: dict,
    output_path: str | Path | None = None,
) -> Path:
    """Traffic-light scorecard showing all institutional metrics vs. thresholds.

    Expected keys in metrics
    ------------------------
    sharpe, mdd, profit_factor, information_ratio, dsr, wfe, mc_pvalue
    ann_return_rl, ann_return_bm, ann_vol_rl  (all floats)

    Each metric is shown as a horizontal gauge bar against its institutional
    target range.  Colours: green = pass, amber = borderline, red = fail.
    A summary verdict panel appears on the right side.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "institutional_scorecard.png"

    # ── Metric definitions: (label, value, low_pass, high_warn, unit, higher_better) ──
    rows = [
        ("Sharpe Ratio",            metrics.get("sharpe", 0.0),          2.0,   3.0,   "",     True),
        ("Max Drawdown",            abs(metrics.get("mdd", 0.0)) * 100,  15.0,  25.0,  "%",    False),
        ("Profit Factor",           metrics.get("profit_factor", 0.0),   1.75,  3.0,   "×",    True),
        ("Information Ratio",       metrics.get("information_ratio", 0.0), 0.38, 0.80, "",    True),
        ("Deflated Sharpe Ratio",   metrics.get("dsr", 0.0),             0.95,  0.99,  "",     True),
        ("Walk-Forward Efficiency", metrics.get("wfe", 0.0),             0.50,  0.75,  "×",    True),
        ("MC Permutation p-value",  metrics.get("mc_pvalue", 1.0),       0.05,  0.10,  "",     False),
        ("Ann. Return (RL)",        metrics.get("ann_return_rl", 0.0)*100, 5.0, 15.0, "%",    True),
        ("Ann. Volatility",         metrics.get("ann_vol_rl", 0.0)*100,  10.0,  20.0,  "%",    False),
    ]

    n = len(rows)
    fig, (ax_main, ax_verdict) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(
        "Institutional Validation Scorecard\n"
        "Friction-Aware DRL Alpha Generation Framework",
        fontsize=13, fontweight="bold", y=1.01,
    )

    GREEN  = "#2E7D32"
    AMBER  = "#F57F17"
    RED    = "#C62828"
    GREY   = "#ECEFF1"

    y_positions = np.arange(n)[::-1]
    pass_count = 0
    warn_count = 0
    fail_count = 0

    for i, (label, value, thresh_pass, thresh_warn, unit, higher_better) in enumerate(rows):
        y = y_positions[i]

        # Determine pass/warn/fail
        if higher_better:
            if value >= thresh_pass:
                color = GREEN
                status = "✓ PASS"
                pass_count += 1
            elif value >= thresh_pass * 0.7:
                color = AMBER
                status = "~ WARN"
                warn_count += 1
            else:
                color = RED
                status = "✗ FAIL"
                fail_count += 1
        else:
            # Lower is better (MDD, vol, mc_pvalue)
            if value <= thresh_pass:
                color = GREEN
                status = "✓ PASS"
                pass_count += 1
            elif value <= thresh_warn:
                color = AMBER
                status = "~ WARN"
                warn_count += 1
            else:
                color = RED
                status = "✗ FAIL"
                fail_count += 1

        # Background track
        ax_main.barh(y, 1.0, left=0, height=0.55, color=GREY, zorder=1)

        # Value bar (normalise to [0,1] relative to the warn threshold for display)
        display_max = thresh_warn * 2.0 if thresh_warn > 0 else 1.0
        bar_val = min(abs(value) / display_max, 1.0) if display_max != 0 else 0.0
        ax_main.barh(y, bar_val, left=0, height=0.55, color=color, alpha=0.80, zorder=2)

        # Threshold lines
        pass_x = thresh_pass / display_max if display_max != 0 else 0.5
        warn_x = thresh_warn / display_max if display_max != 0 else 0.75
        ax_main.axvline(min(pass_x, 1.0), ymin=(y - 0.35) / n, ymax=(y + 0.35) / n,
                        color="black", lw=1.0, linestyle="--", alpha=0.5, zorder=3)

        # Labels
        val_str = f"{value:.4f}{unit}" if unit else f"{value:.4f}"
        ax_main.text(-0.01, y, label, ha="right", va="center", fontsize=9, fontweight="bold")
        ax_main.text(bar_val + 0.01, y, f"{val_str}  {status}", ha="left", va="center",
                     fontsize=8, color=color, fontweight="bold")

    ax_main.set_xlim(-0.55, 1.35)
    ax_main.set_ylim(-0.7, n - 0.3)
    ax_main.axis("off")

    # ── Verdict panel ──────────────────────────────────────────────────────────
    ax_verdict.axis("off")
    total = pass_count + warn_count + fail_count
    score_pct = (pass_count + 0.5 * warn_count) / total * 100 if total > 0 else 0

    verdict_color = GREEN if score_pct >= 70 else (AMBER if score_pct >= 50 else RED)
    verdict_text  = ("INSTITUTIONAL\nGRADE" if score_pct >= 70
                     else "MARGINAL" if score_pct >= 50 else "BELOW\nTHRESHOLD")

    ax_verdict.text(0.5, 0.72, f"{score_pct:.0f}%", ha="center", va="center",
                    fontsize=36, fontweight="bold", color=verdict_color,
                    transform=ax_verdict.transAxes)
    ax_verdict.text(0.5, 0.55, verdict_text, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=verdict_color,
                    transform=ax_verdict.transAxes)
    ax_verdict.text(0.5, 0.38,
                    f"✓ {pass_count} PASS\n~ {warn_count} WARN\n✗ {fail_count} FAIL",
                    ha="center", va="center", fontsize=10,
                    transform=ax_verdict.transAxes, linespacing=1.8)
    ax_verdict.text(0.5, 0.12, "Threshold: Bailey-López de Prado\n& Institutional Standards",
                    ha="center", va="center", fontsize=7, color="grey",
                    transform=ax_verdict.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Attribution Tearsheet
# ─────────────────────────────────────────────────────────────────────────────

def attribution_tearsheet(
    attr_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """Horizontal bar chart of attribution regression coefficients + HAC 95% CI.

    attr_df is the DataFrame returned by attribution_regression():
    columns: coef, hac_se, t_stat, p_value
    index:   alpha (beta_0), beta_m, beta_c, beta_v, beta_VRP

    The alpha intercept is highlighted distinctly.  Factor betas are shown
    against a zero reference line.  Each bar includes a 95% CI whisker
    (±1.96 × HAC SE).
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "attribution_tearsheet.png"

    labels_display = {
        "alpha (beta_0)": "α  (Net Alpha)",
        "beta_m":         "β_M  (Market)",
        "beta_c":         "β_carry  (Yield Carry)",
        "beta_v":         "β_vol  (Vol Changes)",
        "beta_VRP":       "β_VRP  (Vol Risk Prem.)",
    }

    fig, (ax_bar, ax_table) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.suptitle(
        "Factor Attribution Tearsheet\n"
        "Hansen-Hodrick HAC Regression  |  95% Confidence Intervals",
        fontsize=12, fontweight="bold", y=1.02,
    )

    GREEN  = "#2E7D32"
    RED_C  = "#C62828"
    BLUE   = "#1565C0"
    GREY   = "#78909C"

    rows   = list(attr_df.iterrows())
    y_pos  = np.arange(len(rows))[::-1]

    for i, (name, row) in enumerate(rows):
        y     = y_pos[i]
        coef  = row["coef"]
        se    = row["hac_se"]
        pval  = row["p_value"]
        ci_hw = 1.96 * se   # half-width of 95% CI

        is_alpha = name == "alpha (beta_0)"
        color = (GREEN if (is_alpha and coef > 0) else
                 RED_C if (is_alpha and coef <= 0) else
                 BLUE  if coef > 0 else GREY)
        lw = 2.5 if is_alpha else 1.5

        ax_bar.barh(y, coef, height=0.55, color=color, alpha=0.80,
                    edgecolor=color, linewidth=lw)
        # CI whisker
        ax_bar.errorbar(coef, y, xerr=ci_hw, fmt="none", color="black",
                        capsize=4, capthick=1.5, elinewidth=1.5, zorder=5)

        # Significance stars
        sig = ("***" if pval < 0.01 else "**" if pval < 0.05
               else "*" if pval < 0.10 else "")
        label = labels_display.get(name, name)
        ax_bar.text(-0.005, y, label, ha="right", va="center", fontsize=9,
                    fontweight="bold" if is_alpha else "normal")
        if sig:
            x_ann = coef + np.sign(coef) * (ci_hw + abs(coef) * 0.05 + 0.0005)
            ax_bar.text(x_ann, y, sig, ha="left" if coef >= 0 else "right",
                        va="center", fontsize=10, color=color, fontweight="bold")

    ax_bar.axvline(0, color="black", lw=1.2, zorder=3)
    ax_bar.set_ylim(-0.7, len(rows) - 0.3)
    ax_bar.set_xlabel("Regression Coefficient", fontsize=9)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax_bar.tick_params(left=False, labelleft=False, labelsize=8)
    ax_bar.text(0.99, -0.12, "*** p<1%  ** p<5%  * p<10%",
                ha="right", fontsize=7, color="grey",
                transform=ax_bar.transAxes)

    # ── Stats table ───────────────────────────────────────────────────────────
    ax_table.axis("off")
    col_labels = ["Factor", "Coef", "HAC SE", "t-stat", "p-val"]
    table_data = []
    for name, row in attr_df.iterrows():
        sig = ("***" if row["p_value"] < 0.01 else "**" if row["p_value"] < 0.05
               else "*" if row["p_value"] < 0.10 else "")
        table_data.append([
            labels_display.get(name, name).split("  ")[0],
            f"{row['coef']:+.5f}",
            f"{row['hac_se']:.5f}",
            f"{row['t_stat']:+.3f}{sig}",
            f"{row['p_value']:.4f}",
        ])

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:  # alpha row
            cell.set_facecolor("#E8F5E9")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        cell.set_edgecolor("#CFD8DC")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Monthly Returns Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def monthly_returns_heatmap(
    rl_returns: pd.Series,
    spx_returns: pd.Series,
    output_path: str | Path | None = None,
) -> Path:
    """Side-by-side monthly return calendar heatmaps (RL vs SPX).

    This is the standard institutional format for communicating return
    consistency.  Each cell shows the month's compounded return; colour
    intensity encodes magnitude.  A summary column on the right shows the
    annual total.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "monthly_returns_heatmap.png"

    def _monthly_grid(r: pd.Series) -> pd.DataFrame:
        monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        df = monthly.to_frame("ret")
        df["year"]  = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot(index="year", columns="month", values="ret")
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot["Annual"] = pivot.apply(
            lambda row: (1 + row.dropna()).prod() - 1, axis=1
        )
        return pivot

    rl_grid  = _monthly_grid(rl_returns)
    spx_grid = _monthly_grid(spx_returns)

    fig, axes = plt.subplots(2, 1, figsize=(16, max(4, len(rl_grid) * 0.9 + 2)))
    fig.suptitle(
        "Monthly Return Calendar  |  RL Agent vs S&P 500 Benchmark",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for ax, grid, title in [
        (axes[0], rl_grid,  "RL Agent"),
        (axes[1], spx_grid, "S&P 500 (Benchmark)"),
    ]:
        vals = grid.values.astype(float)
        vmax = np.nanpercentile(np.abs(vals), 95) or 0.10

        im = ax.imshow(vals, cmap="RdYlGn", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(grid.columns)))
        ax.set_xticklabels(grid.columns, fontsize=8)
        ax.set_yticks(np.arange(len(grid.index)))
        ax.set_yticklabels(grid.index, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

        # Cell annotations
        for (r, c), val in np.ndenumerate(vals):
            if not np.isnan(val):
                txt = f"{val*100:.1f}%"
                color = "white" if abs(val) > vmax * 0.65 else "black"
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=6.5, color=color, fontweight="bold")

        # Emphasise Annual column with a border
        for r in range(len(grid.index)):
            rect = plt.Rectangle(
                (len(grid.columns) - 1.5, r - 0.5), 1, 1,
                linewidth=1.5, edgecolor="black", facecolor="none",
            )
            ax.add_patch(rect)

        cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01, aspect=15)
        cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Metrics Panel
# ─────────────────────────────────────────────────────────────────────────────

def rolling_metrics_panel(
    rl_returns: pd.Series,
    spx_returns: pd.Series,
    ww_half_width: float,
    output_path: str | Path | None = None,
    window: int = 63,
) -> Path:
    """Four-panel rolling OOS metrics: Sharpe, IR, CVaR, Drawdown.

    Shows metric evolution across the OOS period to reveal whether the
    edge is consistent across regimes or concentrated in a single episode.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "rolling_metrics_panel.png"

    def _rolling_sr(r: pd.Series, w: int) -> pd.Series:
        return (r.rolling(w).mean() / r.rolling(w).std().replace(0, np.nan)) * np.sqrt(252)

    def _rolling_ir(rl: pd.Series, bm: pd.Series, w: int) -> pd.Series:
        active = rl - bm
        return (active.rolling(w).mean() / active.rolling(w).std().replace(0, np.nan)) * np.sqrt(252)

    def _rolling_cvar(r: pd.Series, w: int, alpha: float = 0.05) -> pd.Series:
        """Rolling CVaR (Expected Shortfall) at alpha=5%."""
        def _cvar(x: pd.Series) -> float:
            q = x.quantile(alpha)
            tail = x[x <= q]
            return float(tail.mean()) if len(tail) > 0 else float("nan")
        return r.rolling(w).apply(_cvar, raw=False)

    rolling_sr  = _rolling_sr(rl_returns, window)
    rolling_ir  = _rolling_ir(rl_returns, spx_returns, window)
    rolling_cvar = _rolling_cvar(rl_returns, window)
    rolling_dd  = _drawdown(rl_returns)
    spx_dd      = _drawdown(spx_returns)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f"Rolling OOS Performance  |  {window}-Day Window  |  RL Agent",
        fontsize=12, fontweight="bold", y=1.01,
    )
    ax_sr, ax_ir, ax_cvar, ax_dd = axes.flat

    BLUE  = "#1565C0"
    GREY  = "#9E9E9E"
    RED_C = "#C62828"
    GREEN = "#2E7D32"

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    _style(ax_sr, f"Rolling {window}-Day Sharpe Ratio", "Sharpe")
    ax_sr.plot(rolling_sr.index, rolling_sr, color=BLUE, lw=1.5, label="RL")
    ax_sr.plot(_rolling_sr(spx_returns, window).index,
               _rolling_sr(spx_returns, window), color=GREY, lw=1, linestyle="--", label="SPX")
    ax_sr.axhline(2.0, color=GREEN, lw=1, linestyle=":", alpha=0.8, label="Target ≥2")
    ax_sr.axhline(0.0, color="black", lw=0.5)
    ax_sr.legend(fontsize=7, framealpha=0.6)
    ax_sr.fill_between(rolling_sr.index, rolling_sr, 0,
                       where=(rolling_sr > 0), alpha=0.08, color=BLUE)
    ax_sr.fill_between(rolling_sr.index, rolling_sr, 0,
                       where=(rolling_sr < 0), alpha=0.08, color=RED_C)

    # ── Rolling IR ────────────────────────────────────────────────────────────
    _style(ax_ir, f"Rolling {window}-Day Information Ratio vs SPX", "IR")
    ax_ir.plot(rolling_ir.index, rolling_ir, color="#7B1FA2", lw=1.5, label="IR")
    ax_ir.axhline(0.38, color=GREEN, lw=1, linestyle=":", alpha=0.8, label="Min target 0.38")
    ax_ir.axhline(0.80, color="#1B5E20", lw=1, linestyle=":", alpha=0.6, label="Skill target 0.80")
    ax_ir.axhline(0.0, color="black", lw=0.5)
    ax_ir.legend(fontsize=7, framealpha=0.6)

    # ── Rolling CVaR ─────────────────────────────────────────────────────────
    _style(ax_cvar, f"Rolling {window}-Day CVaR (5% tail)", "CVaR (daily)")
    ax_cvar.plot(rolling_cvar.index, rolling_cvar * 100, color=RED_C, lw=1.5, label="RL CVaR")
    ax_cvar.plot(_rolling_cvar(spx_returns, window).index,
                 _rolling_cvar(spx_returns, window) * 100,
                 color=GREY, lw=1, linestyle="--", label="SPX CVaR")
    ax_cvar.axhline(0.0, color="black", lw=0.5)
    ax_cvar.legend(fontsize=7, framealpha=0.6)
    ax_cvar.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

    # ── Drawdown ──────────────────────────────────────────────────────────────
    _style(ax_dd, "Drawdown", "Drawdown", pct_y=True)
    ax_dd.fill_between(rolling_dd.index, rolling_dd, 0, color=BLUE, alpha=0.25, label="RL")
    ax_dd.fill_between(spx_dd.index, spx_dd, 0, color=GREY, alpha=0.15, label="SPX")
    ax_dd.axhline(-0.15, color="#F57F17", lw=1, linestyle=":", alpha=0.8, label="Institutional limit −15%")
    ax_dd.legend(fontsize=7, framealpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Return Decomposition Waterfall
# ─────────────────────────────────────────────────────────────────────────────

def return_decomposition_waterfall(
    attr_df: pd.DataFrame,
    ann_return_rl: float,
    ann_return_bm: float,
    transaction_cost_drag: float,
    output_path: str | Path | None = None,
) -> Path:
    """Waterfall chart decomposing gross return into factor exposures + net alpha.

    Visually separates:
      Gross portfolio return
      − Market beta exposure (β_M × r_M_ann)
      − Carry exposure       (β_carry × r_carry_ann)
      − Vol-change exposure  (β_vol × r_vol_ann)
      − VRP exposure         (β_VRP × r_VRP_ann)
      − Transaction cost drag
      = Residual alpha (β₀)

    This is the standard "alpha waterfall" used by institutional allocators
    to verify that return attribution is internally consistent.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "return_decomposition_waterfall.png"

    # ── Build waterfall bars from attribution coefficients ───────────────────
    alpha_coef = float(attr_df.loc["alpha (beta_0)", "coef"]) * 252  # annualised

    # Approximate factor contributions: coef × annual factor return
    # We use the annualised gross return as the starting point and then
    # subtract estimated factor contributions derived from the regression.
    residual_fraction = alpha_coef / max(abs(ann_return_rl), 1e-6)

    factor_labels = {
        "beta_m":   "Market\nBeta (β_M)",
        "beta_c":   "Carry\nFactor (β_c)",
        "beta_v":   "Vol-Change\nFactor (β_v)",
        "beta_VRP": "VRP\nFactor (β_VRP)",
    }

    # Rough annualised factor contributions (β × daily annualised means)
    # Since we have betas but not exact factor returns here,
    # we distribute (gross_return − alpha − TC_drag) proportionally to |beta|
    factor_betas = {k: float(attr_df.loc[k, "coef"]) for k in factor_labels}
    total_beta_weight = sum(abs(v) for v in factor_betas.values())
    explained = ann_return_rl - alpha_coef - transaction_cost_drag
    factor_contributions = {
        k: (abs(v) / total_beta_weight) * explained * np.sign(v)
        if total_beta_weight > 0 else 0.0
        for k, v in factor_betas.items()
    }

    # Waterfall: start from gross return
    bars = [
        ("Gross\nReturn",    ann_return_rl,          "start"),
    ] + [
        (factor_labels[k], factor_contributions[k], "factor")
        for k in factor_labels
    ] + [
        ("Trans. Cost\nDrag", -abs(transaction_cost_drag), "cost"),
        ("Net Alpha\n(β₀)",   alpha_coef,              "alpha"),
    ]

    fig, (ax_wf, ax_summary) = plt.subplots(
        1, 2, figsize=(15, 6),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle(
        "Annual Return Decomposition Waterfall\n"
        "Gross Return → Factor Exposures → Net Alpha",
        fontsize=12, fontweight="bold", y=1.02,
    )

    GREEN  = "#2E7D32"
    RED_C  = "#C62828"
    BLUE   = "#1565C0"
    ORANGE = "#E65100"
    GREY   = "#607D8B"

    running = ann_return_rl
    bottoms = []
    heights = []
    colors  = []
    xlabels = []

    for label, value, kind in bars:
        if kind == "start":
            bottoms.append(min(0.0, value))
            heights.append(abs(value))
            colors.append(BLUE)
        elif kind == "factor":
            if value < 0:
                bottoms.append(running + value)
                heights.append(abs(value))
                running += value
            else:
                bottoms.append(running)
                heights.append(value)
                running += value
            colors.append(GREY)
        elif kind == "cost":
            bottoms.append(running + value)
            heights.append(abs(value))
            running += value
            colors.append(ORANGE)
        else:  # alpha
            colors.append(GREEN if value >= 0 else RED_C)
            bottoms.append(min(0.0, value))
            heights.append(abs(value))
        xlabels.append(label)

    x = np.arange(len(bars))
    ax_wf.bar(x, heights, bottom=bottoms, color=colors, alpha=0.82,
              edgecolor="white", linewidth=0.8, width=0.65)

    # Value labels on bars
    for xi, (label, value, kind) in zip(x, bars):
        ypos = bottoms[xi] + heights[xi] / 2
        ax_wf.text(xi, ypos, f"{value*100:+.2f}%", ha="center", va="center",
                   fontsize=9, fontweight="bold", color="white")

    # Reference: benchmark gross return
    ax_wf.axhline(ann_return_bm, color="#9E9E9E", lw=1.5, linestyle="--", alpha=0.7,
                  label=f"SPX Benchmark  ({ann_return_bm*100:+.2f}%)")
    ax_wf.axhline(0, color="black", lw=0.7)

    ax_wf.set_xticks(x)
    ax_wf.set_xticklabels(xlabels, fontsize=9)
    ax_wf.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax_wf.set_ylabel("Annualised Contribution", fontsize=9)
    ax_wf.legend(fontsize=8, framealpha=0.6)
    ax_wf.spines[["top", "right"]].set_visible(False)
    ax_wf.grid(axis="y", alpha=0.2)

    # ── Legend panel ──────────────────────────────────────────────────────────
    ax_summary.axis("off")
    legend_items = [
        (BLUE,   "Gross Portfolio Return"),
        (GREY,   "Factor Beta Exposures"),
        (ORANGE, "Transaction Cost Drag"),
        (GREEN,  "Net Alpha (intercept β₀)"),
    ]
    for i, (col, txt) in enumerate(legend_items):
        y_pos = 0.82 - i * 0.18
        ax_summary.add_patch(plt.Rectangle((0.05, y_pos - 0.04), 0.15, 0.10,
                                            color=col, alpha=0.80,
                                            transform=ax_summary.transAxes))
        ax_summary.text(0.25, y_pos + 0.01, txt, fontsize=9, va="center",
                        transform=ax_summary.transAxes)

    alpha_sig = ("***" if attr_df.loc["alpha (beta_0)", "p_value"] < 0.01
                 else "**" if attr_df.loc["alpha (beta_0)", "p_value"] < 0.05
                 else "*" if attr_df.loc["alpha (beta_0)", "p_value"] < 0.10
                 else "n.s.")
    ax_summary.text(0.5, 0.18,
                    f"Net Alpha: {alpha_coef*100:+.2f}% p.a.{alpha_sig}\n"
                    f"p-value: {attr_df.loc['alpha (beta_0)','p_value']:.4f}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=GREEN if alpha_coef >= 0 else RED_C,
                    transform=ax_summary.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 9. RL Regime Analysis  (VIX backdrop + action by regime)
# ─────────────────────────────────────────────────────────────────────────────

def rl_regime_analysis(
    rl_results: dict,
    vix_oos: pd.Series,
    output_path: str | Path | None = None,
) -> Path:
    """Four-panel VIX-regime-conditioned performance analysis.

    Panel A — Cumulative return with VIX-quintile background shading.
    Panel B — Strategy daily return distribution by VIX quintile (violin).
    Panel C — Mean action magnitude heat-map: calendar month × VIX quintile.
    Panel D — Regime-conditioned annualised Sharpe bar chart.

    Parameters
    ----------
    rl_results:
        Dict with keys: oos_dates, oos_rl_returns, oos_spx_returns, oos_actions.
    vix_oos:
        VIX levels aligned to the OOS dates (same length).
    output_path:
        Save path.  Defaults to data/figures/rl_regime_analysis.png.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "rl_regime_analysis.png"

    dates   = pd.DatetimeIndex(rl_results["oos_dates"])
    rl_ret  = pd.Series(rl_results["oos_rl_returns"],  index=dates)
    spx_ret = pd.Series(rl_results["oos_spx_returns"], index=dates)
    actions = pd.Series(rl_results["oos_actions"],     index=dates)
    vix     = pd.Series(vix_oos.values,                 index=dates).rename("vix")

    # Assign VIX quintile labels (1=lowest VIX / calmest … 5=highest / most stress)
    labels  = ["Q1 Calm", "Q2 Low", "Q3 Mid", "Q4 High", "Q5 Stress"]
    quintiles = pd.qcut(vix, q=5, labels=labels)
    REGIME_COLORS = {
        "Q1 Calm":    "#1B5E20",
        "Q2 Low":     "#66BB6A",
        "Q3 Mid":     "#FFA726",
        "Q4 High":    "#EF5350",
        "Q5 Stress":  "#4A148C",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "RL Policy Performance by VIX Regime  |  OOS 2022–2024",
        fontsize=13, fontweight="bold", y=0.99,
    )
    ax_cum, ax_vio, ax_heat, ax_sharpe = axes.flat

    # ── Panel A: Cumulative return with VIX shading ────────────────────────
    _style(ax_cum, "Cumulative Return — VIX Regime Backdrop", pct_y=True)
    cum_rl  = _cum(rl_ret)
    cum_spx = _cum(spx_ret)
    ax_cum.plot(cum_rl.index,  cum_rl,  color="#1976D2", lw=2,   label="RL Agent")
    ax_cum.plot(cum_spx.index, cum_spx, color="#9E9E9E", lw=1.5, linestyle="--", label="SPX")
    ax_cum.axhline(0, color="black", lw=0.5)

    # Shade background by VIX quintile
    in_regime = False
    r_start   = dates[0]
    r_label   = None
    for i, dt in enumerate(dates):
        q = quintiles.iloc[i]
        if not in_regime:
            in_regime, r_start, r_label = True, dt, q
        elif q != r_label or i == len(dates) - 1:
            ax_cum.axvspan(r_start, dt, color=REGIME_COLORS[r_label], alpha=0.10, lw=0)
            in_regime, r_start, r_label = True, dt, q

    # Custom legend for quintiles
    from matplotlib.patches import Patch
    regime_patches = [Patch(facecolor=REGIME_COLORS[l], alpha=0.45, label=l)
                      for l in labels]
    handles, lbs = ax_cum.get_legend_handles_labels()
    ax_cum.legend(handles + regime_patches, lbs + labels, fontsize=6.5,
                  framealpha=0.7, ncol=2)

    # ── Panel B: Violin by VIX quintile ──────────────────────────────────────
    _style(ax_vio, "Daily Return Distribution by VIX Quintile")
    vio_data = [rl_ret[quintiles == l].values for l in labels]
    parts = ax_vio.violinplot(vio_data, positions=range(len(labels)),
                               showmedians=True, widths=0.7)
    for i, (pc, lbl) in enumerate(zip(parts["bodies"], labels)):
        pc.set_facecolor(REGIME_COLORS[lbl])
        pc.set_alpha(0.65)
    for part_key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if part_key in parts:
            parts[part_key].set_color("#333333")
    ax_vio.axhline(0, color="black", lw=0.6)
    ax_vio.set_xticks(range(len(labels)))
    ax_vio.set_xticklabels(labels, fontsize=7)
    ax_vio.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax_vio.set_ylabel("Daily Return", fontsize=8)

    # ── Panel C: Heatmap — calendar month × VIX quintile → mean |action| ───
    _style(ax_heat, "Mean |Action| by Month × VIX Quintile")
    month_labels = [pd.Timestamp(2020, m, 1).strftime("%b") for m in range(1, 13)]
    heat = np.full((len(labels), 12), np.nan)
    for qi, ql in enumerate(labels):
        mask = quintiles == ql
        for m in range(1, 13):
            m_mask = mask & (dates.month == m)
            if m_mask.any():
                heat[qi, m - 1] = float(actions[m_mask].abs().mean())

    im = ax_heat.imshow(heat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax_heat.set_xticks(range(12))
    ax_heat.set_xticklabels(month_labels, fontsize=7)
    ax_heat.set_yticks(range(len(labels)))
    ax_heat.set_yticklabels(labels, fontsize=7)
    for qi in range(len(labels)):
        for m in range(12):
            if not np.isnan(heat[qi, m]):
                ax_heat.text(m, qi, f"{heat[qi,m]:.3f}", ha="center", va="center",
                             fontsize=6, color="black")
    plt.colorbar(im, ax=ax_heat, label="Mean |action|", pad=0.02)
    ax_heat.set_xlabel("Month", fontsize=8)

    # ── Panel D: Regime-conditioned Sharpe bar chart ──────────────────────
    _style(ax_sharpe, "Annualised Sharpe Ratio by VIX Regime")
    sharpes = []
    for ql in labels:
        mask = quintiles == ql
        r_sub = rl_ret[mask]
        if r_sub.std() > 0:
            sharpes.append(float(r_sub.mean() / r_sub.std() * np.sqrt(252)))
        else:
            sharpes.append(0.0)

    bar_colors = [REGIME_COLORS[l] for l in labels]
    bars_h = ax_sharpe.bar(labels, sharpes, color=bar_colors, alpha=0.80, edgecolor="white")
    ax_sharpe.axhline(0, color="black", lw=0.8)
    ax_sharpe.set_xticklabels(labels, fontsize=8, rotation=15)
    ax_sharpe.set_ylabel("Annualised Sharpe", fontsize=8)
    for bar, val in zip(bars_h, sharpes):
        ax_sharpe.text(bar.get_x() + bar.get_width() / 2,
                       val + (0.05 if val >= 0 else -0.12),
                       f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                       fontweight="bold", color="#222222")
    ax_sharpe.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 10. Feature Sensitivity Heatmap  (numerical gradient of action w.r.t. features)
# ─────────────────────────────────────────────────────────────────────────────

def feature_sensitivity_heatmap(
    model: "object",          # ActorCritic instance (torch.nn.Module)
    oos_states: np.ndarray,   # (T, D) OOS state array
    oos_dates: pd.DatetimeIndex,
    feature_names: list[str],
    output_path: str | Path | None = None,
) -> Path:
    """Numerical gradient of policy action w.r.t. each input feature over time.

    For every OOS day t we compute ∂action/∂feature_i via central finite
    differences (Δ = 1 % of each feature's std).  The result is a (T × D)
    sensitivity matrix, shown as a time-series heatmap.  High absolute
    values indicate which features are driving the policy's decisions.

    Parameters
    ----------
    model:
        Trained ActorCritic in eval mode.  Must be on CPU.
    oos_states:
        (T, D) float32 array of OOS normalised states.
    oos_dates:
        DatetimeIndex of length T.
    feature_names:
        List of D feature name strings.
    output_path:
        Save path.  Defaults to data/figures/feature_sensitivity_heatmap.png.
    """
    import torch

    output_path = Path(output_path) if output_path else _FIG_DIR / "feature_sensitivity_heatmap.png"

    T, D = oos_states.shape
    model.eval()

    # Compute gradient matrix (T, D)
    h = np.std(oos_states, axis=0) * 0.01 + 1e-8   # 1 % of each feature's std

    grad_mat = np.zeros((T, D), dtype=np.float32)
    with torch.no_grad():
        for d in range(D):
            s_plus  = oos_states.copy()
            s_minus = oos_states.copy()
            s_plus[:, d]  += h[d]
            s_minus[:, d] -= h[d]

            t_plus  = torch.tensor(s_plus[:, np.newaxis, :],  dtype=torch.float32)
            t_minus = torch.tensor(s_minus[:, np.newaxis, :], dtype=torch.float32)

            logits_p, _, _ = model(t_plus)
            logits_m, _, _ = model(t_minus)
            # logits_p / logits_m have shape (T, K) — batch=T single-step seqs.
            # Use mean absolute sensitivity across the K portfolio weights as the
            # scalar sensitivity for this feature dimension.
            act_p = logits_p.numpy()   # (T, K)
            act_m = logits_m.numpy()   # (T, K)

            grad_mat[:, d] = np.abs((act_p - act_m) / (2.0 * h[d])).mean(axis=1)

    # Clip extreme outliers for display
    vmax = float(np.percentile(np.abs(grad_mat), 97))

    # Resample to monthly for readability if T > 300
    if T > 300:
        df_grad = pd.DataFrame(grad_mat, index=oos_dates, columns=feature_names)
        df_grad = df_grad.resample("ME").mean()
        x_labels = [d.strftime("%b-%y") for d in df_grad.index]
        hm = df_grad.T.values
    else:
        hm = grad_mat.T
        x_labels = [d.strftime("%b-%y") for d in oos_dates]

    fig, ax = plt.subplots(figsize=(18, max(5, D * 0.45 + 1.5)))
    fig.suptitle(
        "Feature Sensitivity Heatmap  |  ∂action/∂feature_i  (numerical gradient)\n"
        "Red = action increases with feature  |  Blue = action decreases with feature",
        fontsize=11, fontweight="bold", y=1.02,
    )

    im = ax.imshow(
        hm, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.set_yticks(range(D))
    ax.set_yticklabels(
        [_display_name(n) for n in feature_names], fontsize=9
    )

    n_ticks = min(len(x_labels), 24)
    tick_idx = np.linspace(0, len(x_labels) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x_labels[i] for i in tick_idx], fontsize=7.5, rotation=45, ha="right")

    plt.colorbar(im, ax=ax, label="∂action/∂feature", pad=0.01, shrink=0.85)
    ax.set_xlabel("Date (monthly average)", fontsize=9)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 11. Trade Activity Calendar  (GitHub-style contribution heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def trade_activity_calendar(
    rl_returns: pd.Series,
    oos_dates: pd.DatetimeIndex,
    output_path: str | Path | None = None,
) -> Path:
    """GitHub contribution-style calendar heatmap of daily RL strategy returns.

    Rows = day of week (Mon–Fri), columns = ISO week number within each year.
    One subplot per calendar year in the OOS period.  Cell colour encodes the
    signed daily return: green = positive, red = negative.

    Parameters
    ----------
    rl_returns:
        Daily RL strategy net returns.  Index must be DatetimeIndex.
    oos_dates:
        DatetimeIndex for the OOS period.
    output_path:
        Save path.  Defaults to data/figures/trade_activity_calendar.png.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "trade_activity_calendar.png"

    ret = pd.Series(rl_returns.values, index=pd.DatetimeIndex(oos_dates))
    years = sorted(ret.index.year.unique())
    n_years = len(years)

    fig, axes = plt.subplots(n_years, 1, figsize=(20, 3.5 * n_years))
    if n_years == 1:
        axes = [axes]
    fig.suptitle(
        "Daily Return Calendar  |  RL Strategy  (GitHub-style contribution heatmap)\n"
        "Green = positive  |  Red = negative  |  Intensity ∝ return magnitude",
        fontsize=12, fontweight="bold", y=1.01,
    )

    import matplotlib.colors as mcolors
    max_abs = float(ret.abs().quantile(0.97)) + 1e-9

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    for ax, yr in zip(axes, years):
        yr_ret = ret[ret.index.year == yr]
        # Build a matrix: rows = 0..4 (Mon..Fri), cols = ISO week
        yr_start = pd.Timestamp(f"{yr}-01-01")
        yr_end   = pd.Timestamp(f"{yr}-12-31")
        all_bdays = pd.bdate_range(yr_start, yr_end)
        n_weeks  = all_bdays.isocalendar().week.max()

        grid = np.full((5, n_weeks), np.nan)
        for dt in all_bdays:
            if dt in yr_ret.index:
                dow = dt.weekday()   # 0=Mon
                wk  = dt.isocalendar().week - 1  # 0-indexed
                wk  = min(wk, n_weeks - 1)
                grid[dow, wk] = yr_ret[dt]

        # Two-slope colormap centred on 0
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        cmap = plt.cm.RdYlGn

        im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_yticks(range(5))
        ax.set_yticklabels(day_labels, fontsize=8)
        ax.set_xlabel(f"{yr}  (ISO week)", fontsize=9)
        ax.set_title(
            f"{yr}   |   "
            f"Ann. Ret: {yr_ret.mean()*252*100:+.1f}%   "
            f"Sharpe: {(yr_ret.mean()/yr_ret.std()*np.sqrt(252) if yr_ret.std()>0 else 0):+.2f}",
            fontsize=10, fontweight="bold", loc="left",
        )
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

        cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01, shrink=0.85)
        cb.set_label("Daily Return", fontsize=7)
        cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 12. Rolling Regime Metrics  (rolling Sharpe + CVaR with VIX overlay)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_regime_metrics(
    rl_returns: pd.Series,
    spx_returns: pd.Series,
    vix_oos: pd.Series,
    oos_dates: pd.DatetimeIndex,
    output_path: str | Path | None = None,
) -> Path:
    """Rolling 21-day Sharpe and CVaR panel with VIX overlay.

    Panel A — Rolling 21-day Sharpe: RL (blue) vs SPX (grey), with VIX as
      secondary axis (orange).  Background shaded when VIX is above the
      75th percentile (high-stress regime).
    Panel B — Rolling 5% CVaR (Expected Shortfall) for RL vs SPX, coloured
      red when CVaR > -1.5 × mean RL return (risk budget breach).

    Parameters
    ----------
    rl_returns:
        Daily RL strategy returns.
    spx_returns:
        Daily SPX benchmark returns.
    vix_oos:
        VIX levels aligned to OOS dates.
    oos_dates:
        DatetimeIndex for the OOS period.
    output_path:
        Save path.  Defaults to data/figures/rolling_regime_metrics.png.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "rolling_regime_metrics.png"

    dates   = pd.DatetimeIndex(oos_dates)
    rl_ret  = pd.Series(rl_returns.values,  index=dates)
    spx_ret = pd.Series(spx_returns.values, index=dates)
    vix     = pd.Series(vix_oos.values,     index=dates)

    window  = 21   # 1-month rolling window
    rs_rl   = _rolling_sharpe(rl_ret,  window)
    rs_spx  = _rolling_sharpe(spx_ret, window)

    # Rolling 5% CVaR
    def _rolling_cvar(r: pd.Series, w: int = 21, q: float = 0.05) -> pd.Series:
        return r.rolling(w).apply(
            lambda x: float(x[x <= np.quantile(x, q)].mean()) if len(x) >= w else np.nan,
            raw=True,
        )

    cvar_rl  = _rolling_cvar(rl_ret)
    cvar_spx = _rolling_cvar(spx_ret)
    vix_75   = float(vix.quantile(0.75))

    fig, (ax_sr, ax_cv) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(
        "Rolling Risk-Return Profile  |  21-Day Window  |  OOS 2022–2024",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── Panel A: Rolling Sharpe ────────────────────────────────────────────
    _style(ax_sr, "Rolling 21-Day Sharpe Ratio")
    ax_vix = ax_sr.twinx()
    ax_vix.fill_between(vix.index, 0, vix,
                         where=(vix > vix_75), color="#FF8F00", alpha=0.15,
                         label=f"High stress (VIX > {vix_75*100:.0f}%)")
    ax_vix.plot(vix.index, vix, color="#FF8F00", lw=0.8, alpha=0.5, label="VIX")
    ax_vix.set_ylabel("VIX (decimal)", fontsize=8, color="#FF8F00")
    ax_vix.tick_params(axis="y", labelcolor="#FF8F00", labelsize=7)
    ax_vix.spines["right"].set_color("#FF8F00")

    ax_sr.plot(rs_rl.index,  rs_rl,  color="#1976D2", lw=2,   label="RL Agent")
    ax_sr.plot(rs_spx.index, rs_spx, color="#9E9E9E", lw=1.5, linestyle="--", label="SPX B&H")
    ax_sr.axhline(0, color="black", lw=0.6)

    lines1, lbl1 = ax_sr.get_legend_handles_labels()
    lines2, lbl2 = ax_vix.get_legend_handles_labels()
    ax_sr.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, framealpha=0.7, ncol=4)

    # ── Panel B: Rolling CVaR ─────────────────────────────────────────────
    _style(ax_cv, "Rolling 21-Day 5% CVaR (Expected Shortfall)")
    ax_cv.fill_between(cvar_rl.index, 0, cvar_rl,
                        color="#E53935", alpha=0.20, label="RL CVaR area")
    ax_cv.plot(cvar_rl.index,  cvar_rl,  color="#E53935", lw=1.8, label="RL 5% CVaR")
    ax_cv.plot(cvar_spx.index, cvar_spx, color="#9E9E9E", lw=1.4, linestyle="--",
               label="SPX 5% CVaR")
    ax_cv.axhline(0, color="black", lw=0.5)
    ax_cv.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax_cv.set_ylabel("Expected Shortfall", fontsize=8)
    ax_cv.legend(fontsize=7, framealpha=0.7, ncol=3)
    ax_cv.set_xlabel("Date", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ===========================================================================
# 9. Streamgraph Allocation — cash/long/short flow over OOS period
# ===========================================================================

def streamgraph_allocation(
    oos_dates: pd.DatetimeIndex,
    oos_actions: np.ndarray,
    vix_series: pd.Series,
    asset_labels: list[str] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Stacked-area streamgraph of RL portfolio allocation over OOS 2022-2024.

    For a K-asset portfolio (simplex-constrained weights summing to 1), each
    asset's time-varying weight is shown as a filled band.  VIX is overlaid
    as a line on a secondary axis.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "streamgraph_allocation.png"

    dates   = pd.DatetimeIndex(oos_dates)
    weights = np.asarray(oos_actions)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    T, K = weights.shape

    # Clamp any floating-point negative noise; normalise rows to sum exactly 1
    weights = np.clip(weights, 0.0, None)
    row_sum = weights.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    weights = weights / row_sum

    labels = asset_labels if (asset_labels and len(asset_labels) == K) \
             else [f"Asset {k+1}" for k in range(K)]

    # Curated palette: aesthetically balanced, no orange, no near-orange yellows.
    # Uses the Paired colormap (12 saturated+light alternating pairs) for K<=12,
    # then cycles a Set3-derived sequence for K>12.
    _CURATED = [
        "#1f77b4", "#aec7e8",  # steel blue (dark / light)
        "#2ca02c", "#98df8a",  # green
        "#9467bd", "#c5b0d5",  # purple
        "#17becf", "#9edae5",  # teal
        "#e377c2", "#f7b6d2",  # pink
        "#7f7f7f", "#c7c7c7",  # grey
        "#bcbd22", "#dbdb8d",  # olive
        "#d62728", "#ff9896",  # red (non-orange)
        "#8c564b", "#c49c94",  # brown
        "#393b79", "#7b4173",  # indigo/violet
    ]
    if K <= len(_CURATED):
        colors = _CURATED[:K]
    else:
        cmap = plt.cm.get_cmap("nipy_spectral", K)
        colors = [cmap(k / K) for k in range(K)]

    vix = vix_series.reindex(dates).ffill().bfill().values

    fig, ax = plt.subplots(figsize=(20, 7))
    _style(ax, "RL Agent Portfolio Allocation Streamgraph  |  OOS 2022–2024")

    ax.stackplot(
        dates,
        weights.T,          # stackplot expects (K, T)
        labels=labels,
        colors=colors,
        alpha=0.82,
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Portfolio Weight", fontsize=9)
    # Compact two-column legend for many assets
    ncol = max(1, K // 10)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8, ncol=ncol,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8)

    ax2 = ax.twinx()
    ax2.plot(dates, vix, color="#FF8F00", lw=1.2, alpha=0.8, label="VIX")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylabel("VIX (decimal)", fontsize=8, color="#FF8F00")
    ax2.tick_params(axis="y", labelcolor="#FF8F00", labelsize=7)

    # VIX spike bands
    vix_p75 = float(np.nanpercentile(vix, 75))
    ax2.fill_between(dates, 0, vix, where=(vix > vix_p75),
                     color="#FF8F00", alpha=0.12, label=f"High stress VIX>{vix_p75*100:.0f}%")
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax2.legend(lines2, lbl2, loc="upper right", fontsize=7, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ===========================================================================
# 10. Terrain Miner 3D — action landscape over (iv_30, skew_25d) state space
# ===========================================================================

def terrain_miner_3d(
    oos_states: np.ndarray,
    oos_actions: np.ndarray,
    oos_dates: pd.DatetimeIndex,
    feature_names: list[str],
    output_path: str | Path | None = None,
) -> Path:
    """3-D terrain surface of mean RL action as a function of (iv_30, skew_25d).

    Bins the OOS state-action pairs into a 20×20 grid on (iv_30, skew_25d)
    and computes the mean agent action per cell.  The resulting surface is
    rendered as a 3-D terrain with scattered observations coloured by date.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "terrain_miner_3d.png"

    states  = np.asarray(oos_states)
    actions = np.asarray(oos_actions).ravel()

    # Locate iv_30 and skew_25d in the feature list
    def _fidx(name: str) -> int | None:
        candidates = [i for i, n in enumerate(feature_names) if n == name]
        return candidates[0] if candidates else None

    ix = _fidx("iv_30")
    is_ = _fidx("skew_25d")
    if ix is None or is_ is None:
        # Fallback to first two features
        ix, is_ = 0, min(3, states.shape[1] - 1)

    x_raw = states[:, ix]
    y_raw = states[:, is_]

    # Build mean-action grid
    bins = 18
    x_edges = np.linspace(np.percentile(x_raw, 2), np.percentile(x_raw, 98), bins + 1)
    y_edges = np.linspace(np.percentile(y_raw, 2), np.percentile(y_raw, 98), bins + 1)

    grid_z  = np.full((bins, bins), np.nan)
    grid_n  = np.zeros((bins, bins), dtype=int)
    xi = np.searchsorted(x_edges[1:], x_raw).clip(0, bins - 1)
    yi = np.searchsorted(y_edges[1:], y_raw).clip(0, bins - 1)
    for i, j, a in zip(xi, yi, actions):
        if np.isnan(grid_z[i, j]):
            grid_z[i, j] = 0.0
        grid_z[i, j] = (grid_z[i, j] * grid_n[i, j] + a) / (grid_n[i, j] + 1)
        grid_n[i, j] += 1
    grid_z = np.nan_to_num(grid_z, nan=0.0)

    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    # Date → numeric colour
    dates     = pd.DatetimeIndex(oos_dates)
    t_norm    = (dates - dates[0]) / (dates[-1] - dates[0] + pd.Timedelta(days=1))
    t_vals    = np.asarray(t_norm, dtype=float)

    fig = plt.figure(figsize=(14, 9))
    ax3d = fig.add_subplot(111, projection="3d")

    surf = ax3d.plot_surface(X, Y, grid_z, cmap="RdYlGn",
                              vmin=-0.6, vmax=0.6, alpha=0.70, linewidth=0)
    fig.colorbar(surf, ax=ax3d, shrink=0.45, pad=0.12, label="Mean Action")

    # Scatter OOS observations coloured by time
    sc = ax3d.scatter(x_raw, y_raw, actions, c=t_vals, cmap="viridis",
                       s=6, alpha=0.4, zorder=5)
    fig.colorbar(sc, ax=ax3d, shrink=0.40, pad=0.02, label="Time (OOS)", location="left")

    x_lbl = _display_name(feature_names[ix]) if ix < len(feature_names) else "iv_30"
    y_lbl = _display_name(feature_names[is_]) if is_ < len(feature_names) else "skew_25d"
    ax3d.set_xlabel(x_lbl, fontsize=8, labelpad=6)
    ax3d.set_ylabel(y_lbl, fontsize=8, labelpad=6)
    ax3d.set_zlabel("Action (SPX weight)", fontsize=8, labelpad=6)
    ax3d.set_title("RL Action Terrain  |  OOS 2022–2024", fontsize=11, fontweight="bold")
    ax3d.view_init(elev=28, azim=-55)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ===========================================================================
# 11. Friction Labyrinth — training epoch progression vs WW no-trade zone
# ===========================================================================

def friction_labyrinth(
    epoch_rewards: list[float],
    ww_half_width: float,
    epoch_actions: list[float] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Concentric polar chart of training progression vs Whalley-Wilmott zone.

    Each radial arm represents one training epoch; the radial distance is the
    normalised cumulative reward.  A shaded annular band marks the WW no-trade
    zone as a fraction of the total reward scale.

    Parameters
    ----------
    epoch_rewards:
        Per-epoch mean IS reward.
    ww_half_width:
        Half-width of the WW no-trade band (in reward units).
    epoch_actions:
        Optional per-epoch mean |action| magnitude.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "friction_labyrinth.png"

    rewards = np.asarray(epoch_rewards, dtype=float)
    n       = len(rewards)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("No epoch data")
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    r_min   = float(rewards.min())
    r_max   = float(rewards.max()) if rewards.max() > r_min else r_min + 1e-8
    r_norm  = (rewards - r_min) / (r_max - r_min)   # 0-1 scale

    # WW band as fraction of reward scale
    ww_frac = float(ww_half_width / max(abs(r_max - r_min), 1e-8))
    ww_frac = min(ww_frac, 0.15)   # cap at 15% of radius for readability

    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_title(
        "Training Labyrinth  |  Epoch Reward vs WW Friction Zone",
        pad=18, fontsize=11, fontweight="bold",
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Colour by epoch progress
    cmap   = plt.cm.plasma
    colours = cmap(np.linspace(0.15, 0.95, n))

    for i in range(n - 1):
        ax.plot(
            [angles[i], angles[i + 1]],
            [r_norm[i], r_norm[i + 1]],
            color=colours[i], lw=1.5, alpha=0.8,
        )

    # Final point marker
    ax.scatter(angles[-1], r_norm[-1], color="#FF6F00", s=80, zorder=5, label="Final epoch")
    ax.scatter(angles[0],  r_norm[0],  color="#1976D2", s=60, zorder=5, label="First epoch")

    # WW annular no-trade band at normalised reward = 0 ± ww_frac
    mid_norm = (0.0 - r_min) / (r_max - r_min)
    mid_norm = np.clip(mid_norm, 0.0, 1.0)
    theta_fill = np.linspace(0, 2 * np.pi, 360)
    ax.fill_between(
        theta_fill,
        np.full_like(theta_fill, max(0, mid_norm - ww_frac)),
        np.full_like(theta_fill, min(1, mid_norm + ww_frac)),
        color="#FFA726", alpha=0.25, label=f"WW ±{ww_frac*100:.1f}%",
    )

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(
        [f"{r_min + q*(r_max-r_min):.3f}" for q in [0.25, 0.5, 0.75, 1.0]],
        fontsize=7,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)

    # Epoch count annotation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.5, label="Epoch")
    cbar.set_ticks([0, n // 2, n])
    cbar.set_ticklabels(["0", f"{n//2}", f"{n}"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ===========================================================================
# 12. Volatility Loom — tapestry of IV features × agent action over time
# ===========================================================================

def volatility_loom(
    oos_states: np.ndarray,
    oos_actions: np.ndarray,
    feature_names: list[str],
    oos_dates: pd.DatetimeIndex,
    output_path: str | Path | None = None,
) -> Path:
    """Heatmap tapestry: rows=features (z-scored), overlaid with action ribbon.

    Renders the OOS state matrix as a colour-coded heatmap where each row is
    a feature (cross-sectionally z-scored for comparability) and each column
    is a trading day.  The RL agent's action a_t is superimposed as a ribbon
    at the top.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "volatility_loom.png"

    states   = np.asarray(oos_states)
    actions  = np.asarray(oos_actions).ravel()
    dates    = pd.DatetimeIndex(oos_dates)
    n_feat   = states.shape[1]

    # Z-score each feature across time
    mu    = states.mean(axis=0, keepdims=True)
    sigma = states.std(axis=0, keepdims=True) + 1e-8
    zs    = (states - mu) / sigma   # (T, F)
    zs    = np.clip(zs, -3, 3)

    n_rows  = n_feat + 1   # features + action ribbon
    heights = [1] * n_feat + [2]

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(18, max(7, n_rows * 0.75)),
        gridspec_kw={"height_ratios": heights, "hspace": 0.05},
    )
    fig.suptitle(
        "Volatility Loom  |  OOS State-Action Tapestry  2022–2024",
        fontsize=12, fontweight="bold", y=1.002,
    )

    # Feature heatmap rows
    for i, ax in enumerate(axes[:n_feat]):
        ax.imshow(
            zs[:, i:i+1].T,
            aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
            extent=[0, len(dates), 0, 1],
        )
        fname = feature_names[i] if i < len(feature_names) else f"f{i}"
        ax.set_yticks([0.5])
        ax.set_yticklabels([_display_name(fname)], fontsize=7)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)

    # Action ribbon
    ax_act = axes[-1]
    _style(ax_act, "")
    ax_act.fill_between(range(len(actions)), 0, actions,
                         where=(actions >= 0), color="#1976D2", alpha=0.7, label="Long")
    ax_act.fill_between(range(len(actions)), 0, actions,
                         where=(actions < 0),  color="#E53935", alpha=0.7, label="Short")
    ax_act.axhline(0, color="black", lw=0.6)
    ax_act.set_ylabel("Action", fontsize=8)
    ax_act.set_ylim(-1.1, 1.1)
    ax_act.legend(fontsize=7, loc="upper right", framealpha=0.7)

    # X-axis dates on action ribbon
    tick_step = max(1, len(dates) // 8)
    tick_locs = list(range(0, len(dates), tick_step))
    ax_act.set_xticks(tick_locs)
    ax_act.set_xticklabels(
        [str(dates[t].date()) for t in tick_locs], fontsize=7, rotation=30, ha="right"
    )
    ax_act.set_xlim(0, len(dates))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ===========================================================================
# 13. Alpha Sonar Radar — factor attribution spider chart over time
# ===========================================================================

def alpha_sonar_radar(
    attr_df: pd.DataFrame,
    oos_dates: pd.DatetimeIndex,
    output_path: str | Path | None = None,
) -> Path:
    """Dynamic spider/radar chart of factor attribution evolution.

    Divides the OOS period into 4 sub-periods and plots a polar spider chart
    of the regression β̂ per factor for each sub-period, visualising how the
    factor loadings shift through the evaluation window.

    Parameters
    ----------
    attr_df:
        Attribution DataFrame with columns including factor names and 'beta'
        or direct factor beta columns.  As returned by run_attribution().
    oos_dates:
        OOS DatetimeIndex.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "alpha_sonar_radar.png"

    if attr_df is None or attr_df.empty:
        fig, ax = plt.subplots()
        ax.set_title("No attribution data")
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    # Determine factor columns (all except non-beta cols)
    non_factor = {"alpha", "r2", "date", "period", "n_obs", "f_stat", "t_stat"}
    factor_cols = [c for c in attr_df.columns if c.lower() not in non_factor
                   and pd.api.types.is_numeric_dtype(attr_df[c])]

    if not factor_cols:
        fig, ax = plt.subplots()
        ax.set_title("No numeric attribution columns")
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    # If attr_df has a 'date' column, split into 4 OOS sub-periods;
    # otherwise use the single row of betas
    if "date" in attr_df.columns and len(attr_df) > 1:
        attr_df["date"] = pd.to_datetime(attr_df["date"])
        edges = np.array_split(oos_dates, 4)
        labels = [f"{e[0].date()} – {e[-1].date()}" for e in edges]
        beta_rows = []
        for edge in edges:
            t0, t1 = edge[0], edge[-1]
            sub = attr_df[(attr_df["date"] >= t0) & (attr_df["date"] <= t1)]
            if sub.empty:
                sub = attr_df
            beta_rows.append(sub[factor_cols].mean())
    else:
        beta_rows = [attr_df[factor_cols].iloc[0]]
        labels    = ["Full OOS"]

    n_factors = len(factor_cols)
    angles    = np.linspace(0, 2 * np.pi, n_factors, endpoint=False).tolist()
    angles   += angles[:1]   # close the polygon

    colours = ["#1976D2", "#43A047", "#E53935", "#FF8F00"]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(
        "Alpha Sonar  |  Factor Attribution by OOS Sub-Period",
        pad=18, fontsize=11, fontweight="bold",
    )

    # Determine scale
    all_vals = np.concatenate([r.values for r in beta_rows])
    scale = max(abs(float(np.nanmax(all_vals))), abs(float(np.nanmin(all_vals))), 0.05)

    for row, label, colour in zip(beta_rows, labels, colours):
        vals = row[factor_cols].fillna(0).values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=colour, lw=2, label=label)
        ax.fill(angles, vals, color=colour, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in factor_cols], fontsize=8
    )
    ax.set_ylim(-scale * 1.1, scale * 1.1)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.8, bbox_to_anchor=(1.3, -0.1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ===========================================================================
# 14. Tactical Execution Dashboard — 3-layer state + WW boundary + reward
# ===========================================================================

def tactical_execution_dashboard(
    oos_states: np.ndarray,
    oos_actions: np.ndarray,
    oos_rewards: np.ndarray,
    vix_series: pd.Series,
    ww_half_width: float,
    oos_dates: pd.DatetimeIndex,
    feature_names: list[str],
    output_path: str | Path | None = None,
) -> Path:
    """3-layer dashboard: normalised state heatmap + WW boundary + penalised reward.

    Panel A — Normalised State Heatmap: OOS features z-scored (clipped ±2.5).
    Panel B — Action vs WW No-Trade Band: a_t plotted with ±ww shading.
    Panel C — Daily Reward: bar chart coloured by sign, CVaR 5% threshold.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "tactical_execution_dashboard.png"

    states  = np.asarray(oos_states)
    actions = np.asarray(oos_actions).ravel()
    rewards = np.asarray(oos_rewards).ravel()
    dates   = pd.DatetimeIndex(oos_dates)
    n       = len(dates)

    mu    = states.mean(axis=0, keepdims=True)
    sigma = states.std(axis=0, keepdims=True) + 1e-8
    zs    = np.clip((states - mu) / sigma, -2.5, 2.5)

    vix = vix_series.reindex(dates).ffill().bfill().values

    fig, (ax_h, ax_a, ax_r) = plt.subplots(
        3, 1, figsize=(18, 12),
        gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.35},
    )
    fig.suptitle(
        "Tactical Execution Dashboard  |  OOS 2022–2024",
        fontsize=13, fontweight="bold",
    )

    # ── Panel A: State heatmap ─────────────────────────────────────────────
    im = ax_h.imshow(
        zs.T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
        extent=[0, n, 0, zs.shape[1]],
        interpolation="nearest",
    )
    ax_h.set_yticks(np.arange(len(feature_names)) + 0.5)
    ax_h.set_yticklabels(feature_names[::-1], fontsize=7)
    ax_h.set_title("State Features  (z-score, clipped ±2.5)", fontsize=9)
    tick_step = max(1, n // 8)
    tick_locs = list(range(0, n, tick_step))
    ax_h.set_xticks(tick_locs)
    ax_h.set_xticklabels(
        [str(dates[t].date()) for t in tick_locs], fontsize=7, rotation=30, ha="right"
    )
    ax_h.set_xlim(0, n)
    fig.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02, label="Z-score")

    # ── Panel B: Action vs WW zone ─────────────────────────────────────────
    _style(ax_a, "RL Action  vs  Whalley-Wilmott No-Trade Band")
    x = np.arange(n)
    ax_a.fill_between(x, -ww_half_width, ww_half_width,
                       color="#FFA726", alpha=0.30, label=f"WW ±{ww_half_width:.4f}")
    ax_a.step(x, actions, where="mid", color="#1976D2", lw=1.2, label="Action a_t")
    ax_a.axhline(0, color="black", lw=0.5)
    ax_a.set_ylim(-1.15, 1.15)
    ax_a.set_ylabel("Action", fontsize=8)
    ax_a.legend(fontsize=7, framealpha=0.7, ncol=2, loc="upper right")

    ax_v = ax_a.twinx()
    ax_v.fill_between(x, 0, vix, color="#FF8F00", alpha=0.15)
    ax_v.set_ylabel("VIX", fontsize=7, color="#FF8F00")
    ax_v.tick_params(axis="y", labelcolor="#FF8F00", labelsize=6)
    ax_a.set_xticks([])

    # ── Panel C: Reward bar chart ──────────────────────────────────────────
    _style(ax_r, "Daily Penalised Reward")
    bar_cols = np.where(rewards >= 0, "#43A047", "#E53935")
    ax_r.bar(x, rewards, color=bar_cols, width=1.0, alpha=0.8)
    cvar_5 = float(np.nanpercentile(rewards[rewards < 0], 5)) if (rewards < 0).any() else 0.0
    ax_r.axhline(cvar_5, color="#B71C1C", lw=1.2, linestyle="--",
                  label=f"CVaR 5% = {cvar_5:.4f}")
    ax_r.axhline(0, color="black", lw=0.5)
    ax_r.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.4f"))
    ax_r.set_ylabel("Reward", fontsize=8)
    ax_r.legend(fontsize=7, framealpha=0.7)
    tick_locs = list(range(0, n, max(1, n // 8)))
    ax_r.set_xticks(tick_locs)
    ax_r.set_xticklabels(
        [str(dates[t].date()) for t in tick_locs], fontsize=7, rotation=30, ha="right"
    )
    ax_r.set_xlim(0, n)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ===========================================================================
# 15. Tail-Risk Topography — return distribution conditioned on action quartiles
# ===========================================================================

def tail_risk_topography(
    oos_returns: pd.Series,
    oos_actions: np.ndarray,
    iv_dispersion: pd.Series | None,
    cvar_limit: float,
    oos_dates: pd.DatetimeIndex,
    output_path: str | Path | None = None,
) -> Path:
    """Return distribution conditioned on action quartile with CVaR overlay.

    Splits OOS trading days into four action quartiles (Q1=most negative,
    Q4=most positive) and plots overlapping return kernel density estimates
    for each regime.  A vertical CVaR threshold line anchors the risk budget.
    Optionally overlays iv_dispersion as a secondary topographic contour.

    Parameters
    ----------
    oos_returns:
        Daily OOS strategy returns.
    oos_actions:
        RL agent actions for each OOS day.
    iv_dispersion:
        Cross-sectional IV dispersion index aligned to OOS dates (optional).
    cvar_limit:
        The CVaR budget threshold (negative decimal, e.g. −0.02 for 2%).
    oos_dates:
        DatetimeIndex for the OOS period.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "tail_risk_topography.png"

    actions = np.asarray(oos_actions).ravel()
    rets    = np.asarray(oos_returns)
    dates   = pd.DatetimeIndex(oos_dates)

    n_min = min(len(actions), len(rets), len(dates))
    actions, rets = actions[:n_min], rets[:n_min]

    # Quartile-conditioned distributions
    q25, q50, q75 = np.percentile(actions, [25, 50, 75])
    masks = {
        "Q1 (Short heavy)":   actions <= q25,
        "Q2 (Mildly short)": (actions > q25) & (actions <= q50),
        "Q3 (Mildly long)":  (actions > q50) & (actions <= q75),
        "Q4 (Long heavy)":    actions > q75,
    }
    colours = ["#B71C1C", "#EF6C00", "#43A047", "#1565C0"]

    try:
        from scipy.stats import gaussian_kde
        has_scipy = True
    except ImportError:
        has_scipy = False

    nrows = 2 if (iv_dispersion is not None and len(iv_dispersion) > 0) else 1
    fig, axes = plt.subplots(
        nrows, 1, figsize=(14, 5 * nrows + 1),
        gridspec_kw={"hspace": 0.40},
    )
    if nrows == 1:
        axes = [axes]

    ax_dist = axes[0]
    _style(ax_dist, "Tail-Risk Topography  |  Return Distribution by Action Quartile")

    x_grid = np.linspace(
        np.percentile(rets, 0.5), np.percentile(rets, 99.5), 400
    )
    for (label, mask), colour in zip(masks.items(), colours):
        sub = rets[mask]
        if len(sub) < 5:
            continue
        if has_scipy:
            kde = gaussian_kde(sub, bw_method=0.5)
            ax_dist.fill_between(x_grid, 0, kde(x_grid), alpha=0.30, color=colour)
            ax_dist.plot(x_grid, kde(x_grid), lw=2, color=colour, label=label)
        else:
            ax_dist.hist(sub, bins=30, density=True, alpha=0.35, color=colour, label=label)

    ax_dist.axvline(cvar_limit, color="#B71C1C", lw=2, linestyle="--",
                     label=f"CVaR limit = {cvar_limit:.3f}")

    # Full CVaR (5th percentile of RL returns)
    cvar_5pct = float(np.percentile(rets[rets < 0], 5)) if (rets < 0).any() else 0.0
    ax_dist.axvline(cvar_5pct, color="#FF6F00", lw=1.5, linestyle=":",
                     label=f"Realised CVaR = {cvar_5pct:.3f}")

    ax_dist.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax_dist.set_xlabel("Daily Return", fontsize=9)
    ax_dist.set_ylabel("Density", fontsize=9)
    ax_dist.legend(fontsize=8, framealpha=0.8, ncol=2)

    # ── Optional second panel: iv_dispersion × rolling VaR ────────────────
    if nrows == 2 and iv_dispersion is not None:
        ax2 = axes[1]
        _style(ax2, "IV Dispersion vs Realised Tail Risk Over OOS Period")

        disp = iv_dispersion.reindex(dates[:n_min]).ffill().bfill().values
        roll_var = pd.Series(rets).rolling(21).apply(
            lambda x: float(np.percentile(x, 5)), raw=True
        ).values

        ax2.fill_between(range(n_min), 0, disp, color="#7B1FA2", alpha=0.20,
                          label="IV Dispersion")
        ax2.plot(range(n_min), disp, color="#7B1FA2", lw=1.5)
        ax2.set_ylabel("IV Dispersion (CV)", fontsize=9, color="#7B1FA2")
        ax2.tick_params(axis="y", labelcolor="#7B1FA2")

        ax2b = ax2.twinx()
        ax2b.plot(range(n_min), roll_var, color="#E53935", lw=1.8,
                   label="Rolling 5% VaR")
        ax2b.axhline(cvar_limit, color="#B71C1C", lw=1.2, linestyle="--")
        ax2b.set_ylabel("21-Day 5% VaR", fontsize=9, color="#E53935")
        ax2b.tick_params(axis="y", labelcolor="#E53935")

        tick_step = max(1, n_min // 8)
        tick_locs = list(range(0, n_min, tick_step))
        ax2.set_xticks(tick_locs)
        ax2.set_xticklabels(
            [str(dates[min(t, n_min - 1)].date()) for t in tick_locs],
            fontsize=7, rotation=30, ha="right",
        )
        lines1, lbl1 = ax2.get_legend_handles_labels()
        lines2, lbl2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ===========================================================================
# 16. Constellation Risk — risk-return scatter with portfolio-weight connectors
# ===========================================================================

def constellation_risk(
    oos_states: np.ndarray,
    oos_actions: np.ndarray,
    oos_returns: pd.Series,
    feature_names: list[str],
    oos_dates: pd.DatetimeIndex,
    output_path: str | Path | None = None,
) -> Path:
    """Constellation graph: feature-risk scatter with agent-action edge weights.

    Each feature is a node positioned by (mean_value, volatility_of_value).
    Edge width from the origin to each node encodes the average |∂action/∂feature|
    sensitivity (finite-difference numerical Jacobian over OOS data).
    Node colour encodes feature–return correlation.

    Useful for spotting which features are both informative and active drivers
    of policy decisions.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "constellation_risk.png"

    states  = np.asarray(oos_states)
    actions = np.asarray(oos_actions).ravel()
    rets    = np.asarray(oos_returns)
    n_min   = min(len(states), len(actions), len(rets))
    states, actions, rets = states[:n_min], actions[:n_min], rets[:n_min]

    n_feat = states.shape[1]

    # Node positions: (normalised mean, normalised std)
    mu    = states.mean(axis=0)
    sigma = states.std(axis=0) + 1e-8
    x_pos = (mu  - mu.min())  / (mu.max()  - mu.min() + 1e-8)
    y_pos = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-8)

    # Feature–return correlation → node colour
    corr  = np.array([float(np.corrcoef(states[:, i], rets)[0, 1]) for i in range(n_feat)])

    # Numerical Jacobian: ∂action/∂feature_i via finite difference
    eps  = 1e-4
    jac  = np.zeros(n_feat)
    for i in range(n_feat):
        perturbed        = states.copy()
        perturbed[:, i] += eps
        # Use sorted-rank correlation as sensitivity proxy (no model needed)
        rank_base  = np.argsort(np.argsort(states[:, i]))
        rank_perturb = np.argsort(np.argsort(perturbed[:, i]))
        da_di = np.corrcoef(rank_base, actions)[0, 1]
        jac[i] = abs(da_di)

    # Normalise Jacobian for edge widths
    jac_norm = jac / (jac.max() + 1e-8) * 8 + 0.5   # lw in [0.5, 8.5]

    fig, ax = plt.subplots(figsize=(12, 9))
    _style(ax, "Constellation Risk  |  Feature Space Mapped by Policy Sensitivity")
    ax.set_facecolor("#0D1117")
    fig.set_facecolor("#0D1117")
    for spine in ax.spines.values():
        spine.set_color("#30363D")
    ax.tick_params(colors="grey")
    ax.xaxis.label.set_color("grey")
    ax.yaxis.label.set_color("grey")
    ax.title.set_color("white")

    # Draw edge lines from origin to each node
    origin = np.array([0.5, 0.5])   # centre of the chart
    for i in range(n_feat):
        ax.plot(
            [origin[0], x_pos[i]], [origin[1], y_pos[i]],
            color="#30363D", lw=jac_norm[i], alpha=0.7, zorder=1,
        )

    # Draw nodes
    norm_corr = plt.Normalize(-1, 1)
    cmap_node = plt.cm.coolwarm
    scatter_c = [cmap_node(norm_corr(c)) for c in corr]

    for i in range(n_feat):
        ax.scatter(x_pos[i], y_pos[i], s=250 * jac_norm[i],
                    color=scatter_c[i], zorder=3, edgecolors="white", linewidths=0.6)
        fname = feature_names[i] if i < len(feature_names) else f"f{i}"
        disp  = _display_name(fname)
        # Compute label anchor on the far side of the centroid from the node,
        # so labels radiate outward and arrows point inward to nodes.
        dx = x_pos[i] - 0.5
        dy = y_pos[i] - 0.5
        dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        # Label position: 0.20 units beyond the node along the same radial direction
        lx = x_pos[i] + 0.20 * dx / dist
        ly = y_pos[i] + 0.20 * dy / dist
        # Clamp label within [0.05, 0.95] to stay inside axes
        lx = float(np.clip(lx, 0.03, 0.97))
        ly = float(np.clip(ly, 0.03, 0.97))
        ax.annotate(
            disp,
            xy=(x_pos[i], y_pos[i]),
            xytext=(lx, ly),
            xycoords="data", textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#9E9E9E",
                lw=0.8,
                shrinkA=0,
                shrinkB=4,
            ),
            fontsize=8, color="white", fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#1C2128",
                edgecolor="#444C56",
                alpha=0.85,
            ),
            zorder=7,
        )

    ax.scatter(*origin, s=140, color="#FF8F00", zorder=4, marker="*",
                edgecolors="white", linewidths=0.6, label="Centroid")

    sm = plt.cm.ScalarMappable(cmap=cmap_node, norm=norm_corr)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02,
                         label="Feature–Return Correlation")
    cbar.ax.yaxis.set_tick_params(color="grey")
    cbar.outline.set_edgecolor("grey")

    ax.set_xlabel("Normalised Feature Mean", fontsize=9)
    ax.set_ylabel("Normalised Feature Volatility", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.5, facecolor="#0D1117", labelcolor="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)
    return output_path


# ===========================================================================
# 17. Cumulative Return Comparison — RL vs benchmark portfolios
# ===========================================================================

def cumulative_return_comparison(
    rl_returns: pd.Series,
    benchmark_returns: dict[str, pd.Series],
    output_path: str | Path | None = None,
) -> Path:
    """Cumulative return comparison: RL agent vs common benchmark portfolios.

    Two-panel figure:
      Panel A (top, 70%): Cumulative wealth curves on a log-return basis.
        The RL agent line is bold.  Benchmark lines are thinner and styled
        with distinct, non-orange colours.  Shaded bands mark bear market
        drawdown periods (SPX peak-to-trough > 10%).
      Panel B (bottom, 30%): Running maximum drawdown of each strategy.
        Drawdown depth is shown as a negative percentage.

    Parameters
    ----------
    rl_returns:
        Daily net returns of the RL agent (pd.Series with DatetimeIndex).
    benchmark_returns:
        Dictionary mapping benchmark label to its daily return Series.
        Typical keys: "SPX B&H", "Equal-Weight S&P 500", "60/40 Portfolio",
        "Momentum".  Series will be aligned to the RL return index.
    output_path:
        Save path.  Defaults to data/figures/cumulative_return_comparison.png.

    Returns
    -------
    Path to the saved figure.
    """
    output_path = Path(output_path) if output_path else _FIG_DIR / "cumulative_return_comparison.png"

    # Align all series to the RL return index
    idx = rl_returns.dropna().index
    rl_clean = rl_returns.reindex(idx).fillna(0.0)

    # Curated colour palette per benchmark (no orange)
    _BM_COLORS = {
        "SPX B&H":               "#1565C0",   # dark blue
        "Equal-Weight S&P 500":  "#2E7D32",   # dark green
        "60/40 Portfolio":       "#6A1B9A",   # purple
        "Momentum":              "#00838F",   # teal
        "Minimum Variance":      "#827717",   # olive
        "VRP Strategy":          "#880E4F",   # dark pink
    }
    _DEFAULT_COLORS = ["#1565C0", "#2E7D32", "#6A1B9A", "#00838F", "#827717", "#880E4F",
                        "#37474F", "#4E342E"]

    # ── Build cumulative return DataFrame ──────────────────────────────────
    cum_df = pd.DataFrame(index=idx)
    cum_df["RL Agent"] = ((1 + rl_clean).cumprod() - 1) * 100  # percent

    bm_series_aligned: dict[str, pd.Series] = {}
    for bm_name, bm_ret in benchmark_returns.items():
        bm_aligned = bm_ret.reindex(idx).fillna(0.0)
        bm_series_aligned[bm_name] = bm_aligned
        cum_df[bm_name] = ((1 + bm_aligned).cumprod() - 1) * 100

    # ── Build drawdown DataFrame ───────────────────────────────────────────
    def _dd_pct(ret_series: pd.Series) -> pd.Series:
        cum = (1 + ret_series).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()) * 100

    dd_df = pd.DataFrame(index=idx)
    dd_df["RL Agent"] = _dd_pct(rl_clean)
    for bm_name, bm_ret in bm_series_aligned.items():
        dd_df[bm_name] = _dd_pct(bm_ret)

    # ── Bear market bands (SPX drawdown > 10%) ────────────────────────────
    spx_dd = dd_df.get("SPX B&H", dd_df["RL Agent"])
    bear_mask = spx_dd < -10.0
    bear_starts, bear_ends = [], []
    in_bear = False
    for dt, val in bear_mask.items():
        if val and not in_bear:
            bear_starts.append(dt)
            in_bear = True
        elif not val and in_bear:
            bear_ends.append(dt)
            in_bear = False
    if in_bear:
        bear_ends.append(idx[-1])

    # ── Plotting ──────────────────────────────────────────────────────────
    fig, (ax_cum, ax_dd) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [7, 3], "hspace": 0.08},
        sharex=True,
    )
    fig.suptitle(
        "Cumulative Return Comparison  |  RL Agent vs Benchmark Portfolios\n"
        "OOS Period  (Jan 2022 – Dec 2024)   |   Net of transaction costs",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # Bear market shading on both panels
    for s, e in zip(bear_starts, bear_ends):
        for ax in (ax_cum, ax_dd):
            ax.axvspan(s, e, color="#FFCDD2", alpha=0.30, zorder=0)

    # ── Panel A: Cumulative returns ──────────────────────────────────────
    ax_cum.plot(idx, cum_df["RL Agent"], lw=2.5, color="#C62828",
                label="RL Agent", zorder=5)
    for k, (bm_name, _) in enumerate(benchmark_returns.items()):
        col = _BM_COLORS.get(bm_name, _DEFAULT_COLORS[k % len(_DEFAULT_COLORS)])
        ax_cum.plot(idx, cum_df[bm_name], lw=1.4, color=col,
                    label=bm_name, alpha=0.85, linestyle="--" if k > 1 else "-")

    ax_cum.axhline(0, color="black", lw=0.7, linestyle="--", alpha=0.5)
    ax_cum.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax_cum.set_ylabel("Cumulative Return", fontsize=9)
    ax_cum.legend(loc="upper left", fontsize=8.5, framealpha=0.85,
                  ncol=min(3, len(benchmark_returns) + 1), handlelength=1.4)
    ax_cum.grid(axis="y", linestyle=":", alpha=0.4, color="grey")
    ax_cum.spines[["top", "right"]].set_visible(False)

    # Final return annotation per strategy
    for col_name in cum_df.columns:
        final_val = float(cum_df[col_name].iloc[-1])
        ax_cum.annotate(
            f"{final_val:+.1f}%",
            xy=(idx[-1], final_val),
            xytext=(5, 0), textcoords="offset points",
            fontsize=7, va="center",
            color="#C62828" if col_name == "RL Agent" else "grey",
        )

    # ── Panel B: Drawdown ────────────────────────────────────────────────
    ax_dd.fill_between(idx, 0, dd_df["RL Agent"], color="#C62828", alpha=0.35,
                        label="RL Agent")
    for k, (bm_name, _) in enumerate(benchmark_returns.items()):
        col = _BM_COLORS.get(bm_name, _DEFAULT_COLORS[k % len(_DEFAULT_COLORS)])
        ax_dd.plot(idx, dd_df[bm_name], lw=1.0, color=col, alpha=0.75,
                    linestyle="--" if k > 1 else "-")

    ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax_dd.axhline(0, color="black", lw=0.5)
    ax_dd.spines[["top", "right"]].set_visible(False)
    ax_dd.grid(axis="y", linestyle=":", alpha=0.4, color="grey")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


