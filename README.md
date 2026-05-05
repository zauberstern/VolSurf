<p align="center">
  <img src="volsurf_logo.png" width="180" alt="VolSurf"/>
</p>

# VolSurf

A friction-aware alpha generation framework for S&P 500 equity options. The system trains a deep reinforcement learning policy on a synthetic volatility surface environment and evaluates alpha out-of-sample against a four-factor attribution model.

---

## Architecture

The pipeline runs in four sequential phases.

**Phase I — Signal Gating.** Each predictor in the state vector is tested against a one-sided HAC hypothesis: $H_0: \mu_i \leq \bar{c}_i$, where $\bar{c}_i$ is the ATM bid-ask half-spread. The Newey-West variance estimator uses lag $L = \lfloor 0.75 \cdot T^{1/3} \rfloor$. Multiple-testing control is provided by the Benjamini-Yekutieli step-up FDR procedure, which is valid under arbitrary dependence.

**Phase II — VECM Simulation.** A cointegrated VAR is fit on the in-sample state history. Johansen trace statistics are corrected via the Reinsel-Ahn finite-sample adjustment $(T - np)/T$. The lag order is selected by HQIC. $N = 20\,000$ forward paths of length $T = 252$ steps are generated and post-processed with Lee (2004) SSVI no-arbitrage bounds:

$$\sigma^2_{91} \cdot 91 \;\geq\; \sigma^2_{30} \cdot 30 \quad \text{(calendar spread no-arbitrage)}$$

**Phase III — Policy Optimization.** A GRU-64 actor-critic is trained via PPO. Portfolio weights lie on the $K$-simplex via a stick-breaking bijection on the first $K-1$ actor outputs:

$$w_k = \tilde{w}_k \cdot \sigma(x_k), \qquad \tilde{w}_k = 1 - \sum_{j < k} w_j$$

The per-step reward subtracts Almgren-Chriss friction costs:

$$r_t = \sum_k w_{t,k} r_{t,k} - \beta_{\text{pen}} \bar{r}_t^M - \sum_k a_k |\Delta w_{t,k}| - \sum_k \eta_k |\Delta w_{t,k}|^{1+\beta}$$

CVaR tail risk is enforced via an Augmented Lagrangian dual variable $\lambda$ updated as $\lambda \leftarrow \max(0,\, \lambda + \gamma_t (\text{CVaR}_\alpha - \bar{c}))$ with Robbins-Monro step $\gamma_t = \gamma_0 / \sqrt{t}$.

**Phase IV — Attribution.** Out-of-sample returns are regressed on a four-factor model with Newey-West ($L=1$) standard errors:

$$r_t^{\text{RL}} = \beta_0 + \beta_m r_t^M + \beta_c r_t^{\text{carry}} + \beta_v r_t^v + \beta_{\text{VRP}} r_t^{\text{VRP}} + u_t$$

The intercept $\hat{\beta}_0$ is the strategy's true alpha net of factor exposure.

---

## Out-of-Sample Results (2022-04-01 to 2024-12-30)

| Metric | Value |
|---|---|
| Sharpe Ratio | 0.237 |
| Annual Return | 8.49% |
| Annual Vol | 35.78% |
| Max Drawdown | -45.10% |
| $\hat{\beta}_m$ | 0.832 (p < 0.001) |
| $\hat{\alpha}$ | 0.00153 (p = 0.639) |
| Outcome | Factor Exposure |

The policy earns returns primarily through market beta rather than vol-surface alpha. The simulation-to-reality transfer gap is the primary limitation: CVaR constraints are satisfied in the simulated training environment but do not bind on historical realized returns.

---

## Data Sources

| Source | Series |
|---|---|
| WRDS OptionMetrics vsurfd | ATM IV surface (30d, 91d, skew) |
| WRDS CRSP | SPX prices, constituent returns, bid-ask spreads |
| CBOE | VIX, VXO, VXN, VXD |
| FRED | 10-year Treasury yield (DGS10) |
| Ken French Library | Fama-French factors (Mkt-RF, RF) |
| OptionMetrics opvold | SPX log put-call ratio |
| OptionMetrics zero-curve | 30d/365d zero-coupon rates |

In-sample: 2003-04-03 to 2021-12-31 (4685 trading days, 91-day purge).
Out-of-sample: 2022-04-01 to 2024-12-30 (685 trading days).

---

## Usage

```bash
# Full pipeline (data ingestion -> VECM simulation -> DRL training -> evaluation)
python run_pipeline.py

# Cross-sectional portfolio backtest
python run_portfolio.py

# Walk-forward optimization
python run_wfo.py

# Ablation study
python run_ablation.py
```

All parameters are in `config.yaml`. Environment-variable overrides are documented at the top of that file.

---

## Repository Structure

```
src/
  econometrics/     Phase I: HAC gating, data ingestion, WRDS loaders
  simulation/       Phase II: VECM engine, SSVI bounds
  drl_policy/       Phase III: GRU actor-critic, PPO, Lagrangian CVaR
  evaluation/       Phase IV: attribution, Whalley-Wilmott, plots
tests/              pytest suite (231 tests)
config.yaml         master parameter file
run_pipeline.py     end-to-end orchestrator
```

---

## Requirements

Python 3.11+. Key dependencies: `torch`, `numpy`, `pandas`, `statsmodels`, `scipy`, `matplotlib`, `wrds`.

```bash
pip install -r requirements.txt
```

WRDS database credentials are required for the OptionMetrics and CRSP data fetches. Set `WRDS_USERNAME` in your environment.

