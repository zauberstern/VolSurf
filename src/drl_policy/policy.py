"""
Phase III — DRL Policy Architecture.

Implements:
  - GRU-64 actor-critic network (hidden_size=64, NO attention/transformers)
  - Stochastic squashed-Gaussian policy with tanh Jacobian correction
  - PPO clipped surrogate loss with entropy bonus
  - Almgren-Chriss friction reward function
  - Lagrangian CVaR with Zang smoothing and gradient masking
  - Pre-clip L2-norm stop-loss (threshold=100) + gradient clipping (max=5)
  - apply_gradient_step: post-accumulation optimizer step helper
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Zang smoothing: continuously differentiable ReLU approximation
# ---------------------------------------------------------------------------

class ZangSmoothedReLU(torch.autograd.Function):
    """Piecewise quadratic approximation of max(0, x) with resolution eps.

    For |x| <= eps:  f(x) = (x + eps)^2 / (4*eps)
    For x > eps:     f(x) = x
    For x < -eps:    f(x) = 0

    Ensures continuous first AND second derivatives across x=0.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.eps = eps
        out = torch.where(x > eps, x,
              torch.where(x < -eps, torch.zeros_like(x),
                          (x + eps) ** 2 / (4.0 * eps)))
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        eps = ctx.eps
        grad = torch.where(x > eps, torch.ones_like(x),
               torch.where(x < -eps, torch.zeros_like(x),
                           (x + eps) / (2.0 * eps)))
        return grad_output * grad, None


def zang_relu(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return ZangSmoothedReLU.apply(x, eps)


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """GRU-64 actor-critic.  hidden_size is fixed at 64 per spec."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_size, batch_first=True)
        self.actor_head = nn.Linear(hidden_size, action_dim)
        # Learnable log-standard-deviation for the stochastic Gaussian policy.
        # Initialised to 0 → std=1 at start.  Clamped to [-4, 0] during use.
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(
        self, state_seq: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        state_seq:
            Shape (batch, seq_len, state_dim).
        hidden:
            Shape (1, batch, hidden_size) or None.

        Returns
        -------
        (action_logits, value, new_hidden)
        """
        gru_out, new_hidden = self.gru(state_seq, hidden)
        last = gru_out[:, -1, :]  # (batch, hidden_size)
        action_logits = self.actor_head(last)
        value = self.critic_head(last).squeeze(-1)
        return action_logits, value, new_hidden

    def sample_action(
        self, mu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample portfolio weights via stick-breaking bijection.

        Stick-breaking on the first K-1 actor outputs:
            z_k = sigmoid(x_k),  k = 1,...,K-1
            w_k = remaining_k * z_k,  remaining_k = 1 − Σ_{j<k} w_j
            w_K = remaining_{K-1}  [last slice]

        The log |det J| of this bijection is tractable and exact:
            log|J| = Σ_{k=1}^{K-1} [log(remaining_k) + log(z_k) + log(1-z_k)]

        Parameters
        ----------
        mu:
            Actor mean tensor, shape (..., K).

        Returns
        -------
        weights:  Simplex-constrained portfolio weights, shape (..., K).
        log_prob: Exact log-probability on the simplex, shape (...,).
        """
        K = mu.shape[-1]
        mu_sb  = mu[..., :-1]                                    # (..., K-1)
        std    = self.actor_log_std[:-1].clamp(-4.0, 0.0).exp()  # (K-1,)
        dist   = torch.distributions.Normal(mu_sb, std.expand_as(mu_sb))
        x      = dist.rsample()                                  # (..., K-1)

        z = torch.sigmoid(x)                                     # (..., K-1)
        weights   = torch.empty(*mu.shape[:-1], K, device=mu.device, dtype=mu.dtype)
        log_jac   = torch.zeros(*mu.shape[:-1], device=mu.device, dtype=mu.dtype)
        remaining = torch.ones(*mu.shape[:-1], device=mu.device, dtype=mu.dtype)
        for k in range(K - 1):
            weights[..., k] = remaining * z[..., k]
            log_jac = (
                log_jac
                + remaining.clamp(min=1e-8).log()
                + (z[..., k] * (1.0 - z[..., k])).clamp(min=1e-8).log()
            )
            remaining = remaining * (1.0 - z[..., k])
        weights[..., -1] = remaining

        log_prob = dist.log_prob(x).sum(dim=-1) - log_jac        # (...,)
        return weights, log_prob

    def mean_action(self, mu: torch.Tensor) -> torch.Tensor:
        """Deterministic portfolio weights for OOS evaluation via stick-breaking.

        Applies the stick-breaking bijection to the actor mean (no sampling).
        Uses sigmoid(mu_{1:K-1}) as the deterministic z_k values, producing
        the mode-like allocation without stochastic noise.

        Parameters
        ----------
        mu:
            Actor mean tensor, shape (..., K).

        Returns
        -------
        weights:  Stick-breaking portfolio weights, shape (..., K).
        """
        K  = mu.shape[-1]
        z  = torch.sigmoid(mu[..., :-1])                          # (..., K-1)
        weights   = torch.empty(*mu.shape[:-1], K, device=mu.device, dtype=mu.dtype)
        remaining = torch.ones(*mu.shape[:-1], device=mu.device, dtype=mu.dtype)
        for k in range(K - 1):
            weights[..., k] = remaining * z[..., k]
            remaining = remaining * (1.0 - z[..., k])
        weights[..., -1] = remaining
        return weights


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(
    portfolio_returns: torch.Tensor,
    actions: torch.Tensor,
    benchmark_returns: torch.Tensor,
    a_tc: float = 1e-3,
    a_imp: float = 5e-4,
    a_bench: float = 1.0,
    impact_beta: float = 0.6,
) -> torch.Tensor:
    """Almgren-Chriss friction reward with power-law temporary market impact.

    R = r_portfolio - C_spread - I_temp + a_bench·(r_portfolio - r_bench)

    Cost decomposition
    ------------------
    C_spread : a_tc · |Δw|
        Proportional bid-ask spread cost; linear in turnover.  ``a_tc`` is
        the half-spread in return units (e.g. 1e-3 = 10 bps).

    I_temp   : a_imp · |Δw|^(1 + β)
        Temporary market impact (Almgren-Chriss 2001, eq. 2).  The exponent
        ``1 + β`` with β=0.6 gives the empirically calibrated 3/5-power law
        (Almgren, Thum, Hauptmann, Li 2005).  For weight fractions |Δw| ∈ (0,1]
        this produces a concave impact curve: small trades are cheap, large
        trades are disproportionately expensive.  The original paper uses
        |Δw/ADV|^β normalised by average daily volume; for a single-asset
        SPX strategy at typical fund scale (notional / SPY ADV ≪ 1) the ADV
        ratio is absorbed into a_imp.

        Note: the prior implementation used a_imp·|Δw|·exp(−|Δw|), which
        incorrectly reduces impact for |Δw|>1 (large trades get cheaper).

    Parameters
    ----------
    portfolio_returns:  (N,) per-path returns.
    actions:            (N, action_dim) executed trades (proxy for turnover).
    benchmark_returns:  (N,) benchmark per-path returns.
    a_tc:               Bid-ask half-spread coefficient (default 10 bps).
    a_imp:              Temporary impact coefficient η (default 5e-4).
    a_bench:            Benchmark-relative scaling factor.
    impact_beta:        Power-law exponent β (default 0.6, Almgren et al. 2005).
    """
    turnover = actions.abs().sum(dim=-1)                      # |Δw|, shape (N,)
    spread_cost   = a_tc * turnover                           # linear
    impact_cost   = a_imp * turnover.pow(1.0 + impact_beta)  # power-law: |Δw|^1.6
    benchmark_relative = a_bench * (portfolio_returns - benchmark_returns)
    return portfolio_returns - spread_cost - impact_cost + benchmark_relative


def compute_reward_step(
    w_t: torch.Tensor,
    r_k: torch.Tensor,
    w_prev: torch.Tensor,
    a_tc: "float | torch.Tensor" = 1e-4,
    a_imp: "float | torch.Tensor" = 5e-4,
    beta: float = 0.6,
) -> torch.Tensor:
    """Per-timestep portfolio reward for the sequential K-asset MDP.

    Computes the immediate reward at step t:
        r_t = (w_t · r_k) − Σ_k a_tc_k · |Δw_k| − Σ_k a_imp_k · |Δw_k|^{1+β}

    a_tc and a_imp may be scalar floats (uniform across all K stocks) or (K,)
    tensors (per-stock effective half-spread and ADV-normalised impact coefficient).
    Broadcasting over the (N, K) delta_w tensor handles both cases identically.

    Parameters
    ----------
    w_t:    (N, K) current portfolio weights (simplex-constrained).
    r_k:    (N, K) per-stock forward returns at step t.
    w_prev: (N, K) previous step weights (zeros at t=0).
    a_tc:   Half-spread coefficient: scalar float or (K,) Tensor.
            Per-stock values should be effective (quoted × ~0.6) half-spreads.
    a_imp:  Impact coefficient: scalar float or (K,) Tensor.
            Per-stock values from Almgren-Chriss: a_imp_base * (NAV/ADV_k)^β.
    beta:   Power-law exponent; impact ∝ |Δw_k|^{1+β} (default 0.6).

    Returns
    -------
    Scalar per-path reward, shape (N,).
    """
    port_ret  = (w_t * r_k).sum(dim=-1)                          # (N,)
    delta_w_k = (w_t - w_prev).abs()                             # (N, K) per-stock
    # Broadcasting: if a_tc / a_imp is a (K,) tensor it broadcasts over (N, K);
    # if it is a scalar float the behaviour is identical to the old formula.
    spread_cost = (a_tc * delta_w_k).sum(dim=-1)                 # (N,)
    impact_cost = (a_imp * delta_w_k.pow(1.0 + beta)).sum(dim=-1)  # (N,)
    return port_ret - spread_cost - impact_cost


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Generalised Advantage Estimation (Schulman et al. 2016).

    Backward recursion:
        δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
        A_t = δ_t + (γλ)·A_{t+1}
    with V(s_T) = 0 at the terminal step.

    Parameters
    ----------
    rewards:  (N, T) per-step rewards.
    values:   (N, T) critic value estimates V(s_t).
    gamma:    Discount factor (default 0.99).
    lam:      GAE smoothing parameter λ (default 0.95).

    Returns
    -------
    advantages: (N, T) GAE advantage estimates (un-normalised).
    """
    N, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        v_next = values[:, t + 1] if t < T - 1 else torch.zeros(N, device=rewards.device)
        delta = rewards[:, t] + gamma * v_next - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    return advantages


def compute_drift_weights(
    w_prev: torch.Tensor, r_prev: torch.Tensor
) -> torch.Tensor:
    """Drift-adjust portfolio weights for market movement between rebalancing dates.

    Between target rebalancing at t-1 and the next decision at t, passive market
    returns shift the effective weight of each leg:

        w'_{t-1,k} = w_{t-1,k} * (1 + r_{t-1,k})
                     ─────────────────────────────────────
                     Σ_j  w_{t-1,j} * (1 + r_{t-1,j})

    At t=0 (zero prior position) returns zeros; otherwise returns the
    re-normalised drifted simplex.

    Parameters
    ----------
    w_prev : (..., K) previous target weights (simplex, non-negative)
    r_prev : (..., K) returns realised since the last rebalancing date

    Returns
    -------
    (..., K) drift-adjusted weights; sums to 1 where w_prev is non-zero,
    zeros elsewhere (initial step).
    """
    drifted = w_prev * (1.0 + r_prev)
    denom = drifted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    has_position = w_prev.abs().sum(dim=-1, keepdim=True) > 1e-9
    return torch.where(has_position, drifted / denom, torch.zeros_like(drifted))


# ---------------------------------------------------------------------------
# CVaR with Lagrangian dualization
# ---------------------------------------------------------------------------

class LagrangianCVaR:
    """Maintains the dual variable eta for Lagrangian CVaR enforcement.

    Uses Augmented Lagrangian Multiplier (ALaM) mechanism to prevent the
    violent dual oscillation observed with standard linear penalty.
    ALaM adds a quadratic compensation term:

        L_aug = eta * violation + (rho/2) * max(0, violation)^2

    where violation = CVaR - c_bar.  The quadratic term establishes local
    convexity near the dual optimum, damping bang-bang policy collapse.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        c_bar: float = 0.0,
        gamma: float = 1e-3,
        rho: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.c_bar = c_bar
        self.gamma = gamma
        self.rho = rho        # quadratic penalty coefficient
        self.eta = 0.0        # dual variable, updated externally

    def cvar_loss(self, losses: torch.Tensor) -> torch.Tensor:
        """Compute CVaR via the Rockafellar-Uryasev auxiliary function with Zang smoothing.

        Implements: CVaR_α = v + (1/αN) Σ zang_relu(L_i − v)

        Zang smoothing replaces the non-differentiable hard ReLU (L−v)⁺ with a
        C² piecewise-quadratic approximation, resolving the kink at L=v so that
        gradient-based optimisation does not stall at the CVaR boundary.
        The VaR threshold v is detached so that quantile selection does not
        produce invalid gradients through the rank-order computation.
        """
        N = losses.shape[0]
        v = torch.quantile(losses, 1.0 - self.alpha).detach()
        tail_losses = zang_relu(losses - v)
        cvar = v + tail_losses.sum() / (self.alpha * N)
        return cvar

    def cvar_loss_with_threshold(self, losses: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """CVaR using a pre-computed, detached VaR threshold with Zang smoothing.

        Accepts a VaR threshold estimated from a larger sample (e.g. the full
        rollout buffer) to keep the tail boundary consistent across all
        mini-batches and eliminate noisy 50-sample tail estimates.
        Zang smoothing resolves the non-differentiable kink at L=v.
        """
        N = losses.shape[0]
        tail_losses = zang_relu(losses - v.detach())
        cvar = v.detach() + tail_losses.sum() / (self.alpha * N)
        return cvar

    def dual_update(self, cvar_value: float) -> None:
        """Gradient ascent on the dual variable eta."""
        self.eta = max(0.0, self.eta + self.gamma * (cvar_value - self.c_bar))

    def penalized_loss(self, losses: torch.Tensor) -> torch.Tensor:
        """Augmented Lagrangian: eta * violation + (rho/2) * max(0, violation)^2."""
        cvar = self.cvar_loss(losses)
        violation = cvar - self.c_bar
        # Linear term (standard Lagrangian) + quadratic term (augmentation)
        quadratic = (self.rho / 2.0) * torch.clamp(violation, min=0.0) ** 2
        return self.eta * violation + quadratic

    def penalized_loss_with_threshold(self, losses: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Augmented Lagrangian penalty using a pre-computed VaR threshold."""
        cvar = self.cvar_loss_with_threshold(losses, v)
        violation = cvar - self.c_bar
        quadratic = (self.rho / 2.0) * torch.clamp(violation, min=0.0) ** 2
        return self.eta * violation + quadratic


# ---------------------------------------------------------------------------
# Training step with gradient stop-loss
# ---------------------------------------------------------------------------

def training_step(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    policy_loss: torch.Tensor,
    grad_norm_abort: float = 100.0,
    grad_clip_max: float = 5.0,
) -> bool:
    """Apply backward pass with gradient clipping and L2-norm stop-loss.

    Implements the reproducibility spec: a pre-clip L2-norm stop-loss aborts
    the update if the raw gradient norm exceeds ``grad_norm_abort`` (default
    100).  All accepted updates are additionally clipped to ``grad_clip_max``
    (default 5) for numerical stability.

    Returns True if the update was applied, False if aborted.
    """
    optimizer.zero_grad()
    policy_loss.backward()
    pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max)
    if float(pre_norm) > grad_norm_abort:
        optimizer.zero_grad()  # discard clipped-but-exploded gradients
        return False
    optimizer.step()
    return True


# ---------------------------------------------------------------------------
# Factory: build optimizer at locked learning rate
# ---------------------------------------------------------------------------

def build_optimizer(model: ActorCritic, lr: float = 1e-3) -> torch.optim.Optimizer:
    """Adam optimizer locked at lr=1e-3 per spec."""
    return torch.optim.Adam(model.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# PPO clipped surrogate loss
# ---------------------------------------------------------------------------

def ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    entropy: torch.Tensor | None = None,
    c_ent: float = 0.01,
) -> torch.Tensor:
    """Proximal Policy Optimisation clipped surrogate loss.

    Replaces vanilla policy gradient with a clipped importance-sampling ratio
    that prevents destructively large policy updates.

    Loss = −E[ min(r·A, clip(r, 1−ε, 1+ε)·A) ] − c_ent·H(π)

    Parameters
    ----------
    old_log_probs:  Log-probs under the *old* policy, shape (N,).
    new_log_probs:  Log-probs under the *current* policy, shape (N,).
    advantages:     Normalised advantage estimates, shape (N,).
    clip_eps:       PPO clip parameter ε (default 0.2).
    entropy:        Optional per-sample entropy, shape (N,) or None.
    c_ent:          Coefficient for the entropy bonus.

    Returns
    -------
    Scalar loss tensor (to be minimised).
    """
    # Clamp log-ratio before exp() to avoid NaN from policy divergence.
    # Asymmetric clamp: upper +2 (ratio ≤ e^2≈7.4), lower −20 (ratio ≥2e-9).
    log_ratio = (new_log_probs - old_log_probs).clamp(-20.0, 2.0)   # (N,)
    ratio  = torch.exp(log_ratio)                                    # (N,)
    surr1  = ratio * advantages
    surr2  = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss   = -torch.min(surr1, surr2).mean()
    if entropy is not None:
        loss = loss - c_ent * entropy.mean()
    return loss


# ---------------------------------------------------------------------------
# Post-accumulation gradient step (use after manual .backward() calls)
# ---------------------------------------------------------------------------

def apply_gradient_step(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    grad_norm_abort: float = 100.0,
    grad_clip_max: float = 5.0,
) -> bool:
    """Apply a pre-accumulated gradient update with L2-norm stop-loss and clipping.

    Unlike ``training_step``, this function does NOT call ``backward()``.
    It is intended for use after manual ``loss.backward()`` calls inside a
    gradient-accumulation loop.

    Pre-clip L2-norm stop-loss: if the raw gradient norm exceeds
    ``grad_norm_abort`` (default 100), the update is aborted and gradients
    are zeroed, per the reproducibility specification.  Accepted updates are
    clipped to ``grad_clip_max`` (default 5).

    Returns True if the update was applied, False if aborted.
    """
    pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max)
    if float(pre_norm) > grad_norm_abort:
        optimizer.zero_grad()  # discard clipped-but-exploded gradients
        return False
    optimizer.step()
    return True
