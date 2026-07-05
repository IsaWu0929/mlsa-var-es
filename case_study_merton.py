"""
case_study_merton.py
=====================
Model
-----
    dS_t = (r - lam * kappa_J) S_t dt + sigma S_t dW_t + S_{t-} dJ_t
    where   J_t  = sum_{i=1..N_t} (Y_i - 1),   N_t ~ Poisson(lam),
            ln Y_i ~ N(mu_J, sigma_J^2),  kappa_J = E[Y_i - 1] = e^{mu_J + sig_J^2/2} - 1.

Closed-form
-----------
European put price under Merton:
    P_Mert(t, S, K, T)  =  sum_{n=0}^{inf}  e^{-lam' (T-t)} (lam'(T-t))^n / n!
                            * P_BS(S, K, r_n, sigma_n, T-t)
    with
        lam'    = lam * (1 + kappa_J)
        r_n     = r - lam*kappa_J + n*ln(1+kappa_J)/(T-t)
        sigma_n = sqrt(sigma^2 + n*sigma_J^2/(T-t))

Loss:
    X_0 = -P0 + e^{-r T} (K - S_T)^+    (over the full simulation)
    Conditionally on (S_delta), the inner expectation is
    e^{-r T} * E[(K - S_T)^+ | S_delta] = P_Mert(delta, S_delta, K, T-delta) * e^{-r delta}.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from math import exp, lgamma, log


# ---------------------------------------------------------------------------
#  Parameters
# ---------------------------------------------------------------------------
@dataclass
class MertonParams:
    S0       : float = 100.0
    r        : float = 0.02
    sigma    : float = 0.20
    lam      : float = 0.5     # jump intensity
    mu_J     : float = -0.10   # mean of ln(Y)
    sigma_J  : float = 0.15    # std of ln(Y)
    T        : float = 0.25    # 3 months
    K_strike : float = 100.0
    delta    : float = 1.0 / 52
    alpha    : float = 0.975

    @property
    def kappa_J(self):
        return np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1.0


# ---------------------------------------------------------------------------
#  Black-Scholes put helper
# ---------------------------------------------------------------------------
def _bs_put(S, K, r, sigma, tau):
    if tau <= 0:
        return np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
#  Merton put price (truncated series; 50 terms is plenty)
# ---------------------------------------------------------------------------
def merton_put(S, K, T_minus_t, p: MertonParams, n_terms: int = 60):
    """Vectorised in S."""
    if T_minus_t <= 0:
        return np.maximum(K - S, 0.0)
    lam_p = p.lam * (1.0 + p.kappa_J)
    out   = np.zeros_like(np.atleast_1d(S), dtype=float)
    log_lt = np.log(lam_p * T_minus_t) if lam_p * T_minus_t > 0 else -np.inf
    for n in range(n_terms):
        # Poisson weight  e^{-lam' tau} (lam' tau)^n / n!
        if log_lt == -np.inf and n > 0:
            break
        log_w = -lam_p * T_minus_t + n * log_lt - (lgamma(n + 1) if n else 0.0)
        w     = np.exp(log_w)
        if w < 1e-16 and n > 5:
            break
        r_n     = p.r - p.lam * p.kappa_J + n * np.log(1.0 + p.kappa_J) / T_minus_t
        sigma_n = np.sqrt(p.sigma ** 2 + n * p.sigma_J ** 2 / T_minus_t)
        out    += w * _bs_put(S, K, r_n, sigma_n, T_minus_t)
    return out if np.ndim(S) > 0 else float(np.atleast_1d(out)[0])


# ---------------------------------------------------------------------------
#  Exact simulator
#     S_T  =  S_t * exp[(r - lam*kappa_J - sigma^2/2)(T-t) + sigma sqrt(T-t) Z
#                       + sum_{i=1..N} ln Y_i ]
# ---------------------------------------------------------------------------
def simulate_S_one_step(S_init, t_start, t_end, n_paths, p: MertonParams,
                        rng: np.random.Generator) -> np.ndarray:
    if t_end <= t_start:
        return np.broadcast_to(S_init, (n_paths,)).astype(float).copy()
    tau   = t_end - t_start
    drift = (p.r - p.lam * p.kappa_J - 0.5 * p.sigma ** 2) * tau
    Z     = rng.standard_normal(n_paths)
    Ns    = rng.poisson(p.lam * tau, n_paths)
    # vectorised aggregate jump:  sum_{i=1..N} ln Y_i ~ N(N*mu_J, N*sigma_J^2)
    jump_mean = Ns * p.mu_J
    jump_std  = np.sqrt(Ns) * p.sigma_J
    jumps     = jump_mean + jump_std * rng.standard_normal(n_paths)
    return np.broadcast_to(S_init, (n_paths,)) * np.exp(drift + p.sigma * np.sqrt(tau) * Z + jumps)


# ---------------------------------------------------------------------------
#  Simulators required by mlsa_core
# ---------------------------------------------------------------------------
def make_simulators(p: MertonParams):
    """Returns (simulate_X0, simulate_Xh, simulate_coupled_pair, P0)."""
    # --- exact P0 from the Merton series
    P0_arr = merton_put(np.array([p.S0]), p.K_strike, p.T, p)
    P0 = float(np.atleast_1d(P0_arr)[0])

    def simulate_X0(N, rng):
        """
        Algorithm 1 requires direct simulation of X_0 = E[loss | S_delta].
        For Merton this is exactly:
            X_0 = -P0 + e^{-r delta} * P_Mert(delta, S_delta, K, T-delta)
        which we have in closed form via the Merton series.
        """
        S_d = simulate_S_one_step(p.S0, 0.0, p.delta, N, p, rng)
        inner_exp = merton_put(S_d, p.K_strike, p.T - p.delta, p)
        return -P0 + np.exp(-p.r * p.delta) * inner_exp

    def simulate_Xh(N, K_inner, rng):
        # Memory-safe batch: keep n_b * K_inner <= 1e6
        batch = max(1, min(N, 1_000_000 // max(K_inner, 1)))
        out = np.empty(N)
        for start in range(0, N, batch):
            end = min(start + batch, N)
            n_b = end - start
            S_d  = simulate_S_one_step(p.S0, 0.0, p.delta, n_b, p, rng)
            S_d_rep = np.repeat(S_d, K_inner)
            S_T  = simulate_S_one_step(S_d_rep, p.delta, p.T, n_b * K_inner, p, rng)
            payoff = np.exp(-p.r * p.T) * np.maximum(p.K_strike - S_T, 0.0)
            out[start:end] = -P0 + payoff.reshape(n_b, K_inner).mean(axis=1)
        return out

    def simulate_coupled_pair(N, K_coarse, K_fine, rng):
        batch = max(1, min(N, 1_000_000 // max(K_fine, 1)))
        Xc_out = np.empty(N); Xf_out = np.empty(N)
        for start in range(0, N, batch):
            end = min(start + batch, N)
            n_b = end - start
            S_d  = simulate_S_one_step(p.S0, 0.0, p.delta, n_b, p, rng)
            S_d_rep = np.repeat(S_d, K_fine)
            S_T  = simulate_S_one_step(S_d_rep, p.delta, p.T, n_b * K_fine, p, rng)
            payoff = np.exp(-p.r * p.T) * np.maximum(p.K_strike - S_T, 0.0)
            payoff = payoff.reshape(n_b, K_fine)
            Xf_out[start:end] = -P0 + payoff.mean(axis=1)
            Xc_out[start:end] = -P0 + payoff[:, :K_coarse].mean(axis=1)
        return Xc_out, Xf_out

    return simulate_X0, simulate_Xh, simulate_coupled_pair, P0


# ---------------------------------------------------------------------------
#  Benchmark
# ---------------------------------------------------------------------------
def compute_benchmark(p: MertonParams, n_samples: int = 2_000_000,
                       seed: int = 99):
    rng = np.random.default_rng(seed)
    S_d = simulate_S_one_step(p.S0, 0.0, p.delta, n_samples, p, rng)
    inner_exp = merton_put(S_d, p.K_strike, p.T - p.delta, p)
    P0_arr = merton_put(np.array([p.S0]), p.K_strike, p.T, p)
    P0 = float(np.atleast_1d(P0_arr)[0])
    L = -P0 + np.exp(-p.r * p.delta) * inner_exp
    xi_star  = float(np.quantile(L, p.alpha))
    chi_star = float(L[L >= xi_star].mean())
    return xi_star, chi_star
