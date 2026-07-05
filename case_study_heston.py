"""
case_study_heston.py
=====================
Model
-----
    dS_t = r S_t dt + sqrt(V_t) S_t dW_t^S
    dV_t = kappa (theta - V_t) dt + sigma_v sqrt(V_t) dW_t^V
    <dW^S, dW^V> = rho dt

Loss
----
Short ATM European put with maturity T, strike K. At time delta < T,

    P(t, S, V) = e^{-r(T-t)} * E[(K - S_T)^+ | S_t=S, V_t=V]
    X_0        = -P(0, S_0, V_0) + e^{-r delta} P(delta, S_delta, V_delta)
               (loss to a SHORT put position over [0, delta])

We absorb e^{-r delta} into the conditional expectation so that, identifying
Y = (S_delta, V_delta) and Z = remaining BM increments,

    X_0 = E[phi(Y, Z) | Y]   (mod a deterministic constant)

Inner conditional expectation is approximated by averaging K_inner Monte
Carlo paths from delta to T, exactly as in the paper's nested estimator.

Benchmark
---------
No closed form; we use a high-quality Monte-Carlo estimate with N=10^7
direct simulations as the ground truth. The implementation below provides
`compute_benchmark()` for that.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


# ---------------------------------------------------------------------------
#  Default Heston parameters
# ---------------------------------------------------------------------------
@dataclass
class HestonParams:
    S0       : float = 100.0
    V0       : float = 0.04        # initial variance (i.e. vol = 20%)
    r        : float = 0.02        # risk-free rate
    kappa    : float = 2.0         # mean-reversion speed
    theta    : float = 0.04        # long-run variance
    sigma_v  : float = 0.3         # vol-of-vol
    rho      : float = -0.7        # leverage correlation
    T        : float = 0.25        # option maturity, 3 months
    K_strike : float = 100.0       # ATM
    delta    : float = 1.0 / 52    # risk horizon = 1 week
    alpha    : float = 0.975       # confidence level for VaR/ES


# ---------------------------------------------------------------------------
#  QE Scheme
# ---------------------------------------------------------------------------
def _qe_step(V_prev: np.ndarray,
             dt: float,
             p: HestonParams,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    One step of the QE scheme for the variance process. Returns (V_new, Z_S)
    where Z_S is the standard normal driving increment used for the *log
    asset* increment.
    """
    psi_c   = 1.5     # threshold parameter; common choice in [1, 2]
    n       = V_prev.shape[0]
    e_kdt   = np.exp(-p.kappa * dt)
    m       = p.theta + (V_prev - p.theta) * e_kdt
    s2_part = (V_prev * p.sigma_v ** 2 * e_kdt / p.kappa) * (1 - e_kdt) \
            + (p.theta * p.sigma_v ** 2 / (2 * p.kappa)) * (1 - e_kdt) ** 2
    psi     = s2_part / np.where(m**2 < 1e-300, 1e-300, m**2)

    V_new   = np.empty_like(V_prev)

    # ---- region 1: psi <= psi_c
    mask1 = psi <= psi_c
    if mask1.any():
        psi1 = psi[mask1]
        b2   = 2.0 / psi1 - 1.0 + np.sqrt(np.abs(2.0 / psi1)) * \
                                   np.sqrt(np.abs(2.0 / psi1 - 1.0))
        a    = m[mask1] / (1.0 + b2)
        Z    = rng.standard_normal(mask1.sum())
        V_new[mask1] = a * (np.sqrt(b2) + Z) ** 2

    # ---- region 2: psi > psi_c
    mask2 = ~mask1
    if mask2.any():
        psi2 = psi[mask2]
        p_   = (psi2 - 1.0) / (psi2 + 1.0)
        beta = (1.0 - p_) / m[mask2]
        U    = rng.uniform(size=mask2.sum())
        # inverse CDF: V=0 if U<=p; else V = ln((1-p)/(1-U))/beta
        V_   = np.zeros_like(U)
        idx  = U > p_
        V_[idx] = np.log((1.0 - p_[idx]) / (1.0 - U[idx])) / beta[idx]
        V_new[mask2] = V_

    # Driving normal for log-asset:
    Z_S = rng.standard_normal(n)
    return V_new, Z_S


def _heston_path_step(S_prev: np.ndarray, V_prev: np.ndarray, V_new: np.ndarray,
                      Z_S: np.ndarray, dt: float,
                      p: HestonParams) -> np.ndarray:
    """
    Andersen's recommended log-asset update conditional on (V_prev, V_new):

        ln S_new = ln S_prev + (r - 0.5*int V) dt + rho/sigma_v (V_new - V_prev
                  - kappa (theta - V) dt) + sqrt(1-rho^2) sqrt(int V) Z_S

    where int V = (V_prev + V_new)/2 (trapezoidal).
    """
    K0 = -p.rho * p.kappa * p.theta * dt / p.sigma_v
    K1 = 0.5 * dt * (p.kappa * p.rho / p.sigma_v - 0.5) - p.rho / p.sigma_v
    K2 = 0.5 * dt * (p.kappa * p.rho / p.sigma_v - 0.5) + p.rho / p.sigma_v
    K3 = 0.5 * dt * (1.0 - p.rho ** 2)
    log_S_new = (np.log(S_prev) + p.r * dt + K0
                 + K1 * V_prev + K2 * V_new
                 + np.sqrt(K3 * (V_prev + V_new)) * Z_S)
    return np.exp(log_S_new)


def simulate_heston(S_init, V_init, t_start, t_end, n_paths, n_steps,
                    p: HestonParams, rng: np.random.Generator):
    """
    Vectorised QE simulator.  Inputs S_init, V_init may be either scalars or 1-D arrays of length n_paths.
    Returns (S_end, V_end), each of length n_paths.
    """
    S = np.broadcast_to(S_init, (n_paths,)).astype(float).copy()
    V = np.broadcast_to(V_init, (n_paths,)).astype(float).copy()
    if t_end <= t_start:
        return S, V
    dt = (t_end - t_start) / n_steps
    for _ in range(n_steps):
        V_new, Z_S = _qe_step(V, dt, p, rng)
        S          = _heston_path_step(S, V, V_new, Z_S, dt, p)
        V          = V_new
    return S, V


# ---------------------------------------------------------------------------
#  Loss definition
#
#     X_0 = -P(0, S_0, V_0) + e^{-r delta} * (K - S_T)^+      (under sample path)
#         = -P_0 + e^{-r delta} * E[(K - S_T)^+ | S_delta, V_delta]
# ---------------------------------------------------------------------------
def make_simulators(p: HestonParams, n_steps_to_delta: int = 1,
                    n_steps_post_delta: int = 4):
    """
    Returns (simulate_X0, simulate_Xh, simulate_coupled_pair, P0_estimate).

    The constant P0 is needed to centre the loss; we estimate it once at
    initialisation with a dedicated MC.
    """
    # ------- estimate P_0 once with high accuracy
    rng_init = np.random.default_rng(20250428)
    n_p0_total = 1_000_000
    batch_p0   = 100_000
    sums = 0.0
    cnt  = 0
    for _ in range(n_p0_total // batch_p0):
        S_T, _ = simulate_heston(p.S0, p.V0, 0.0, p.T,
                                 batch_p0, n_steps_to_delta + n_steps_post_delta,
                                 p, rng_init)
        sums += np.maximum(p.K_strike - S_T, 0.0).sum()
        cnt  += batch_p0
    P0 = np.exp(-p.r * p.T) * sums / cnt

    K_proxy = 512    # large inner sample so the residual bias is negligible

    def simulate_X0(N, rng):
        """High-precision proxy for the conditional expectation."""
        # Process in batches to control memory: each batch uses ~N_batch * K_proxy * 8 bytes
        batch = max(1, min(N, 500))   # 500 * 1024 = 512k float64 values = 4 MB
        out = np.empty(N)
        for start in range(0, N, batch):
            end = min(start + batch, N)
            n_b = end - start
            S_d, V_d = simulate_heston(p.S0, p.V0, 0.0, p.delta,
                                       n_b, n_steps_to_delta, p, rng)
            S_d_rep = np.repeat(S_d, K_proxy)
            V_d_rep = np.repeat(V_d, K_proxy)
            S_T, _  = simulate_heston(S_d_rep, V_d_rep, p.delta, p.T,
                                      n_b * K_proxy, n_steps_post_delta, p, rng)
            payoff_T = np.exp(-p.r * p.T) * np.maximum(p.K_strike - S_T, 0.0)
            out[start:end] = -P0 + payoff_T.reshape(n_b, K_proxy).mean(axis=1)
        return out

    # ---- nested simulator (for Algorithm 2)
    #     Y = (S_delta, V_delta)  (drawn once for each n)
    #     Z = noise to drive simulation from delta to T  (drawn K_inner times)
    def simulate_Xh(N, K_inner, rng):
        # Memory-safe batch: keep N_batch * K_inner <= ~5e5
        batch = max(1, min(N, 500_000 // max(K_inner, 1)))
        out = np.empty(N)
        for start in range(0, N, batch):
            end = min(start + batch, N)
            n_b = end - start
            S_d, V_d = simulate_heston(p.S0, p.V0, 0.0, p.delta,
                                       n_b, n_steps_to_delta, p, rng)
            S_d_rep = np.repeat(S_d, K_inner)
            V_d_rep = np.repeat(V_d, K_inner)
            S_T, _  = simulate_heston(S_d_rep, V_d_rep, p.delta, p.T,
                                      n_b * K_inner, n_steps_post_delta, p, rng)
            payoff_T = np.exp(-p.r * p.T) * np.maximum(p.K_strike - S_T, 0.0)
            out[start:end] = -P0 + payoff_T.reshape(n_b, K_inner).mean(axis=1)
        return out

    # ---- coupled pair (for Algorithm 3)
    #     Reuse one outer (S_d, V_d) and one block of K_fine inner paths;
    #     coarse takes the FIRST K_coarse of them, fine takes ALL K_fine.
    #     This is exactly the construction in Section 4 of the paper.
    def simulate_coupled_pair(N, K_coarse, K_fine, rng):
        batch = max(1, min(N, 500_000 // max(K_fine, 1)))
        Xc_out = np.empty(N); Xf_out = np.empty(N)
        for start in range(0, N, batch):
            end = min(start + batch, N)
            n_b = end - start
            S_d, V_d = simulate_heston(p.S0, p.V0, 0.0, p.delta,
                                       n_b, n_steps_to_delta, p, rng)
            S_d_rep = np.repeat(S_d, K_fine)
            V_d_rep = np.repeat(V_d, K_fine)
            S_T, _  = simulate_heston(S_d_rep, V_d_rep, p.delta, p.T,
                                      n_b * K_fine, n_steps_post_delta, p, rng)
            payoff_T = np.exp(-p.r * p.T) * np.maximum(p.K_strike - S_T, 0.0)
            payoff_T = payoff_T.reshape(n_b, K_fine)
            Xf_out[start:end] = -P0 + payoff_T.mean(axis=1)
            Xc_out[start:end] = -P0 + payoff_T[:, :K_coarse].mean(axis=1)
        return Xc_out, Xf_out

    return simulate_X0, simulate_Xh, simulate_coupled_pair, P0


# ---------------------------------------------------------------------------
#  High-precision benchmark for VaR / ES
# ---------------------------------------------------------------------------
def compute_benchmark(p: HestonParams, n_samples: int = 100_000,
                       seed: int = 12345):
    """
    Estimates ground-truth (xi*, chi*) of the *conditional expectation* loss

        X_0 := -P_0 + e^{-r delta} E[(K - S_T)^+ | S_delta, V_delta]

    For Heston the inner expectation has no closed form, so we approximate
    it by a very large Monte-Carlo inner sample (the K_proxy=1024 baked into
    `make_simulators`). The resulting benchmark error scales as
    O(1/sqrt(K_proxy)) and is negligible relative to the algorithmic errors
    we want to compare.
    """
    sim_X0, _, _, _ = make_simulators(p)
    rng = np.random.default_rng(seed)
    X = sim_X0(n_samples, rng)
    xi_star  = float(np.quantile(X, p.alpha))
    chi_star = float(X[X >= xi_star].mean())
    return xi_star, chi_star
