"""
case_study_2.py
================
Setup: r=2%, S_0=1%, kappabar=12%, sigmabar=20%, Delta_i = 3 months,
       Tbar = 1 year, delta = 1 week, alpha = 85%.
       Day-count 30/360.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

DEFAULT_PARAMS = dict(
    r=0.02, S0=0.01, kappabar=0.12, sigmabar=0.20,
    Delta=3.0/12.0,        # 3 months in years
    Tbar=1.0,              # 1 year
    delta=7.0/360.0,       # 1 week in 30/360
    alpha=0.85,
    d=4,                   # number of coupon dates: T_i = i*Delta, i=1..4
)

# ---------------------------------------------------------------------------
#  Helpers : coupon dates, S_bar, A, N_bar
# ---------------------------------------------------------------------------
def derive_constants(p=DEFAULT_PARAMS):
    Delta_arr = np.full(p["d"], p["Delta"])
    T_arr     = np.cumsum(Delta_arr)                # T_1, T_2, ..., T_d
    rho_T     = np.exp(-p["r"] * T_arr)             # discount factors
    # S_bar  (eq. 5.6)
    num = np.sum(rho_T * Delta_arr * np.exp(p["kappabar"] * (T_arr - Delta_arr)))
    den = np.sum(rho_T * Delta_arr)
    S_bar = num / den * p["S0"]
    # A      (eq. 5.7)         ---  sum_{i=2..d}  rho_{T_i} * Delta_i * e^{kappabar T_{i-1}}
    A = np.sum(rho_T[1:] * Delta_arr[1:] *
               np.exp(p["kappabar"] * (T_arr[1:] - Delta_arr[1:])))
    # nominal N_bar : each leg = 1 at time 0
    N_bar = 1.0 / (p["S0"] * den)
    return T_arr, Delta_arr, rho_T, S_bar, A, N_bar


# ---------------------------------------------------------------------------
#  Closed-form benchmarks
# ---------------------------------------------------------------------------
def closed_form(p=DEFAULT_PARAMS):
    _, _, _, S_bar, A, N_bar = derive_constants(p)
    sd = p["sigmabar"] * np.sqrt(p["delta"])
    xi_star = N_bar * A * p["S0"] * (
        np.exp(norm.ppf(p["alpha"]) * sd - sd**2 / 2.0) - 1.0
    )
    omega   = p["S0"] + xi_star / (N_bar * A)
    eta_minus = (np.log(omega / p["S0"]) - sd**2 / 2.0) / sd
    chi_star = N_bar * A * p["S0"] * (p["alpha"] - norm.cdf(eta_minus)) / (1.0 - p["alpha"])
    return xi_star, chi_star


# ---------------------------------------------------------------------------
#  Simulators
# ---------------------------------------------------------------------------
def make_simulators(p=DEFAULT_PARAMS):
    T_arr, Delta_arr, rho_T, S_bar, A, N_bar = derive_constants(p)
    sd = p["sigmabar"] * np.sqrt(p["delta"])

    # Direct simulator for X_0 = N_bar * A * S_0 * (exp(-sigma^2 * delta /2 + sigma sqrt(delta) U) - 1)
    def simulate_X0(N, rng):
        U = rng.standard_normal(N)
        return N_bar * A * p["S0"] * (np.exp(-sd**2 / 2.0 + sd * U) - 1.0)

    # ---- nested simulator
    # phi(y, z_1,...,z_{d-1}) = N_bar * S_0 * sum_{i=2..d} rho_{T_i} * Delta_i *
    #                                e^{kappabar T_{i-1}} * (y * prod_{j<=i-1} z_j - 1)
    coeff = N_bar * p["S0"] * rho_T[1:] * Delta_arr[1:] * \
            np.exp(p["kappabar"] * (T_arr[1:] - Delta_arr[1:]))   # length d-1

    sigma_T1mdelta = p["sigmabar"] * np.sqrt(p["Delta"] - p["delta"])
    sigma_per_leg  = p["sigmabar"] * np.sqrt(Delta_arr[1:])       # length d-2 below
    # We model:  Y = exp(-sigma^2/2 * delta + sigma sqrt(delta) U_0)
    # Z_1 = exp(-sigma^2/2 (Delta - delta) + sigma sqrt(Delta - delta) U_1)
    # Z_i (i>=2) = exp(-sigma^2/2 Delta + sigma sqrt(Delta) U_i)

    def _draw_Y(rng, n):
        return np.exp(-sd**2 / 2.0 + sd * rng.standard_normal(n))

    def _draw_Z(rng, n, K):
        """
        For each of n outer samples, draw K i.i.d. copies of Z = (Z_1,...,Z_{d-1}).
        Returns array of shape (n, K, d-1).
        """
        # Z_1
        u1 = rng.standard_normal((n, K))
        Z1 = np.exp(-sigma_T1mdelta**2 / 2.0 + sigma_T1mdelta * u1)
        Zs = [Z1]
        for j in range(p["d"] - 2):
            u = rng.standard_normal((n, K))
            sj = sigma_per_leg[j + 1] if j + 1 < len(sigma_per_leg) else sigma_per_leg[-1]
            Zs.append(np.exp(-sj**2 / 2.0 + sj * u))
        return np.stack(Zs, axis=-1)   # shape (n, K, d-1)

    def _phi_avg(Y, Z):
        """
        Averages phi(Y, Z) over the K replicates.
        Y shape: (n,);  Z shape: (n, K, d-1).
        """
        n, K, dm1 = Z.shape
        # cumulative product over j=1..i-1: shape (n, K, d-1)
        cumZ = np.cumprod(Z, axis=-1)
        bracket = Y[:, None, None] * cumZ - 1.0          # (n, K, d-1)
        weighted = bracket * coeff[None, None, :]        # broadcast
        # phi = sum_i (...). then average over K.
        phi = weighted.sum(axis=-1)                      # (n, K)
        return phi.mean(axis=-1)                         # (n,)

    def simulate_Xh(N, K, rng):
        Y = _draw_Y(rng, N)
        Z = _draw_Z(rng, N, K)
        return _phi_avg(Y, Z)

    def simulate_coupled_pair(N, K_coarse, K_fine, rng):
        Y = _draw_Y(rng, N)
        Z = _draw_Z(rng, N, K_fine)
        Xf = _phi_avg(Y, Z)
        Xc = _phi_avg(Y, Z[:, :K_coarse, :])
        return Xc, Xf

    return simulate_X0, simulate_Xh, simulate_coupled_pair
