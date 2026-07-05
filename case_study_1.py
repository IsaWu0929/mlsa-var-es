"""
case_study_1.py
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
#  Closed-form benchmarks
# ---------------------------------------------------------------------------
def closed_form(alpha: float, delta: float):
    """Returns (xi_star^0, chi_star^0)."""
    F_inv_half = norm.ppf((1.0 - alpha) / 2.0)
    xi_star  = delta * (F_inv_half ** 2 - 1.0)
    mu       = np.sqrt(1.0 + xi_star / delta)
    chi_star = (2.0 * delta / (1.0 - alpha)) * (
        mu * norm.pdf(mu) + norm.cdf(-mu) - (1.0 - alpha) / 2.0
    )
    return xi_star, chi_star


# ---------------------------------------------------------------------------
#  Simulators required by mlsa_core
# ---------------------------------------------------------------------------
def make_simulators(delta: float):
    """Returns the three simulator callbacks for case study 1."""
    sqrt_d   = np.sqrt(delta)
    sqrt_1md = np.sqrt(1.0 - delta)

    # phi(y, z) = -( sqrt(delta) y + sqrt(1-delta) z )^2
    def _phi(y, z):
        return -((sqrt_d * y + sqrt_1md * z) ** 2)

    # ---- Algorithm 1 :   X_0 ~ delta * (Y^2 - 1)
    def simulate_X0(N, rng):
        Y = rng.standard_normal(N)
        return delta * (Y ** 2 - 1.0)

    # ---- Algorithm 2 :   X_h = -1 - (1/K) sum phi(Y, Z_k)
    def simulate_Xh(N, K, rng):
        Y = rng.standard_normal(N)                 # shape (N,)
        Z = rng.standard_normal((N, K))            # shape (N, K)
        # For each n, average phi(Y_n, Z_{n,k}) over k
        phi_vals = _phi(Y[:, None], Z)             # broadcasted: (N, K)
        return -1.0 - phi_vals.mean(axis=1)

    # ---- Algorithm 3 :   coupled pair  (X_{h_{l-1}}, X_{h_l})
    def simulate_coupled_pair(N, K_coarse, K_fine, rng):
        Y = rng.standard_normal(N)
        Z = rng.standard_normal((N, K_fine))
        phi_full = _phi(Y[:, None], Z)
        Xf = -1.0 - phi_full.mean(axis=1)
        Xc = -1.0 - phi_full[:, :K_coarse].mean(axis=1)
        return Xc, Xf

    return simulate_X0, simulate_Xh, simulate_coupled_pair
