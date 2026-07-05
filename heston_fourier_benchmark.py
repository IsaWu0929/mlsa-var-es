"""
Independent Heston benchmark via Heston (1993) characteristic function
+ COS method (Fang & Oosterlee 2008).
NOT used in main experiments; only for benchmark sanity check.
"""
import numpy as np


def heston_char_fn(omega, S, V, T, r, kappa, theta, sigma_v, rho):
    """
    Heston characteristic function for log(S_T) given (S_t=S, V_t=V).
    Heston 1993, formula (10).
    """
    iu = 1j * omega
    d = np.sqrt((rho * sigma_v * iu - kappa) ** 2 + sigma_v ** 2 * (iu + omega ** 2))
    g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)

    C = r * iu * T + (kappa * theta / sigma_v ** 2) * (
            (kappa - rho * sigma_v * iu - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = ((kappa - rho * sigma_v * iu - d) / sigma_v ** 2) * (
            (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )

    return np.exp(C + D * V + iu * np.log(S))


def heston_put_cos(S, V, K, T, r, kappa, theta, sigma_v, rho, N=128, L=10):
    """
    Heston put price via COS method (Fang & Oosterlee 2008).
    S, V can be scalars or arrays (vectorised).

    Args:
        S, V: spot price and variance (can be arrays)
        K: strike
        T: maturity
        N: number of COS terms (128 is plenty)
        L: truncation parameter (10 is safe)

    Returns:
        Put price (array, same shape as S)
    """
    S = np.atleast_1d(S).astype(float)
    V = np.atleast_1d(V).astype(float)

    x = np.log(S / K)

    # COS truncation [a, b]
    c1 = (r - 0.5 * theta) * T
    c2 = theta * T  # approximate variance of log(S_T/K)
    a = c1 - L * np.sqrt(np.abs(c2))
    b = c1 + L * np.sqrt(np.abs(c2))

    # Cosine coefficients
    n_idx = np.arange(N)
    omega = n_idx * np.pi / (b - a)

    # Char fn at omega values - vectorized over (S, V)
    # Shape: (n_paths, N)
    n_paths = S.shape[0]
    cf_vals = np.zeros((n_paths, N), dtype=complex)
    for k in range(N):
        cf_vals[:, k] = heston_char_fn(omega[k], S, V, T, r, kappa, theta, sigma_v, rho)

    # Put COS coefficients (analytical for vanilla put)
    # P(x) = K * exp(-rT) * sum_{k} Re[ phi(omega_k) * exp(i*omega_k*(x-a)) ] * V_k
    # V_k = chi_k - psi_k for put

    chi = lambda c, d: (1.0 / (1 + (omega / (b - a)) ** 2)) * (
            np.cos(omega * (d - a)) * np.exp(d) - np.cos(omega * (c - a)) * np.exp(c)
            + (omega / (b - a)) * np.sin(omega * (d - a)) * np.exp(d)
            - (omega / (b - a)) * np.sin(omega * (c - a)) * np.exp(c)
    )
    psi = lambda c, d: np.where(omega == 0, d - c,
                                (np.sin(omega * (d - a)) - np.sin(omega * (c - a))) * (b - a) / (
                                            n_idx * np.pi + 1e-300))

    # For put: integrate over [a, 0] (i.e., S_T < K)
    Uk = (2 / (b - a)) * (-chi(a, 0) + psi(a, 0))
    Uk[0] *= 0.5  # halve first term (COS convention)

    # Discount * sum
    ix = (n_idx * np.pi / (b - a))[None, :] * (x[:, None] - a)
    real_part = np.real(cf_vals * np.exp(1j * ix))

    put = K * np.exp(-r * T) * (real_part * Uk[None, :]).sum(axis=1)

    return put.squeeze()


def compute_fourier_benchmark(p, n_outer=2_000_000, seed=12345):
    """
    Heston VaR/ES benchmark using:
    - Exact-via-QE outer simulation of (S_delta, V_delta)
      (or could replace with Broadie-Kaya later)
    - Fourier-COS inner pricer (truly closed-form)

    Returns (xi_star_fourier, chi_star_fourier)
    """
    from case_study_heston import simulate_heston, HestonParams
    if not isinstance(p, HestonParams):
        p = HestonParams()

    rng = np.random.default_rng(seed)

    # Outer: simulate (S_delta, V_delta) via QE
    n_steps_outer = 4
    S_d = np.full(n_outer, p.S0)
    V_d = np.full(n_outer, p.V0)
    S_d, V_d = simulate_heston(S_d, V_d, 0, p.delta, n_outer, n_steps_outer, p, rng)

    # Inner: Fourier price for each (S_delta, V_delta)
    P_delta = heston_put_cos(S_d, V_d, p.K_strike, p.T - p.delta,
                             p.r, p.kappa, p.theta, p.sigma_v, p.rho)

    # P_0 via Fourier at t=0
    P_0 = float(heston_put_cos(np.array([p.S0]), np.array([p.V0]),
                               p.K_strike, p.T, p.r, p.kappa,
                               p.theta, p.sigma_v, p.rho))

    # Loss: X_0 = -P_0 + e^(-r*delta) * P_delta
    X_loss = -P_0 + np.exp(-p.r * p.delta) * P_delta

    xi_star = np.quantile(X_loss, p.alpha)
    chi_star = X_loss[X_loss >= xi_star].mean()

    return xi_star, chi_star


if __name__ == "__main__":
    from case_study_heston import HestonParams

    p = HestonParams()

    print("Computing Heston Fourier-COS benchmark...")
    xi_f, chi_f = compute_fourier_benchmark(p)

    print(f"\nFourier benchmark: ξ = {xi_f:.4f}, χ = {chi_f:.4f}")
    print(f"Thesis benchmark (K_proxy=512): ξ = 3.3171, χ = 4.2137")
    print(f"Difference: ξ = {abs(xi_f - 3.3171):.4f}, χ = {abs(chi_f - 4.2137):.4f}")