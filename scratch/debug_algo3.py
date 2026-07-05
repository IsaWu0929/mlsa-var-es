"""
debug_algo3.py
===============
Surgical instrumentation of algo3_MLSA to reveal the bug.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from case_study_1 import closed_form, make_simulators
from mlsa_core import H1, H2

ALPHA, DELTA = 0.975, 0.5
xi_star, chi_star = closed_form(ALPHA, DELTA)
print(f"benchmark xi*  = {xi_star:.4f}")
print(f"benchmark chi* = {chi_star:.4f}")
sim_X0, sim_Xh, sim_coupled = make_simulators(DELTA)

h0, L, M = 1/32, 2, 2
gamma_factor, gamma_denom = 0.75, 9000
Ns = [100_000, 50_000, 25_000]
K_levels = [int(round(1/(h0 / M**ell))) for ell in range(L+1)]
print(f"K_levels = {K_levels}")
print(f"Ns_levels = {Ns}")
gamma = lambda n: gamma_factor / (gamma_denom + n)
rng = np.random.default_rng(42)

# level 0
print("\n=== LEVEL 0 ===")
X0 = sim_Xh(Ns[0], K_levels[0], rng)
xi_h0, chi_h0 = 0.0, 0.0
for n in range(Ns[0]):
    x = X0[n]
    gn = gamma(n + 1)
    xi_h0  -= gn * H1(xi_h0, x, ALPHA)
    chi_h0 -= 1.0 / (n + 1) * H2(chi_h0, xi_h0, x, ALPHA)
    if (n+1) in (1000, 10_000, 100_000):
        print(f"  n={n+1:>7d} xi={xi_h0:+.4f} chi={chi_h0:+.4f} g={gn:.5f}")
print(f"  -> xi_h0={xi_h0:+.4f} chi_h0={chi_h0:+.4f}")

xi_incs  = [xi_h0]
chi_incs = [chi_h0]
for ell in range(1, L+1):
    print(f"\n=== LEVEL {ell}  K_c={K_levels[ell-1]} K_f={K_levels[ell]} ===")
    Xc, Xf = sim_coupled(Ns[ell], K_levels[ell-1], K_levels[ell], rng)
    xi_c, chi_c = 0.0, 0.0
    xi_f, chi_f = 0.0, 0.0
    for n in range(Ns[ell]):
        xc, xf = Xc[n], Xf[n]
        gn = gamma(n + 1)
        xi_c  -= gn * H1(xi_c, xc, ALPHA)
        chi_c -= 1.0 / (n + 1) * H2(chi_c, xi_c, xc, ALPHA)
        xi_f  -= gn * H1(xi_f, xf, ALPHA)
        chi_f -= 1.0 / (n + 1) * H2(chi_f, xi_f, xf, ALPHA)
        if (n+1) in (1000, 10_000, Ns[ell]):
            print(f"  n={n+1:>6d} xi_c={xi_c:+.4f} xi_f={xi_f:+.4f} "
                  f"chi_c={chi_c:+.4f} chi_f={chi_f:+.4f}")
    xi_incs.append(xi_f - xi_c)
    chi_incs.append(chi_f - chi_c)
    print(f"  -> xi_inc={xi_f-xi_c:+.4f}  chi_inc={chi_f-chi_c:+.4f}")

print("\n=== FINAL ===")
print(f"xi increments:  {[f'{v:+.4f}' for v in xi_incs]}  sum = {sum(xi_incs):+.4f}  (benchmark {xi_star:+.4f})")
print(f"chi increments: {[f'{v:+.4f}' for v in chi_incs]}  sum = {sum(chi_incs):+.4f}  (benchmark {chi_star:+.4f})")
