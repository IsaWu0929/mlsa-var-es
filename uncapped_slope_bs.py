"""
uncapped_slope_bs.py
=====================
Verify that thesis's flat empirical slopes are due to N <= 2e5 cap, not
implementation bug. Run BS-only experiment with N = eps^-2 (uncapped) and
fit log T vs log eps slopes. Compare against CFL's reported -2/-3/-2.7.

Run: python3 uncapped_slope_bs.py
"""
import time, pickle
from pathlib import Path
import numpy as np

from mlsa_core import algo1_SA, algo2_NSA, algo3_MLSA
from case_study_1 import make_simulators as make_bs_simulators, closed_form

OUT_DIR = Path("tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
DELTA = 0.5
ALPHA = 0.975
GAMMA_DENOM = 500
GAMMA_FACTOR_VaR = 0.75
N_REPS = 30  # less than main 200, but enough for slope estimation
EPSILONS = [1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512]


def make_gamma(factor, denom=GAMMA_DENOM):
    return lambda n: factor / (denom + n)


def run_uncapped_bs():
    sim_X0, sim_Xh, sim_coupled = make_bs_simulators(DELTA)
    xi_star, chi_star = closed_form(ALPHA, DELTA)
    print(f"BS benchmark: xi_star = {xi_star:.4f}, chi_star = {chi_star:.4f}")

    results = {}

    for eps in EPSILONS:
        N_uncapped = int(np.ceil(eps ** (-2)))  # NO cap!
        K_inner = int(np.ceil(eps ** (-1)))

        # MLSA recipe (matching adaptive_mlsa.py at eps=1/128 scaled)
        # L levels, geometric sample sizes
        L = max(1, int(np.ceil(np.log2(K_inner / 32))))  # K0 = K/2^L ≈ 32
        h0 = 1.0 / 32  # coarse step
        M = 2
        Ns_mlsa = [N_uncapped // (M ** ell) for ell in range(L + 1)]

        print(f"\n=== eps = 1/{int(round(1 / eps))} ===")
        print(f"   N_uncapped = {N_uncapped:,}, K_inner = {K_inner}")
        print(f"   MLSA: L = {L}, Ns = {Ns_mlsa}, total work = {sum(N * K_inner for N in Ns_mlsa):,}")

        cell_data = {'SA': {'xi': [], 'time': []},
                     'NSA': {'xi': [], 'time': []},
                     'MLSA': {'xi': [], 'time': []}}

        for r in range(N_REPS):
            seed = 99000 + r * 100 + int(round(np.log2(1 / eps)))

            # SA: uncapped
            rng = np.random.default_rng(seed)
            t0 = time.perf_counter()
            xi_sa, chi_sa = algo1_SA(sim_X0, N_uncapped, ALPHA,
                                     make_gamma(GAMMA_FACTOR_VaR), rng=rng)
            t_sa = time.perf_counter() - t0
            cell_data['SA']['xi'].append(xi_sa)
            cell_data['SA']['time'].append(t_sa)

            # NSA: uncapped
            rng = np.random.default_rng(seed + 50000)
            t0 = time.perf_counter()
            xi_nsa, chi_nsa = algo2_NSA(sim_Xh, N_uncapped, K_inner, ALPHA,
                                        make_gamma(GAMMA_FACTOR_VaR), rng=rng)
            t_nsa = time.perf_counter() - t0
            cell_data['NSA']['xi'].append(xi_nsa)
            cell_data['NSA']['time'].append(t_nsa)

            # MLSA: uncapped
            rng = np.random.default_rng(seed + 100000)
            t0 = time.perf_counter()
            xi_ml, chi_ml = algo3_MLSA(sim_coupled, sim_Xh, L, h0, M,
                                       Ns_mlsa, ALPHA,
                                       make_gamma(GAMMA_FACTOR_VaR), rng=rng)
            t_ml = time.perf_counter() - t0
            cell_data['MLSA']['xi'].append(xi_ml)
            cell_data['MLSA']['time'].append(t_ml)

            if (r + 1) % 5 == 0:
                print(f"   rep {r + 1}/{N_REPS}: SA t={t_sa:.2f}s NSA t={t_nsa:.2f}s MLSA t={t_ml:.2f}s")

        results[eps] = {
            'N': N_uncapped,
            'K': K_inner,
            **{alg: {'xi_mean': np.mean(d['xi']),
                     'rmse': np.sqrt(np.mean((np.array(d['xi']) - xi_star) ** 2)),
                     'mean_time': np.mean(d['time']),
                     'all_times': d['time']}
               for alg, d in cell_data.items()}
        }

    # Save raw
    with open(OUT_DIR / 'uncapped_bs.pkl', 'wb') as f:
        pickle.dump({'results': results, 'xi_star': xi_star}, f)

    # Compute slopes
    print("\n" + "=" * 80)
    print("UNCAPPED SLOPE ANALYSIS (BS VaR, N = eps^-2)")
    print("=" * 80)
    print(f"\n{'eps':>10s}  {'N':>10s}  {'K':>6s}  {'SA t':>8s}  {'NSA t':>8s}  {'MLSA t':>8s}")
    for eps in EPSILONS:
        r = results[eps]
        print(f"  1/{int(round(1 / eps)):<8d}  {r['N']:>10,d}  {r['K']:>6d}  "
              f"{r['SA']['mean_time']:>7.2f}s  {r['NSA']['mean_time']:>7.2f}s  {r['MLSA']['mean_time']:>7.2f}s")

    print(f"\n=== Slope estimates: log T vs log eps ===")
    print(f"   {'Algorithm':<10s}  {'Empirical slope':>16s}  {'Theoretical slope':>20s}  {'CFL Table 2':>14s}")
    print(f"   {'-' * 10}  {'-' * 16}  {'-' * 20}  {'-' * 14}")

    log_eps = np.log(EPSILONS)

    cfl_slopes = {'SA': -2.0, 'NSA': -3.0, 'MLSA': -2.7}
    theory = {'SA': -2.0, 'NSA': -3.0, 'MLSA': -2.0 - 0.7}

    for alg in ['SA', 'NSA', 'MLSA']:
        times = np.array([results[eps][alg]['mean_time'] for eps in EPSILONS])
        log_t = np.log(times)
        slope, _ = np.polyfit(log_eps, log_t, 1)
        print(f"   {alg:<10s}  {slope:>15.3f}   {theory[alg]:>19.2f}   {cfl_slopes[alg]:>13.2f}")

    print()
    print("=" * 80)
    print("RMSE-vs-eps for sanity check (should converge to xi_star = 2.0119)")
    print("=" * 80)
    print(f"\n{'eps':>10s}  {'SA RMSE':>10s}  {'NSA RMSE':>10s}  {'MLSA RMSE':>11s}")
    for eps in EPSILONS:
        r = results[eps]
        print(f"  1/{int(round(1 / eps)):<8d}  {r['SA']['rmse']:>10.4f}  "
              f"{r['NSA']['rmse']:>10.4f}  {r['MLSA']['rmse']:>11.4f}")

    print()
    print("=" * 80)
    print("INTERPRETATION:")
    print("  - If empirical slopes are close to CFL Table 2 (-2, -3, -2.7),")
    print("    the cap was responsible for thesis's flat slopes.")
    print("  - If slopes still flat: implementation issue or recipe issue.")
    print("=" * 80)


if __name__ == "__main__":
    run_uncapped_bs()