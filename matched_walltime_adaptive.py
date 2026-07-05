"""
matched_walltime_adaptive.py
=============================
Run standard MLSA with N0 inflated 14x to match Adaptive-MLSA's wall-time budget.

The adaptive method's real overhead (per existing adaptive_mlsa.py config) is
~14x of standard MLSA wall-time, dominated by the per-level pilot work
(K=4 candidates × m=4 reps × n_pilot=2000 × (L+1)=4 levels).

This script reruns Standard MLSA at N_l = 14 * Ns_baseline to give it the same
total wall-time, then compares RMSE against the existing adaptive results.

Run: python3 matched_walltime_adaptive.py
"""
import time, csv, pickle
from pathlib import Path
import numpy as np

from mlsa_core import algo3_MLSA

OUT_DIR = Path("tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inflation factor from analysis above
INFLATION = 14

# Same as adaptive_mlsa.py at eps=1/128
H0 = 1/8
L = 3
M = 2
GAMMA_DENOM = 500
NS_BASELINE = [4_000, 2_000, 1_000, 500]
NS_INFLATED = [n * INFLATION for n in NS_BASELINE]
N_REPS = 80

# Standard recipe (same as adaptive_mlsa.py)
STANDARD_GAMMA = {
    ("bs",     "VaR"): 0.75,  ("bs",     "ES"): 0.1,
    ("heston", "VaR"): 2.0,   ("heston", "ES"): 1.5,
    ("merton", "VaR"): 0.5,   ("merton", "ES"): 1.0,
}


def get_bundle(model):
    """Same as adaptive_mlsa.py _bundle function."""
    if model == "heston":
        from case_study_heston import HestonParams, make_simulators, compute_benchmark
        P = HestonParams()
        _, sim_Xh, sim_coupled, _ = make_simulators(P, n_steps_to_delta=1, n_steps_post_delta=2)
        xi_b, chi_b = compute_benchmark(P, n_samples=20_000)
        return dict(name="heston", alpha=P.alpha,
                    sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                    xi_star=xi_b, chi_star=chi_b)
    if model == "merton":
        from case_study_merton import MertonParams, make_simulators, compute_benchmark
        P = MertonParams()
        _, sim_Xh, sim_coupled, _ = make_simulators(P)
        xi_b, chi_b = compute_benchmark(P, n_samples=300_000)
        return dict(name="merton", alpha=P.alpha,
                    sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                    xi_star=xi_b, chi_star=chi_b)
    if model == "bs":
        from case_study_1 import closed_form, make_simulators
        _, sim_Xh, sim_coupled = make_simulators(0.5)
        xi_b, chi_b = closed_form(0.975, 0.5)
        return dict(name="bs", alpha=0.975,
                    sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                    xi_star=xi_b, chi_star=chi_b)
    raise ValueError(model)


def run_inflated_standard(model, target):
    """Run standard MLSA with N_l * 14 iterations."""
    bundle = get_bundle(model)
    bench = bundle["xi_star"] if target == "VaR" else bundle["chi_star"]
    gamma_factor = STANDARD_GAMMA[(model, target)]

    print(f"\n--- {model.upper()} {target} (inflated standard MLSA) ---")
    print(f"   Ns_inflated = {NS_INFLATED}")
    print(f"   gamma_factor = {gamma_factor}, denom = {GAMMA_DENOM}")
    print(f"   benchmark = {bench:.4f}")

    estimates = []
    times = []

    for run in range(N_REPS):
        rng = np.random.default_rng(7 + run)  # same seed pattern as adaptive_mlsa.py
        gamma = lambda n: gamma_factor / (GAMMA_DENOM + n)

        t0 = time.perf_counter()
        xi, chi = algo3_MLSA(
            simulate_coupled_pair=bundle["sim_coupled"],
            simulate_Xh0=bundle["sim_Xh"],
            L=L, h0=H0, M=M,
            Ns=NS_INFLATED,
            alpha=bundle["alpha"],
            gamma=gamma, rng=rng,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        estimates.append(xi if target == "VaR" else chi)

        if (run + 1) % 20 == 0:
            print(f"   rep {run+1}/{N_REPS}, mean_time so far = {np.mean(times):.3f}s")

    estimates = np.array(estimates)
    rmse = float(np.sqrt(np.mean((estimates - bench)**2)))
    mean_time = float(np.mean(times))

    print(f"   RESULT: RMSE = {rmse:.4f}, mean_time = {mean_time:.3f}s")
    return {'rmse': rmse, 'mean_time': mean_time, 'estimates': estimates.tolist(), 'bench': bench}


def main():
    print("=" * 80)
    print("Matched Wall-Time: Standard MLSA at N_l * 14 vs existing Adaptive-MLSA")
    print("=" * 80)
    print(f"Inflation factor: {INFLATION}x (matches measured Adaptive wall-time overhead)")
    print(f"Replications per cell: {N_REPS}")

    cells = [
        ("bs", "VaR"), ("bs", "ES"),
        ("heston", "VaR"), ("heston", "ES"),
        ("merton", "VaR"), ("merton", "ES"),
    ]

    results = {}
    overall_t0 = time.time()

    for model, target in cells:
        cell_t0 = time.time()
        try:
            res = run_inflated_standard(model, target)
            results[(model, target)] = res
            print(f"   Cell elapsed: {time.time() - cell_t0:.0f}s")
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[(model, target)] = None

    print(f"\nTotal elapsed: {time.time() - overall_t0:.0f}s")

    # Save full results
    with open(OUT_DIR / 'matched_walltime_inflated.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save CSV summary
    with open(OUT_DIR / 'matched_walltime_inflated.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'target', 'inflated_RMSE', 'inflated_mean_time'])
        for (m, t), r in results.items():
            if r is None:
                w.writerow([m, t, 'ERROR', 'ERROR'])
            else:
                w.writerow([m, t, f"{r['rmse']:.4f}", f"{r['mean_time']:.3f}"])

    # Print comparison table
    print()
    print("=" * 80)
    print("MATCHED-WALL-TIME COMPARISON")
    print("=" * 80)
    print()
    print("Standard MLSA at N_l * 14 vs existing Adaptive-MLSA at standard N_l")
    print("(both have approximately equal total wall-time)")
    print()

    # Load existing adaptive results for comparison
    adaptive_csv_template = "tables/adaptive_{m}_{t}.csv"

    print(f"   {'Cell':<13s} | {'Std@N₀ RMSE':>12s} | {'Std@14N₀ RMSE':>14s} | {'Adaptive RMSE':>14s} | {'Verdict'}")
    print("   " + "-" * 88)

    for (m, t) in cells:
        # Read existing adaptive csv
        adp_path = adaptive_csv_template.format(m=m, t=t)
        try:
            with open(adp_path) as f:
                lines = f.readlines()
            std_rmse_baseline = float(lines[1].split(',')[1])
            adp_rmse = float(lines[2].split(',')[1])
        except Exception:
            std_rmse_baseline = float('nan')
            adp_rmse = float('nan')

        std_inflated_rmse = results.get((m, t), {}).get('rmse', float('nan'))

        # Verdict
        if not np.isnan(std_inflated_rmse) and not np.isnan(adp_rmse):
            if std_inflated_rmse < adp_rmse:
                verdict = f"Std@14N₀ wins ({adp_rmse/std_inflated_rmse:.2f}× lower RMSE)"
            else:
                verdict = f"Adaptive wins ({std_inflated_rmse/adp_rmse:.2f}× lower RMSE)"
        else:
            verdict = "n/a"

        cell_label = f"{m} {t}"
        print(f"   {cell_label:<13s} | {std_rmse_baseline:>11.4f} | {std_inflated_rmse:>13.4f} | {adp_rmse:>13.4f} | {verdict}")

    print()
    print("=" * 80)
    print("Interpretation:")
    print("  - 'Std@N₀ RMSE': baseline standard MLSA at original N₀ (matched-N₀ baseline)")
    print("  - 'Std@14N₀ RMSE': standard MLSA with 14× the iterations (matched-wall-time)")
    print("  - 'Adaptive RMSE': adaptive variant at standard N₀")
    print("  - If Std@14N₀ < Adaptive → standard with bigger budget beats adaptive")
    print("  - If Adaptive < Std@14N₀ → adaptive wins even at fair wall-time comparison")
    print("=" * 80)


if __name__ == "__main__":
    main()