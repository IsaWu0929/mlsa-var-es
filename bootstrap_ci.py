"""
bootstrap_ci.py
================
Compute 95% bootstrap confidence intervals for key thesis numbers:
- Speedup (NSA / MLSA wall-time) at eps=1/512
- RMSE values (point estimate + CI)

Pure post-processing, no new experiments.

Run: python3 bootstrap_ci.py
"""
import pickle
import numpy as np
from pathlib import Path

EXP_DIR = Path("experiments")
N_BOOT = 10_000


def bootstrap_speedup(model, target, eps_idx=4, n_boot=N_BOOT, seed=42):
    """
    95% bootstrap CI for NSA/MLSA wall-time speedup at given eps_idx.
    eps_idx=4 → eps=1/512.
    """
    rng = np.random.default_rng(seed)

    shard_path = EXP_DIR / f"shards/{model}_eps{eps_idx}_{target}.pkl"
    with open(shard_path, "rb") as f:
        d = pickle.load(f)

    nsa_times = np.array([r['NSA']['time'] for r in d['replications']])
    mlsa_times = np.array([r['MLSA']['time'] for r in d['replications']])
    n = len(nsa_times)

    # Bootstrap
    boot_speedups = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_speedups.append(nsa_times[idx].mean() / mlsa_times[idx].mean())

    point_estimate = nsa_times.mean() / mlsa_times.mean()
    ci_lo, ci_hi = np.percentile(boot_speedups, [2.5, 97.5])

    return point_estimate, ci_lo, ci_hi


def bootstrap_rmse(model, target, eps_idx=4, alg='MLSA', n_boot=N_BOOT, seed=42):
    """
    95% bootstrap CI for RMSE at given.
    """
    rng = np.random.default_rng(seed)

    shard_path = EXP_DIR / f"shards/{model}_eps{eps_idx}_{target}.pkl"
    with open(shard_path, "rb") as f:
        d = pickle.load(f)

    if target == 'VaR':
        bench = d['xi_star']
        estimates = np.array([r[alg]['xi'] for r in d['replications']])
    else:
        bench = d['chi_star']
        estimates = np.array([r[alg]['chi'] for r in d['replications']])

    n = len(estimates)

    # Bootstrap RMSE
    boot_rmses = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rmses.append(np.sqrt(np.mean((estimates[idx] - bench) ** 2)))

    point_estimate = np.sqrt(np.mean((estimates - bench) ** 2))
    ci_lo, ci_hi = np.percentile(boot_rmses, [2.5, 97.5])

    return point_estimate, ci_lo, ci_hi


def main():
    print("=" * 80)
    print("Bootstrap 95% CIs for Key Thesis Numbers")
    print(f"Bootstrap iterations: {N_BOOT}")
    print("=" * 80)

    # === Speedups at eps=1/512 ===
    print("\n=== Wall-time Speedup (NSA / MLSA) at eps=1/512 ===")
    print(f"   {'Cell':<13s}  {'Point':>8s}  {'95% CI (low)':>13s}  {'95% CI (high)':>14s}")

    for model in ['bs', 'heston', 'merton']:
        for target in ['VaR', 'ES']:
            point, lo, hi = bootstrap_speedup(model, target)
            print(f"   {model:>6s} {target:>3s}  {point:>7.2f}x  {lo:>12.2f}x  {hi:>13.2f}x")

    # === RMSEs at eps=1/512 ===
    print("\n=== RMSE at eps=1/512 ===")
    print(f"   {'Cell':<13s}  {'Algorithm':<10s}  {'Point':>8s}  {'95% CI (low)':>13s}  {'95% CI (high)':>14s}")

    for model in ['bs', 'heston', 'merton']:
        for target in ['VaR', 'ES']:
            for alg in ['SA', 'NSA', 'MLSA']:
                point, lo, hi = bootstrap_rmse(model, target, alg=alg)
                print(f"   {model:>6s} {target:>3s}  {alg:<10s}  {point:>7.4f}   {lo:>12.4f}   {hi:>13.4f}")
            print()  # blank line between models/targets

    # === Adaptive RMSE comparison ===
    print("\n=== Adaptive vs Standard MLSA RMSE (eps=1/128) ===")
    print(f"   {'Cell':<13s}  {'Method':<10s}  {'Point':>8s}  {'95% CI (low)':>13s}  {'95% CI (high)':>14s}")

    adaptive_files = {
        'bs_VaR': 'tables/adaptive_bs_VaR.csv',
        'bs_ES': 'tables/adaptive_bs_ES.csv',
        'heston_VaR': 'tables/adaptive_heston_VaR.csv',
        'heston_ES': 'tables/adaptive_heston_ES.csv',
        'merton_VaR': 'tables/adaptive_merton_VaR.csv',
        'merton_ES': 'tables/adaptive_merton_ES.csv',
    }

    for cell_name, csv_path in adaptive_files.items():
        try:
            # Adaptive CSV: variant, RMSE, mean_time_s, n_runs
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            # parse simple csv
            header = lines[0].strip().split(',')
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    variant = parts[0]
                    rmse = float(parts[1])
                    print(f"   {cell_name:<13s}  {variant:<10s}  {rmse:>7.4f}   "
                          f"{'(see note)':>12s}   {'(see note)':>13s}")
        except FileNotFoundError:
            print(f"   {cell_name:<13s}  (CSV not found at {csv_path})")

    print("\n   Note: Adaptive CIs require per-replication data not stored in CSV.")
    print("         If you have full replication data, modify this script accordingly.")

    print()
    print("=" * 80)
    print("Suggested thesis text:")
    print("  Replace 'MLSA delivers 1.73x speedup' with")
    print("          'MLSA delivers 1.73x speedup [95% CI: ...]'")
    print("=" * 80)


if __name__ == "__main__":
    main()