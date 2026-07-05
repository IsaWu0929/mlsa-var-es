"""
equal_rmse_table.py
====================
Compute equal-RMSE speedup: for each target RMSE, interpolate the wall-time
each algorithm needs to achieve that RMSE.  Pure post-processing of shards.

Run: python3 equal_rmse_table.py
"""
import pickle
import numpy as np
from pathlib import Path

EXP_DIR = Path("experiments")


def aggregate_from_shards(model, target):
    """
    Reaggregate RMSE and mean wall-time from individual shards.
    Returns (epsilons, rmse_dict, time_dict) where dicts are keyed by 'SA','NSA','MLSA'.
    """
    epsilons = []
    data = {'SA': {'rmse': [], 'time': []},
            'NSA': {'rmse': [], 'time': []},
            'MLSA': {'rmse': [], 'time': []}}

    for i in range(5):  # eps0 ... eps4
        shard_path = EXP_DIR / f"shards/{model}_eps{i}_{target}.pkl"
        with open(shard_path, "rb") as f:
            d = pickle.load(f)

        eps = d['eps']        # ← FIX: was 'epsilon'
        epsilons.append(eps)

        # Get benchmark
        if target == 'VaR':
            bench = d['xi_star']
        else:
            bench = d['chi_star']

        for alg in ['SA', 'NSA', 'MLSA']:
            estimates = np.array([
                r[alg]['xi'] if target == 'VaR' else r[alg]['chi']
                for r in d['replications']
            ])
            times = np.array([r[alg]['time'] for r in d['replications']])

            rmse = np.sqrt(np.mean((estimates - bench)**2))
            mean_time = times.mean()

            data[alg]['rmse'].append(rmse)
            data[alg]['time'].append(mean_time)

    epsilons = np.array(epsilons)
    for alg in data:
        data[alg]['rmse'] = np.array(data[alg]['rmse'])
        data[alg]['time'] = np.array(data[alg]['time'])

    return epsilons, data


def equal_rmse_time(target_rmse, rmse_grid, time_grid):
    """
    Interpolate (in log-log) the wall-time at which RMSE = target_rmse.
    Returns time at target RMSE, or NaN if outside range.
    """
    log_target = np.log(target_rmse)
    log_rmse = np.log(rmse_grid)
    log_time = np.log(time_grid)

    # Sort by log_rmse so np.interp works (must be monotonic increasing)
    order = np.argsort(log_rmse)
    log_rmse_sorted = log_rmse[order]
    log_time_sorted = log_time[order]

    if log_target < log_rmse_sorted.min() or log_target > log_rmse_sorted.max():
        return np.nan  # out of grid range

    log_t = np.interp(log_target, log_rmse_sorted, log_time_sorted)
    return float(np.exp(log_t))


def equal_rmse_speedup_for_cell(model, target):
    """Compute and print equal-RMSE speedup table for a single cell."""
    eps_grid, data = aggregate_from_shards(model, target)

    rmse_nsa = data['NSA']['rmse']
    time_nsa = data['NSA']['time']
    rmse_mlsa = data['MLSA']['rmse']
    time_mlsa = data['MLSA']['time']

    # Choose target RMSE values within both algorithms' overlapping range
    rmse_lo = max(rmse_nsa.min(), rmse_mlsa.min())
    rmse_hi = min(rmse_nsa.max(), rmse_mlsa.max())

    print(f"\n=== {model.upper()} {target} : Equal-RMSE Speedup ===")
    print(f"   eps grid: {[f'1/{int(round(1/e))}' for e in eps_grid]}")
    print(f"   NSA RMSE values: {rmse_nsa.round(4).tolist()}")
    print(f"   MLSA RMSE values: {rmse_mlsa.round(4).tolist()}")
    print(f"   NSA time values: {time_nsa.round(2).tolist()}")
    print(f"   MLSA time values: {time_mlsa.round(2).tolist()}")
    print(f"   Overlapping RMSE range: [{rmse_lo:.4f}, {rmse_hi:.4f}]")
    print()

    # Equal-eps speedup at finest grid point
    eq_eps_speedup_finest = time_nsa[-1] / time_mlsa[-1]

    if rmse_lo >= rmse_hi:
        print("   No overlapping RMSE range; equal-RMSE comparison not possible.")
        print(f"   Equal-eps speedup at finest grid (eps=1/512): {eq_eps_speedup_finest:.2f}x")
        return

    # 5 RMSE targets log-spaced within overlapping range
    target_rmses = np.geomspace(rmse_lo * 1.05, rmse_hi * 0.95, 5)

    print(f"   {'Target RMSE':>11s}  {'NSA t (s)':>10s}  {'MLSA t (s)':>11s}  "
          f"{'Eq-RMSE x':>10s}  {'Eq-eps x (1/512)':>18s}")
    print(f"   {'-'*11}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*18}")

    for tr in target_rmses:
        t_nsa = equal_rmse_time(tr, rmse_nsa, time_nsa)
        t_mlsa = equal_rmse_time(tr, rmse_mlsa, time_mlsa)

        if not (np.isnan(t_nsa) or np.isnan(t_mlsa)):
            speedup = t_nsa / t_mlsa
            print(f"   {tr:>11.4f}  {t_nsa:>10.2f}  {t_mlsa:>11.2f}  "
                  f"{speedup:>9.2f}x  {eq_eps_speedup_finest:>17.2f}x")
        else:
            print(f"   {tr:>11.4f}  {'(out)':>10s}  {'(out)':>11s}  "
                  f"{'n/a':>10s}  {eq_eps_speedup_finest:>17.2f}x")


def main():
    print("=" * 80)
    print("Equal-RMSE vs Equal-eps Speedup Comparison")
    print("=" * 80)
    print()
    print("Equal-eps speedup (matched prescribed accuracy) = thesis Table 5 numbers.")
    print("Equal-RMSE speedup (matched achieved accuracy) = practitioner-relevant.")

    for model in ['bs', 'heston', 'merton']:
        for target in ['VaR', 'ES']:
            try:
                equal_rmse_speedup_for_cell(model, target)
            except Exception as e:
                print(f"\n=== {model} {target} ===")
                print(f"   ERROR: {e}")
                import traceback
                traceback.print_exc()

    print()
    print("=" * 80)
    print("Interpretation guide:")
    print("  - If 'Eq-RMSE x' < 'Eq-eps x' (e.g., 1.0x vs 1.73x), MLSA's apparent")
    print("    speedup is partly attributable to MLSA having larger RMSE at matched eps.")
    print("  - If 'Eq-RMSE x' >= 1.0, MLSA reaches a target RMSE faster than NSA.")
    print("  - If 'Eq-RMSE x' < 1.0, NSA is actually faster at matched RMSE.")
    print("=" * 80)


if __name__ == "__main__":
    main()