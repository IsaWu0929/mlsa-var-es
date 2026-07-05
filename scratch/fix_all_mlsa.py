"""
fix_all_mlsa.py
================
Comprehensive MLSA fix for all three models (BS, Merton, Heston).
Re-runs ONLY the MLSA algorithm with corrected step-size recipes;
preserves SA and NSA shards untouched.

USAGE
-----
    python3 fix_all_mlsa.py
    python3 fix_all_mlsa.py --n-jobs 4
    python3 fix_all_mlsa.py --model bs        # one model only
"""
from __future__ import annotations
import argparse, os, sys, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from joblib import Parallel, delayed

from mlsa_core import algo3_MLSA
from parallel_driver import EPSILONS, _shard_path

M = 2

# -------------------------------------------------------------------
#  Per-model recipes
#  Format:  { eps : (h0, L, gamma_factor, gamma_denom) }
# -------------------------------------------------------------------

# All three models share the same denominator scale (100-2000) which
# empirically works -- the difference is in gamma_factor.
# For ES, gamma_factor is reduced ~5-10x because ES update is
# more sensitive (per sensitivity_study results: ES optimal gamma is
# 0.1-0.3 across all models, while VaR optimum is 0.5-2.0).

BS_RECIPE_VAR = {
    1/32 : (1/8, 1, 2.0,  100),
    1/64 : (1/8, 2, 1.0,  200),
    1/128: (1/8, 3, 0.5,  500),
    1/256: (1/8, 4, 0.3, 1000),
    1/512: (1/8, 5, 0.2, 2000),
}
BS_RECIPE_ES = {
    1/32 : (1/8, 1, 0.3,  100),     # empirically best from grid search
    1/64 : (1/8, 2, 0.2,  200),
    1/128: (1/8, 3, 0.15, 500),
    1/256: (1/8, 4, 0.1, 1000),
    1/512: (1/8, 5, 0.08, 2000),
}

MERTON_RECIPE_VAR = {       # already perfect, keep as-is for completeness
    1/32 : (1/8, 1, 2.0,  100),
    1/64 : (1/8, 2, 1.0,  200),
    1/128: (1/8, 3, 0.5,  500),
    1/256: (1/8, 4, 0.3, 1000),
    1/512: (1/8, 5, 0.2, 2000),
}
MERTON_RECIPE_ES = {
    1/32 : (1/8, 1, 0.3,  100),
    1/64 : (1/8, 2, 0.2,  200),
    1/128: (1/8, 3, 0.15, 500),
    1/256: (1/8, 4, 0.1, 1000),
    1/512: (1/8, 5, 0.08, 2000),
}

HESTON_RECIPE_VAR = {       # already good, keep as-is
    1/32 : (1/8, 1, 2.0,  100),
    1/64 : (1/8, 2, 1.0,  200),
    1/128: (1/8, 3, 0.5,  500),
    1/256: (1/8, 4, 0.3, 1000),
    1/512: (1/8, 5, 0.2, 2000),
}
HESTON_RECIPE_ES = {
    1/32 : (1/8, 1, 0.3,  100),
    1/64 : (1/8, 2, 0.2,  200),
    1/128: (1/8, 3, 0.15, 500),
    1/256: (1/8, 4, 0.1, 1000),
    1/512: (1/8, 5, 0.08, 2000),
}


# -------------------------------------------------------------------
#  Bundles (lazily imported because Heston is slow to set up)
# -------------------------------------------------------------------
def _get_bundle(model: str):
    if model == "bs":
        from case_study_1 import make_simulators, closed_form
        sim_X0, sim_Xh, sim_coupled = make_simulators(0.5)
        xi_star, chi_star = closed_form(0.975, 0.5)
        return dict(
            sim_Xh=sim_Xh, sim_coupled=sim_coupled,
            alpha=0.975,
            xi_star=xi_star, chi_star=chi_star,
            recipe_var=BS_RECIPE_VAR, recipe_es=BS_RECIPE_ES,
        )
    if model == "merton":
        from case_study_merton import MertonParams, make_simulators, compute_benchmark
        P = MertonParams()
        sim_X0, sim_Xh, sim_coupled, _ = make_simulators(P)
        xi_star, chi_star = compute_benchmark(P, n_samples=1_000_000)
        return dict(
            sim_Xh=sim_Xh, sim_coupled=sim_coupled,
            alpha=P.alpha,
            xi_star=xi_star, chi_star=chi_star,
            recipe_var=MERTON_RECIPE_VAR, recipe_es=MERTON_RECIPE_ES,
        )
    if model == "heston":
        from case_study_heston import HestonParams, make_simulators
        P = HestonParams()
        sim_X0, sim_Xh, sim_coupled, _ = make_simulators(
            P, n_steps_to_delta=1, n_steps_post_delta=2
        )
        # Quick benchmark via SA proxy
        rng = np.random.default_rng(12345)
        X = sim_X0(20_000, rng)
        xi_star  = float(np.quantile(X, P.alpha))
        chi_star = float(X[X >= xi_star].mean())
        return dict(
            sim_Xh=sim_Xh, sim_coupled=sim_coupled,
            alpha=P.alpha,
            xi_star=xi_star, chi_star=chi_star,
            recipe_var=HESTON_RECIPE_VAR, recipe_es=HESTON_RECIPE_ES,
        )
    raise ValueError(model)


# -------------------------------------------------------------------
#  Budget calculation
# -------------------------------------------------------------------
def _Ns(eps: float, h0: float, L: int, gamma_denom: int) -> list[int]:
    natural = max(int(np.ceil(eps ** (-2.0) * h0)), 200)
    floor = 8 * gamma_denom
    base = max(natural, floor)
    base = min(base, 500_000)
    return [int(round(base / (M ** ell))) for ell in range(L + 1)]


# -------------------------------------------------------------------
#  Per-replication kernel (must be top-level for pickling)
# -------------------------------------------------------------------
def _run_one_mlsa(model: str, eps: float, target: str, run: int) -> dict:
    """One MLSA replication. Builds bundle locally to avoid pickling sims."""
    bundle = _get_bundle(model)
    recipe = bundle["recipe_var"] if target == "VaR" else bundle["recipe_es"]
    h0, L, gf, gd = recipe[eps]
    g = lambda n, _gf=gf, _gd=gd: _gf / (_gd + n)
    Ns = _Ns(eps, h0, L, gd)
    seed = (2024 + run) if model == "bs" else (
           7 + run if model == "merton" else 50_000 + run)

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    xi, chi = algo3_MLSA(bundle["sim_coupled"], bundle["sim_Xh"],
                          L, h0, M, Ns, bundle["alpha"], g, rng=rng)
    return dict(xi=float(xi), chi=float(chi),
                time=time.perf_counter() - t0)


# -------------------------------------------------------------------
#  Per-shard fix
# -------------------------------------------------------------------
def _fix_shard(model: str, eps_idx: int, target: str, n_jobs: int,
                bundle: dict) -> None:
    eps = EPSILONS[eps_idx]
    path = _shard_path(model, eps_idx, target)
    if not path.exists():
        print(f"  WARN: {path.name} doesn't exist, skipping")
        return

    with open(path, "rb") as f:
        old = pickle.load(f)
    n_runs = len(old["replications"])

    print(f"  RE-RUN MLSA on {model} eps_idx={eps_idx} {target}  "
          f"({n_runs} reps)", flush=True)
    t0 = time.perf_counter()

    if n_jobs == 1:
        new_mlsa = [_run_one_mlsa(model, eps, target, r)
                     for r in range(n_runs)]
    else:
        new_mlsa = Parallel(n_jobs=n_jobs)(
            delayed(_run_one_mlsa)(model, eps, target, r)
            for r in range(n_runs)
        )

    elapsed = time.perf_counter() - t0
    for i, rep in enumerate(old["replications"]):
        rep["MLSA"] = new_mlsa[i]

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(old, f)
    os.replace(tmp, path)

    if target == "VaR":
        vals  = np.array([m["xi"]  for m in new_mlsa])
        bench = bundle["xi_star"]
    else:
        vals  = np.array([m["chi"] for m in new_mlsa])
        bench = bundle["chi_star"]
    rmse = np.sqrt(np.mean((vals - bench) ** 2))
    print(f"    -> mean={vals.mean():+.4f} (bench {bench:.4f}) "
          f"std={vals.std():.4f} RMSE={rmse:.4f}  ({elapsed:.1f}s)",
          flush=True)


# -------------------------------------------------------------------
#  Top-level driver
# -------------------------------------------------------------------
def fix_one_model(model: str, n_jobs: int) -> None:
    print(f"\n=== Fixing MLSA for {model} ===", flush=True)
    bundle = _get_bundle(model)
    print(f"   xi_star  = {bundle['xi_star']:.4f}")
    print(f"   chi_star = {bundle['chi_star']:.4f}\n")

    for eps_idx in range(len(EPSILONS)):
        for target in ["VaR", "ES"]:
            _fix_shard(model, eps_idx, target, n_jobs, bundle)

    # Re-aggregate and write slope table
    print(f"\n  Re-aggregating {model} master + slopes ...", flush=True)
    import parallel_driver as pd
    master = pd._aggregate(model, n_runs=200)
    pd._slope_table(master, pd.Path('tables') / f'{model}_slopes.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="all",
                     choices=["all", "bs", "merton", "heston"])
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    if args.model == "all":
        for m in ["bs", "merton", "heston"]:
            fix_one_model(m, args.n_jobs)
    else:
        fix_one_model(args.model, args.n_jobs)

    print("\n=== ALL DONE ===")
    print("Check the slope tables: ")
    print("  cat tables/bs_slopes.csv")
    print("  cat tables/merton_slopes.csv")
    print("  cat tables/heston_slopes.csv")


if __name__ == "__main__":
    main()
