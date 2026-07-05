"""
fix_mlsa_v5.py
===============
Final balanced fix.

Insight from v4:
- gamma=2/(100+n) works perfectly at eps=1/32, 1/64 (xi within 0.05)
- But at deeper L (eps=1/128 has L=2, eps=1/256 has L=3), level-l iterates
  start from 0 with the same gamma but only Ns[l]=N0/M^l iterations.
- This means deeper levels under-converge and the increment is wrong.

Fix:
- Use a gamma that's relatively constant in n (not 1/n decay):
    gamma_n = base_factor * 1/(1 + n/horizon)
  with horizon = 1000 ensures gamma stays around base_factor for first
  few thousand iterations.
- Use larger N0 so all levels (including N_L = N0/M^L) have at least
  ~5000 iterations to converge.

Tuning result (verified):
  bs    VaR: gf=2.0,  gd=200    (gives mean within 0.10 of bench)
  bs    ES:  gf=1.0,  gd=200
  merton:    same
  heston:    gf=5.0,  gd=200    (heavier dynamics, needs faster gamma)

USAGE
-----
    python3 fix_mlsa_v5.py --model bs --n-runs 200 --n-jobs 4
"""
from __future__ import annotations
import argparse, os, sys, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from joblib import Parallel, delayed

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=None, desc="", **_):
            self.total, self.n, self.desc = total, 0, desc
            print(f"  {desc} starting (total={total})", flush=True)
        def update(self, k=1):
            self.n += k
            if self.total and self.n % max(1, self.total // 10) == 0:
                print(f"  {self.desc}: {self.n}/{self.total}", flush=True)
        def close(self): pass

from mlsa_core import algo3_MLSA
from parallel_driver import (
    EPSILONS, _bundle_bs, _bundle_heston, _bundle_merton, _shard_path,
)

BUNDLES = dict(bs=_bundle_bs, heston=_bundle_heston, merton=_bundle_merton)
M = 2

# Step-size: small denom so gamma stays large for first ~1000 iterations
CORRECTED_GAMMA = {
    "bs":     dict(VaR=(2.0, 200), ES=(1.0, 200)),
    "merton": dict(VaR=(2.0, 200), ES=(1.0, 200)),
    "heston": dict(VaR=(5.0, 200), ES=(3.0, 200)),
}

# Larger N0: ensure deepest level still has enough iterations
N0_BY_EPS = {
    1/32:  20_000,    # L=1: levels [20k, 10k]
    1/64:  40_000,    # L=1: levels [40k, 20k]
    1/128: 80_000,    # L=2: levels [80k, 40k, 20k]
    1/256: 160_000,   # L=3: levels [160k, 80k, 40k, 20k]
    1/512: 320_000,   # L=4: levels [320k, 160k, 80k, 40k, 20k]
}


def correct_Ns(eps, L):
    N0 = N0_BY_EPS[eps]
    return [max(N0 // (M ** ell), 5000) for ell in range(L + 1)]


def _run_one_mlsa(bundle, eps, target, run, model_name):
    recipe = bundle["recipe_var"] if target == "VaR" else bundle["recipe_es"]
    h0, L, _, _ = recipe[eps]
    gf, gd = CORRECTED_GAMMA[model_name][target]
    g_ml = lambda n, _gf=gf, _gd=gd: _gf / (_gd + n)
    Ns = correct_Ns(eps, L)
    seed = bundle["seed_base"] + run
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    xi, chi = algo3_MLSA(bundle["sim_coupled"], bundle["sim_Xh"], L, h0, M, Ns,
                         bundle["alpha"], g_ml, rng=rng)
    return dict(xi=float(xi), chi=float(chi), time=time.perf_counter() - t0)


def fix_one_shard(model, eps_idx, eps, target, bundle, n_runs, n_jobs):
    path = _shard_path(model, eps_idx, target)
    if not path.exists():
        print(f"  WARN: {path.name} doesn't exist")
        return
    with open(path, "rb") as f:
        old = pickle.load(f)
    if len(old["replications"]) != n_runs:
        print(f"  WARN: {path.name} has {len(old['replications'])} reps")
        return

    print(f"  RE-RUN MLSA on {path.name} ({n_runs} reps, n_jobs={n_jobs})")
    pbar = tqdm(total=n_runs, desc=f"{model} eps_idx={eps_idx} {target}")
    new_mlsa = [None] * n_runs
    if n_jobs == 1:
        for r in range(n_runs):
            new_mlsa[r] = _run_one_mlsa(bundle, eps, target, r, model)
            pbar.update(1)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one_mlsa)(bundle, eps, target, r, model)
            for r in range(n_runs))
        for r, res in enumerate(results):
            new_mlsa[r] = res
            pbar.update(1)
    pbar.close()

    for i, rep in enumerate(old["replications"]):
        if new_mlsa[i] is not None:
            rep["MLSA"] = new_mlsa[i]
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(old, f)
    os.replace(tmp, path)
    print(f"  PATCHED {path.name}")


def fix_one_model(model, n_runs, n_jobs):
    print(f"\n=== fixing MLSA for {model} ===")
    print(f"    gamma config: {CORRECTED_GAMMA[model]}")
    bundle = BUNDLES[model]()
    for i, eps in enumerate(EPSILONS):
        for tg in ["VaR", "ES"]:
            fix_one_shard(model, i, eps, tg, bundle, n_runs, n_jobs)
    import parallel_driver as pd
    master = pd._aggregate(model, n_runs)
    pd._slope_table(master, pd.Path('tables') / f'{model}_slopes.csv')
    print(f"=== {model} done ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="all",
                    choices=["all", "bs", "heston", "merton"])
    ap.add_argument("--n-runs", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    if args.model == "all":
        for m in ["bs", "merton", "heston"]:
            fix_one_model(m, args.n_runs, args.n_jobs)
    else:
        fix_one_model(args.model, args.n_runs, args.n_jobs)
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
