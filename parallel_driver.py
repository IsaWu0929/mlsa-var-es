"""
parallel_driver.py
===================
Stage-5 experiment driver: runs the full {3 models x 5 epsilons x 3 algorithms
x 2 risk measures x 200 replications} matrix using:

    * joblib.Parallel(n_jobs=-1)  -- saturates all CPU cores
    * tqdm                         -- nested progress bars
    * per-(model, eps, target) shard pickles  -- crash-safe granularity

This is *strictly orthogonal* to the existing run_*.py scripts. Those still
work for one-off interactive runs; this driver is the right tool when you
have hours/days to spend and want maximum throughput.

USAGE
-----
    python parallel_driver.py --model bs       --paper
    python parallel_driver.py --model heston   --paper
    python parallel_driver.py --model merton   --paper
    python parallel_driver.py --model all      --paper          # all three sequentially
    python parallel_driver.py --model bs       --quick          # n_runs=20 fast smoke
    python parallel_driver.py --status                          # show progress

    # control parallelism explicitly
    python parallel_driver.py --model heston   --paper --n-jobs 4

OUTPUTS
-------
    experiments/shards/<model>_eps<idx>_<target>.pkl       -- one per (eps, target) pair
    experiments/<model>_master.pkl                          -- aggregated final results
    experiments/shards/_log.txt                              -- timestamped progress log

CRASH RECOVERY
--------------
- If a shard already exists with the right number of replications, it is
  SKIPPED.
- If a shard exists but is partial (fewer replications than requested), it
  is RECOMPUTED.
- If the master .pkl exists, the script just rebuilds plots & tables from it
  unless --force-aggregate is passed.

Re-running the same command always picks up where the last run stopped.
"""
from __future__ import annotations
import argparse, os, time, pickle, sys, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

# tqdm is optional -- fall back to a minimal shim if not installed
try:
    from tqdm.auto import tqdm
except ImportError:                                                # pragma: no cover
    class tqdm:                                                    # noqa: N801
        """Minimal stdout-only fallback so the script works without tqdm."""
        def __init__(self, total=None, desc="", position=0, leave=True, **_):
            self.total, self.n = total, 0
            self.desc = desc
            print(f"  {desc} starting (total={total})", flush=True)
        def update(self, k=1):
            self.n += k
            if self.total and (self.n % max(1, self.total // 20) == 0
                               or self.n == self.total):
                pct = 100.0 * self.n / self.total
                print(f"  {self.desc}: {self.n}/{self.total} ({pct:.0f}%)",
                      flush=True)
        def close(self): pass

# ---- import models and algorithms (relative to this file's dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlsa_core import algo1_SA, algo2_NSA, algo3_MLSA


# =====================================================================
#  Configuration
# =====================================================================
EPSILONS = [1/32, 1/64, 1/128, 1/256, 1/512]
TARGETS  = ["VaR", "ES"]
M = 2

SHARDS_DIR = Path("experiments/shards")
MASTER_DIR = Path("experiments")


# =====================================================================
#  Per-model bundles  (loss simulators + recipes + step sizes)
# =====================================================================
def _bundle_bs():
    """Black-Scholes case study bundle."""
    from case_study_1 import closed_form, make_simulators
    ALPHA, DELTA = 0.975, 0.5
    xi_star, chi_star = closed_form(ALPHA, DELTA)
    sim_X0, sim_Xh, sim_coupled = make_simulators(DELTA)

    # Recipe (h0, L, gamma_factor, gamma_denom)  per epsilon
    RECIPE_VAR = {
        1/32 : (1/16, 1, 2.00,  2500),
        1/64 : (1/32, 1, 2.00,  4000),
        1/128: (1/32, 2, 0.75,  9000),
        1/256: (1/32, 3, 0.25, 10000),
        1/512: (1/32, 4, 0.09, 10000),
    }
    RECIPE_ES = {
        1/32 : (1/16, 1, 0.1, 10000),
        1/64 : (1/32, 1, 0.1, 10000),
        1/128: (1/32, 2, 0.1, 10000),
        1/256: (1/32, 3, 0.1, 20000),
        1/512: (1/32, 4, 0.1, 25000),
    }
    g_sa  = lambda n: 1.0 / (100 + n)
    g_nsa = g_sa

    return dict(
        name="bs",
        alpha=ALPHA,
        xi_star=xi_star, chi_star=chi_star,
        sim_X0=sim_X0, sim_Xh=sim_Xh, sim_coupled=sim_coupled,
        recipe_var=RECIPE_VAR, recipe_es=RECIPE_ES,
        g_sa=g_sa, g_nsa=g_nsa,
        seed_base=2024,
    )


def _bundle_heston():
    from case_study_heston import HestonParams, make_simulators, compute_benchmark
    P = HestonParams()
    sim_X0, sim_Xh, sim_coupled, _ = make_simulators(P, n_steps_to_delta=1, n_steps_post_delta=2)

    print("[heston] computing benchmark (xi*, chi*) ...")
    rng = np.random.default_rng(12345)
    X = sim_X0(20_000, rng)
    xi_star  = float(np.quantile(X, P.alpha))
    chi_star = float(X[X >= xi_star].mean())
    print(f"[heston] xi*={xi_star:.4f}  chi*={chi_star:.4f}")

    RECIPE_VAR = {
        1/32 : (1/8, 1, 5.0, 100),
        1/64 : (1/8, 2, 3.0, 200),
        1/128: (1/8, 3, 2.0, 500),
        1/256: (1/8, 4, 1.0, 1000),
        1/512: (1/8, 5, 0.5, 2000),
    }
    RECIPE_ES = {
        1/32 : (1/8, 1, 3.0, 100),
        1/64 : (1/8, 2, 2.0, 300),
        1/128: (1/8, 3, 1.5, 500),
        1/256: (1/8, 4, 1.0, 1000),
        1/512: (1/8, 5, 0.5, 2000),
    }
    g_sa  = lambda n: 5.0 / (100 + n)
    g_nsa = g_sa

    return dict(
        name="heston",
        alpha=P.alpha,
        xi_star=xi_star, chi_star=chi_star,
        sim_X0=sim_X0, sim_Xh=sim_Xh, sim_coupled=sim_coupled,
        recipe_var=RECIPE_VAR, recipe_es=RECIPE_ES,
        g_sa=g_sa, g_nsa=g_nsa,
        seed_base=50_000,
    )


def _bundle_merton():
    from case_study_merton import MertonParams, make_simulators, compute_benchmark
    P = MertonParams()
    sim_X0, sim_Xh, sim_coupled, _ = make_simulators(P)

    print("[merton] computing benchmark (xi*, chi*) ...")
    xi_star, chi_star = compute_benchmark(P, n_samples=1_000_000)
    print(f"[merton] xi*={xi_star:.4f}  chi*={chi_star:.4f}")

    RECIPE_VAR = {
        1/32 : (1/8, 1, 2.0,  100),
        1/64 : (1/8, 2, 1.0,  200),
        1/128: (1/8, 3, 0.5,  500),
        1/256: (1/8, 4, 0.3, 1000),
        1/512: (1/8, 5, 0.2, 2000),
    }
    RECIPE_ES = {
        1/32 : (1/8, 1, 2.0,  100),
        1/64 : (1/8, 2, 1.0,  200),
        1/128: (1/8, 3, 1.0,  500),
        1/256: (1/8, 4, 0.5, 1000),
        1/512: (1/8, 5, 0.3, 2000),
    }
    g_sa  = lambda n: 1.0 / (100 + n)
    g_nsa = g_sa

    return dict(
        name="merton",
        alpha=P.alpha,
        xi_star=xi_star, chi_star=chi_star,
        sim_X0=sim_X0, sim_Xh=sim_Xh, sim_coupled=sim_coupled,
        recipe_var=RECIPE_VAR, recipe_es=RECIPE_ES,
        g_sa=g_sa, g_nsa=g_nsa,
        seed_base=60_000,
    )


BUNDLES = dict(bs=_bundle_bs, heston=_bundle_heston, merton=_bundle_merton)


# =====================================================================
#  Per-replication kernel (called by joblib workers)
# =====================================================================
def _budget(eps: float, h0: float, L: int, M: int = 2) -> tuple[int, list[int]]:
    """Returns (N_single, Ns_levels) given prescribed eps."""
    h_target = h0 / (M ** L)
    N_single = max(int(np.ceil(eps**(-2.0) / h_target)), 100)
    # Cap aggressively for memory safety. N=20k for a 2-worker container is safe.
    N_single = min(N_single, 200_000)

    base = max(int(np.ceil(eps**(-2.0) * h0)), 200)
    base = min(base, 60_000)                        # tight cap to fit small RAM
    Ns = [int(round(base / (M ** ell))) for ell in range(L + 1)]
    return N_single, Ns


def _run_one(bundle: dict, eps: float, target: str, run: int) -> dict:
    """A single replication for one (model, eps, target). Runs SA, NSA, MLSA."""
    recipe = bundle["recipe_var"] if target == "VaR" else bundle["recipe_es"]
    h0, L, gf, gd = recipe[eps]
    h_target = h0 / (M ** L)
    K_target = int(round(1.0 / h_target))
    N_single, Ns = _budget(eps, h0, L, M)
    g_ml = lambda n, _gf=gf, _gd=gd: _gf / (_gd + n)

    seed = bundle["seed_base"] + run

    out = dict(eps=eps, target=target, run=run)

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    xi, chi = algo1_SA(bundle["sim_X0"], N_single, bundle["alpha"],
                        bundle["g_sa"], rng=rng)
    out["SA"]   = dict(xi=float(xi), chi=float(chi),
                       time=time.perf_counter() - t0)

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    xi, chi = algo2_NSA(bundle["sim_Xh"], N_single, K_target, bundle["alpha"],
                         bundle["g_nsa"], rng=rng)
    out["NSA"]  = dict(xi=float(xi), chi=float(chi),
                       time=time.perf_counter() - t0)

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    xi, chi = algo3_MLSA(bundle["sim_coupled"], bundle["sim_Xh"], L, h0, M, Ns,
                          bundle["alpha"], g_ml, rng=rng)
    out["MLSA"] = dict(xi=float(xi), chi=float(chi),
                       time=time.perf_counter() - t0)

    return out


# =====================================================================
#  Atomic shard I/O
# =====================================================================
def _shard_path(model: str, eps_idx: int, target: str) -> Path:
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    return SHARDS_DIR / f"{model}_eps{eps_idx}_{target}.pkl"


def _atomic_pickle_dump(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _shard_is_complete(path: Path, n_runs_needed: int) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return len(data.get("replications", [])) >= n_runs_needed
    except Exception:
        return False


def _log(msg: str) -> None:
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    with open(SHARDS_DIR / "_log.txt", "a") as f:
        f.write(line)
    print(line, end="")


# =====================================================================
#  One shard:  parallel over the 200 replications
# =====================================================================
def _run_shard(bundle: dict,
               eps_idx: int,
               eps: float,
               target: str,
               n_runs: int,
               n_jobs: int,
               position: int = 0) -> dict:
    """Compute one shard with joblib parallelism + tqdm bar."""
    name = bundle["name"]
    path = _shard_path(name, eps_idx, target)

    if _shard_is_complete(path, n_runs):
        _log(f"SKIP {name}/eps={eps:.5f}/{target}  (shard already complete)")
        with open(path, "rb") as f:
            return pickle.load(f)

    _log(f"RUN  {name}/eps={eps:.5f}/{target}  ({n_runs} reps, n_jobs={n_jobs})")

    t0 = time.perf_counter()
    # joblib with verbose=0; we add our own tqdm bar
    pbar = tqdm(total=n_runs,
                desc=f"{name} eps=1/{int(round(1/eps))} {target}",
                position=position, leave=False)

    # Wrap the kernel in a callback that updates the bar.  joblib runs
    # tasks out of order, but we accept that -- the bar still increments.
    def _wrapped(run_idx):
        out = _run_one(bundle, eps, target, run_idx)
        return out

    # Use a generator pattern so tqdm updates as tasks finish:
    # NB: prefer="threads" avoids the loky-fork memory blow-up for our
    # NumPy-heavy workload. We lose a bit of GIL parallelism but gain
    # huge stability on memory-constrained machines.
    # NB2: For n_jobs=1, skip joblib entirely -- joblib's generator
    # pattern can hold references to all results in memory, causing
    # accumulating memory pressure on small machines. A plain loop
    # is more memory-friendly.
    results = []
    if n_jobs == 1:
        import gc
        for r in range(n_runs):
            res = _wrapped(r)
            results.append(res)
            pbar.update(1)
            if r % 10 == 9:
                gc.collect()
    else:
        with Parallel(n_jobs=n_jobs, prefer="threads",
                      return_as="generator_unordered") as par:
            for res in par(delayed(_wrapped)(r) for r in range(n_runs)):
                results.append(res)
                pbar.update(1)
    pbar.close()

    elapsed = time.perf_counter() - t0
    payload = dict(
        model=name, eps=eps, eps_idx=eps_idx, target=target,
        alpha=bundle["alpha"],
        xi_star=bundle["xi_star"], chi_star=bundle["chi_star"],
        n_runs=n_runs, elapsed=elapsed,
        replications=results,
    )
    _atomic_pickle_dump(payload, path)
    _log(f"DONE {name}/eps={eps:.5f}/{target}  in {elapsed:.1f}s -> {path.name}")
    return payload


# =====================================================================
#  Aggregation: shards -> master pkl + slope tables
# =====================================================================
def _aggregate(model: str, n_runs: int) -> dict:
    """Walk through all shards for this model, build the master payload."""
    bundle_fn = BUNDLES[model]
    # don't actually need to build the whole bundle here, but we need xi*/chi*
    # which we already saved into each shard:
    shards = []
    for i, eps in enumerate(EPSILONS):
        for tg in TARGETS:
            p = _shard_path(model, i, tg)
            if not p.exists():
                _log(f"WARN  missing shard {p.name}; aggregation incomplete")
                continue
            with open(p, "rb") as f:
                shards.append(pickle.load(f))

    # results layout consistent with run_case1.py / run_heston.py
    results = {tg: {al: {"rmse": [], "time": []}
                    for al in ["SA", "NSA", "MLSA"]}
               for tg in TARGETS}
    for i, eps in enumerate(EPSILONS):
        for tg in TARGETS:
            shard = next((s for s in shards
                          if s["eps_idx"] == i and s["target"] == tg), None)
            if shard is None:
                for alg in ["SA", "NSA", "MLSA"]:
                    results[tg][alg]["rmse"].append(float("nan"))
                    results[tg][alg]["time"].append(float("nan"))
                continue
            bench = shard["xi_star"] if tg == "VaR" else shard["chi_star"]
            for alg in ["SA", "NSA", "MLSA"]:
                vals  = [r[alg]["xi" if tg == "VaR" else "chi"]
                         for r in shard["replications"]]
                times = [r[alg]["time"] for r in shard["replications"]]
                rmse = float(np.sqrt(np.mean((np.array(vals) - bench)**2)))
                results[tg][alg]["rmse"].append(rmse)
                results[tg][alg]["time"].append(float(np.mean(times)))

    master = dict(
        model=model, n_runs=n_runs,
        epsilons=EPSILONS, results=results,
        xi_star=shards[0]["xi_star"]  if shards else None,
        chi_star=shards[0]["chi_star"] if shards else None,
    )
    master_path = MASTER_DIR / f"{model}_master.pkl"
    _atomic_pickle_dump(master, master_path)
    _log(f"AGG  {model} master saved -> {master_path}")
    return master


def _slope_table(master: dict, out_csv: Path) -> None:
    import csv
    rows = [("algorithm", "VaR slope (RMSE)", "VaR slope (eps)",
                          "ES slope (RMSE)",  "ES slope (eps)")]
    res = master["results"]
    for alg in ["NSA", "MLSA", "SA"]:
        r_var = np.log(res["VaR"][alg]["rmse"])
        r_es  = np.log(res["ES" ][alg]["rmse"])
        t_var = np.log(res["VaR"][alg]["time"])
        t_es  = np.log(res["ES" ][alg]["time"])
        ln_eps = np.log(EPSILONS)
        s1, _ = np.polyfit(r_var,  t_var, 1)
        s2, _ = np.polyfit(ln_eps, t_var, 1)
        s3, _ = np.polyfit(r_es,   t_es,  1)
        s4, _ = np.polyfit(ln_eps, t_es,  1)
        rows.append((alg, f"{s1:+.2f}", f"{s2:+.2f}",
                          f"{s3:+.2f}", f"{s4:+.2f}"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    _log(f"AGG  {master['model']} slope table -> {out_csv}")


# =====================================================================
#  Top-level driver
# =====================================================================
def run_one_model(model: str, n_runs: int, n_jobs: int) -> None:
    bundle = BUNDLES[model]()        # builds simulators + benchmark

    # outer progress bar over the 10 (eps, target) shards
    outer = tqdm(total=len(EPSILONS) * len(TARGETS),
                 desc=f"shards [{model}]", position=0)
    for i, eps in enumerate(EPSILONS):
        for tg in TARGETS:
            try:
                _run_shard(bundle, i, eps, tg, n_runs, n_jobs, position=1)
            except Exception:
                _log(f"ERROR in {model}/eps={eps:.5f}/{tg}: "
                     f"{traceback.format_exc()}")
            outer.update(1)
    outer.close()

    master = _aggregate(model, n_runs)
    _slope_table(master, Path("tables") / f"{model}_slopes.csv")


def print_status() -> None:
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Shard inventory ===")
    for model in ["bs", "heston", "merton"]:
        n = 0
        for i, eps in enumerate(EPSILONS):
            for tg in TARGETS:
                p = _shard_path(model, i, tg)
                if p.exists():
                    try:
                        with open(p, "rb") as f:
                            data = pickle.load(f)
                        n += 1
                        print(f"  [done] {p.name:<40s} "
                              f"({len(data['replications'])} reps, "
                              f"{data.get('elapsed', 0.0):.1f}s)")
                    except Exception:
                        print(f"  [BAD ] {p.name}")
        print(f"  {model}: {n}/{len(EPSILONS)*len(TARGETS)} shards complete\n")

    print("=== Master files ===")
    for model in ["bs", "heston", "merton"]:
        p = MASTER_DIR / f"{model}_master.pkl"
        if p.exists():
            print(f"  [done] {p.name}")
        else:
            print(f"  [    ] {p.name}")


# =====================================================================
#  CLI
# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["bs", "heston", "merton", "all"],
                    help="which model to run")
    ap.add_argument("--quick",  action="store_true", help="n_runs=20 (smoke test)")
    ap.add_argument("--paper",  action="store_true", help="n_runs=200 (paper)")
    ap.add_argument("--n-runs", type=int, default=None,
                    help="override n_runs explicitly (takes precedence over --quick/--paper)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="joblib n_jobs (default -1 = all cores)")
    ap.add_argument("--status", action="store_true",
                    help="show progress and exit")
    args = ap.parse_args()

    if args.status:
        print_status(); sys.exit(0)

    if args.model is None:
        ap.error("specify --model (bs / heston / merton / all)")

    if args.n_runs is not None:
        n_runs = args.n_runs
    elif args.quick:
        n_runs = 20
    elif args.paper:
        n_runs = 200
    else:
        n_runs = 50         # sensible default

    print(f"=== Stage-5 driver ===")
    print(f"   n_runs = {n_runs}, n_jobs = {args.n_jobs}")
    print(f"   models = {args.model}")
    print(f"   total shards = "
          f"{(3 if args.model=='all' else 1) * len(EPSILONS) * len(TARGETS)}\n")

    if args.model == "all":
        for m in ["bs", "merton", "heston"]:        # cheapest first
            run_one_model(m, n_runs, args.n_jobs)
    else:
        run_one_model(args.model, n_runs, args.n_jobs)

    print("\n=== Done.  See experiments/ for shards and master pickles. ===")
    print_status()
