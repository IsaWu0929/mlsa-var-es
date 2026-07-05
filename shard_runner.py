"""
shard_runner.py
================
Runs a single (model, eps_idx, target) shard, then exits cleanly. Designed
to be called repeatedly from a shell loop -- this gives a fresh Python
process per shard, eliminating any cumulative memory pressure.

USAGE
-----
    python3 shard_runner.py --model bs --eps-idx 0 --target VaR --n-runs 80
"""
import argparse, os, pickle, time, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from parallel_driver import (
    EPSILONS, _bundle_bs, _bundle_heston, _bundle_merton,
    _run_one, _shard_path, _shard_is_complete,
)

BUNDLES = dict(bs=_bundle_bs, heston=_bundle_heston, merton=_bundle_merton)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, choices=["bs", "heston", "merton"])
    ap.add_argument("--eps-idx", type=int, required=True)
    ap.add_argument("--target",  required=True, choices=["VaR", "ES"])
    ap.add_argument("--n-runs",  type=int, default=80)
    args = ap.parse_args()

    eps = EPSILONS[args.eps_idx]
    path = _shard_path(args.model, args.eps_idx, args.target)
    if _shard_is_complete(path, args.n_runs):
        print(f"  SKIP {path.name} (already complete)")
        return

    print(f"  RUN {args.model}/eps_idx={args.eps_idx}/{args.target} "
          f"(n_runs={args.n_runs})")
    bundle = BUNDLES[args.model]()

    results = []
    t0 = time.time()
    for r in range(args.n_runs):
        out = _run_one(bundle, eps, args.target, r)
        results.append(out)
        if r % 20 == 19:
            print(f"    r={r+1:3d}/{args.n_runs}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    payload = dict(
        model=args.model, eps=eps, eps_idx=args.eps_idx, target=args.target,
        alpha=bundle["alpha"],
        xi_star=bundle["xi_star"], chi_star=bundle["chi_star"],
        n_runs=args.n_runs, elapsed=elapsed,
        replications=results,
    )

    # Atomic write
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)
    print(f"  DONE {path.name} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
