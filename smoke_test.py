"""
smoke_test.py  (with checkpointing)
=====================================
60-second end-to-end verification that everything works.

Runs Algorithms 1, 2, 3 once each on Case Study 1 and prints the
results next to the closed-form benchmark. Also exercises:
- The `progress` callback on every algorithm (you should see periodic
  "iter ... xi=... chi=..." heartbeats);
- The checkpoint mechanism (the run is split into 3 sub-tasks; a Ctrl-C
  midway and a re-run will resume from the last completed sub-task).

USAGE
-----
    python smoke_test.py             # run / resume
    python smoke_test.py --restart   # ignore prior checkpoint
    python smoke_test.py --status    # show what is done
"""
from __future__ import annotations
import argparse, time
import numpy as np

from mlsa_core    import algo1_SA, algo2_NSA, algo3_MLSA
from case_study_1 import closed_form, make_simulators
import checkpoint as ckpt_mod


CKPT_NAME = "smoke"

ALPHA, DELTA = 0.975, 0.5
XI_STAR, CHI_STAR = closed_form(ALPHA, DELTA)
sim_X0, sim_Xh, sim_coupled = make_simulators(DELTA)


def step_algo1(ckpt):
    if ckpt_mod.is_done(ckpt, "algo1"):
        rec = ckpt_mod.get(ckpt, "algo1")
        print(f"Algo 1:   xi  = {rec['xi']:.4f}    chi  = {rec['chi']:.4f}   (cached)")
        return
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    xi, chi = algo1_SA(sim_X0, N=200_000, alpha=ALPHA,
                       gamma=lambda n: 1.0/(100+n),
                       rng=rng,
                       progress=print, heartbeat_every=50_000)
    dt = time.perf_counter() - t0
    ckpt_mod.mark_done(ckpt, "algo1", dict(xi=xi, chi=chi, time=dt),
                       log_msg=f"algo1 done in {dt:.1f}s")
    print(f"Algo 1:   xi  = {xi:.4f}    chi  = {chi:.4f}   ({dt:.1f}s)")


def step_algo2(ckpt):
    if ckpt_mod.is_done(ckpt, "algo2"):
        rec = ckpt_mod.get(ckpt, "algo2")
        print(f"Algo 2:   xi  = {rec['xi']:.4f}    chi  = {rec['chi']:.4f}   (cached)")
        return
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    xi, chi = algo2_NSA(sim_Xh, N=200_000, K=128, alpha=ALPHA,
                        gamma=lambda n: 1.0/(100+n),
                        rng=rng,
                        progress=print, heartbeat_every=50_000)
    dt = time.perf_counter() - t0
    ckpt_mod.mark_done(ckpt, "algo2", dict(xi=xi, chi=chi, time=dt),
                       log_msg=f"algo2 done in {dt:.1f}s")
    print(f"Algo 2 (K=128):  xi = {xi:.4f}    chi = {chi:.4f}   ({dt:.1f}s)")


def step_algo3(ckpt):
    if ckpt_mod.is_done(ckpt, "algo3"):
        rec = ckpt_mod.get(ckpt, "algo3")
        print(f"Algo 3:   xi  = {rec['xi']:.4f}    chi  = {rec['chi']:.4f}   (cached)")
        return
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    xi, chi = algo3_MLSA(sim_coupled, sim_Xh,
                         L=3, h0=1/8, M=2,
                         Ns=[100_000, 50_000, 25_000, 12_500],
                         alpha=ALPHA,
                         gamma=lambda n: 1.0/(100+n),
                         rng=rng,
                         progress=print, heartbeat_every=50_000)
    dt = time.perf_counter() - t0
    ckpt_mod.mark_done(ckpt, "algo3", dict(xi=xi, chi=chi, time=dt),
                       log_msg=f"algo3 done in {dt:.1f}s")
    print(f"Algo 3:   xi  = {xi:.4f}    chi  = {chi:.4f}   ({dt:.1f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--restart", action="store_true",
                   help="discard existing smoke-test checkpoint")
    p.add_argument("--status",  action="store_true",
                   help="show what is done and exit")
    args = p.parse_args()

    if args.status:
        ckpt = ckpt_mod.load_or_init(CKPT_NAME, restart=False)
        print(ckpt_mod.summary(ckpt))
        for line in ckpt["logs"][-10:]:
            print("  " + line)
        raise SystemExit(0)

    print(f"Truth:    xi* = {XI_STAR:.4f}    chi* = {CHI_STAR:.4f}\n")
    ckpt = ckpt_mod.load_or_init(CKPT_NAME, restart=args.restart)
    print(ckpt_mod.summary(ckpt) + "\n")

    step_algo1(ckpt)
    step_algo2(ckpt)
    step_algo3(ckpt)

    print("\n[smoke_test] OK")
    print("Tip: delete experiments/smoke.ckpt.pkl to force a clean rerun")
