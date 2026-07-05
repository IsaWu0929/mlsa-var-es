"""
run_case2.py  (with checkpointing)
====================================
Reproduces Figure 3 and Tables 3, 4 from Section 5.2 of
Crepey-Frikha-Louzi (2025): swap on a Black-Scholes rate.

USAGE
-----
    python run_case2.py --quick
    python run_case2.py --paper
    python run_case2.py --restart        # discard checkpoint, start fresh
    python run_case2.py --status         # show what is done

CHECKPOINTING
-------------
Each (epsilon, target, run) triple is persisted *immediately* to
    experiments/case2.ckpt.pkl
so that a PyCharm crash or power loss only loses one sub-task. Re-running
the same command resumes exactly where it stopped.
"""
from __future__ import annotations
import argparse, os, time, csv, pickle
import numpy as np
import matplotlib.pyplot as plt

from mlsa_core    import algo1_SA, algo2_NSA, algo3_MLSA
from case_study_2 import closed_form, make_simulators, DEFAULT_PARAMS
import checkpoint as ckpt_mod


ALPHA     = DEFAULT_PARAMS["alpha"]
EPSILONS  = [1/32, 1/64, 1/128, 1/256, 1/512]
M         = 2
CKPT_NAME = "case2"

XI_STAR, CHI_STAR = closed_form()
sim_X0, sim_Xh, sim_coupled = make_simulators()

TABLE3_VAR = [
    (1/32 , 1/8 , 2, lambda n:  6.0 / (10  + n)),
    (1/64 , 1/16, 2, lambda n: 20.0 / (500 + n)),
    (1/128, 1/16, 3, lambda n: 21.0 / (1e3 + n)),
    (1/256, 1/16, 4, lambda n: 20.0 / (2e3 + n)),
    (1/512, 1/16, 5, lambda n: 21.0 / (3e3 + n)),
]
TABLE3_ES = [
    (1/32 , 1/8 , 2, lambda n:  5.0 / (10  + n)),
    (1/64 , 1/16, 2, lambda n: 20.0 / (500 + n)),
    (1/128, 1/16, 3, lambda n: 20.0 / (500 + n)),
    (1/256, 1/16, 4, lambda n: 20.0 / (750 + n)),
    (1/512, 1/32, 4, lambda n: 50.0 / (2e3 + n)),
]

GAMMA_SA  = lambda n: 100.0 / n
GAMMA_NSA = lambda n:  50.0 / n


def n_iters_per_level(eps, L, h0, M=2):
    base = max(int(np.ceil(eps**(-2.0) * h0)), 100)
    base = min(base, 80_000)
    return [int(round(base / (M ** ell))) for ell in range(L + 1)]


def _run_single(eps, idx, target, run):
    h0_v, L_v, g_v = TABLE3_VAR[idx][1:4]
    h0_e, L_e, g_e = TABLE3_ES [idx][1:4]
    h0, L, g = (h0_v, L_v, g_v) if target == "VaR" else (h0_e, L_e, g_e)

    h_target = h0 / (M ** L)
    K_target = int(round(1.0 / h_target))
    N_target = max(int(np.ceil(eps**(-2.0) / h_target)), 100)
    N_target = min(N_target, 200_000)
    Ns_levels = n_iters_per_level(eps, L, h0, M)

    out = {}
    rng = np.random.default_rng(7 + run)
    t0 = time.perf_counter()
    xi, chi = algo1_SA(sim_X0, N_target, ALPHA, GAMMA_SA, rng=rng)
    out["SA"]   = dict(xi=xi, chi=chi, time=time.perf_counter() - t0)

    rng = np.random.default_rng(7 + run)
    t0 = time.perf_counter()
    xi, chi = algo2_NSA(sim_Xh, N_target, K_target, ALPHA, GAMMA_NSA, rng=rng)
    out["NSA"]  = dict(xi=xi, chi=chi, time=time.perf_counter() - t0)

    rng = np.random.default_rng(7 + run)
    t0 = time.perf_counter()
    xi, chi = algo3_MLSA(sim_coupled, sim_Xh, L, h0, M,
                         Ns_levels, ALPHA, g, rng=rng)
    out["MLSA"] = dict(xi=xi, chi=chi, time=time.perf_counter() - t0)
    return out


def figure3(n_runs, ckpt):
    total_tasks = len(EPSILONS) * 2 * n_runs
    done = sum(1 for k in ckpt["tasks"] if k.startswith("fig3::"))
    print(f"  [fig3] starting at {done}/{total_tasks} sub-tasks already done")

    for idx, eps in enumerate(EPSILONS):
        for target in ["VaR", "ES"]:
            for run in range(n_runs):
                key = f"fig3::eps={eps:.5f}::tg={target}::run={run}"
                if ckpt_mod.is_done(ckpt, key):
                    continue
                t0 = time.perf_counter()
                payload = _run_single(eps, idx, target, run)
                ckpt_mod.mark_done(
                    ckpt, key, payload,
                    log_msg=f"eps=1/{int(round(1/eps))} {target} run={run} "
                            f"({time.perf_counter()-t0:.1f}s)")
                done += 1
            print(f"  [fig3] eps=1/{int(round(1/eps))} {target} done -- "
                  f"{done}/{total_tasks} ({100*done/total_tasks:.1f}%)")

    bench = dict(VaR=XI_STAR, ES=CHI_STAR)
    results = {tg: {al: {"rmse": [], "time": []}
                    for al in ["SA", "NSA", "MLSA"]}
               for tg in ["VaR", "ES"]}
    for idx, eps in enumerate(EPSILONS):
        for target in ["VaR", "ES"]:
            for alg in ["SA", "NSA", "MLSA"]:
                vals, times = [], []
                for run in range(n_runs):
                    key = f"fig3::eps={eps:.5f}::tg={target}::run={run}"
                    rec = ckpt_mod.get(ckpt, key)[alg]
                    val = rec["xi"] if target == "VaR" else rec["chi"]
                    vals.append(val); times.append(rec["time"])
                rmse = float(np.sqrt(np.mean((np.array(vals) - bench[target])**2)))
                results[target][alg]["rmse"].append(rmse)
                results[target][alg]["time"].append(float(np.mean(times)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    style = dict(SA=("s-","tab:green"), NSA=("^-","tab:blue"), MLSA=("o-","tab:orange"))
    for col, target in enumerate(["VaR", "ES"]):
        ax = axes[0, col]
        for alg in ["SA", "NSA", "MLSA"]:
            ax.loglog(results[target][alg]["rmse"], results[target][alg]["time"],
                      style[alg][0], color=style[alg][1],
                      label=alg if alg != "MLSA" else "Multilevel SA")
        ax.set_xlabel("RMSE"); ax.set_ylabel("Average execution time (s)")
        ax.set_title("Value-at-risk" if target == "VaR" else "Expected shortfall")
        ax.legend(); ax.grid(alpha=.3, which="both")
        ax = axes[1, col]
        for alg in ["SA", "NSA", "MLSA"]:
            ax.loglog(EPSILONS, results[target][alg]["time"],
                      style[alg][0], color=style[alg][1],
                      label=alg if alg != "MLSA" else "Multilevel SA")
        ax.set_xlabel(r"Prescribed accuracy $\varepsilon$")
        ax.set_ylabel("Average execution time (s)")
        ax.legend(); ax.grid(alpha=.3, which="both"); ax.invert_xaxis()
    fig.suptitle("Fig. 3 - Performance comparison of Algorithms 1, 2 and 3")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig3_case2.png", dpi=150)
    fig.savefig("figures/fig3_case2.pdf")
    plt.close(fig)
    print("[fig3] saved")
    return results


def make_table3(out_path="tables/table3.csv"):
    rows = [("epsilon","VaR h0","VaR L","VaR gamma_n",
                       "ES h0","ES L","ES gamma_n")]
    for v, e in zip(TABLE3_VAR, TABLE3_ES):
        rows.append((f"1/{int(round(1/v[0]))}",
                     f"1/{int(round(1/v[1]))}", v[2], "(see code)",
                     f"1/{int(round(1/e[1]))}", e[2], "(see code)"))
    os.makedirs("tables", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[table3] saved")


def make_table4(results, out_path="tables/table4.csv"):
    rows = [("SA scheme","VaR RMSE slope","VaR eps slope",
                         "ES RMSE slope","ES eps slope")]
    for alg in ["NSA", "MLSA", "SA"]:
        r_var = np.log(results["VaR"][alg]["rmse"])
        r_es  = np.log(results["ES" ][alg]["rmse"])
        t_var = np.log(results["VaR"][alg]["time"])
        t_es  = np.log(results["ES" ][alg]["time"])
        ln_eps = np.log(EPSILONS)
        s1, _ = np.polyfit(r_var,  t_var, 1)
        s2, _ = np.polyfit(ln_eps, t_var, 1)
        s3, _ = np.polyfit(r_es,   t_es,  1)
        s4, _ = np.polyfit(ln_eps, t_es,  1)
        rows.append((alg, f"{s1:.2f}", f"{s2:.2f}", f"{s3:.2f}", f"{s4:.2f}"))
    os.makedirs("tables", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[table4] saved")
    for r in rows: print("  " + " | ".join(map(str, r)))


def print_status():
    ckpt = ckpt_mod.load_or_init(CKPT_NAME, restart=False)
    print(ckpt_mod.summary(ckpt))
    fig3_keys = [k for k in ckpt["tasks"] if k.startswith("fig3::")]
    print(f"  Figure 3 sub-tasks done: {len(fig3_keys)}")
    for line in ckpt["logs"][-10:]:
        print("    " + line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick",   action="store_true")
    p.add_argument("--paper",   action="store_true")
    p.add_argument("--restart", action="store_true")
    p.add_argument("--status",  action="store_true")
    args = p.parse_args()

    if args.status:
        print_status(); raise SystemExit(0)

    n_runs = 30 if args.quick else 200

    print(f"xi*  ~ {XI_STAR:.4f}    chi* ~ {CHI_STAR:.4f}")
    ckpt = ckpt_mod.load_or_init(CKPT_NAME, restart=args.restart)
    print(ckpt_mod.summary(ckpt) + "\n")

    print(">>> Table 3"); make_table3()
    print(">>> Figure 3 (slow) -- safe to interrupt; rerun to resume")
    results = figure3(n_runs, ckpt)
    print(">>> Table 4"); make_table4(results)

    os.makedirs("experiments", exist_ok=True)
    payload = dict(epsilons=EPSILONS, n_runs=n_runs,
                   xi_star=XI_STAR, chi_star=CHI_STAR, results=results)
    with open("experiments/case2_results.pkl", "wb") as f:
        pickle.dump(payload, f)
    print("\nDONE -- experiments/case2_results.pkl ready")
