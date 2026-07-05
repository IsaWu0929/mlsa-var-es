"""
run_case1.py  (with checkpointing)
====================================
Reproduces Figure 1, Figure 2 and Tables 1, 2 from Section 5.1 of
Crepey-Frikha-Louzi (2025).

USAGE
-----
    python run_case1.py --quick          # ~3 min,  small N_runs
    python run_case1.py --paper          # ~hours,  paper-grade settings
    python run_case1.py --restart        # ignore any existing checkpoint
    python run_case1.py --status         # just print what's done so far

CHECKPOINTING
-------------
Every (epsilon, target, run) result is persisted to
    experiments/case1.ckpt.pkl
*immediately* upon completion.  If the process dies (PyCharm crash, OOM,
power loss, etc.) re-running the same command will SKIP all already-done
sub-tasks and pick up exactly where it left off.

Use --restart to discard the checkpoint and start over.
Use --status to inspect the checkpoint without running anything.
"""
from __future__ import annotations
import argparse, os, time, csv, pickle
import numpy as np
import matplotlib.pyplot as plt

from mlsa_core    import algo1_SA, algo2_NSA, algo3_MLSA
from case_study_1 import closed_form, make_simulators
import checkpoint as ckpt_mod


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
ALPHA = 0.975
DELTA = 0.5
EPSILONS  = [1/32, 1/64, 1/128, 1/256, 1/512]
M = 2
CKPT_NAME = "case1"

XI_STAR, CHI_STAR = closed_form(ALPHA, DELTA)
sim_X0, sim_Xh, sim_coupled = make_simulators(DELTA)

TABLE1_VAR = [
    (1/32 , 1/16, 1, lambda n:  2.0 / (2.5e3 + n)),
    (1/64 , 1/32, 1, lambda n:  2.0 / (4.0e3 + n)),
    (1/128, 1/32, 2, lambda n: 0.75 / (9.0e3 + n)),
    (1/256, 1/32, 3, lambda n: 0.25 / (1.0e4 + n)),
    (1/512, 1/32, 4, lambda n: 0.09 / (1.0e4 + n)),
]
TABLE1_ES = [
    (1/32 , 1/16, 1, lambda n: 0.1 / (1.0e4 + n)),
    (1/64 , 1/32, 1, lambda n: 0.1 / (1.0e4 + n)),
    (1/128, 1/32, 2, lambda n: 0.1 / (1.0e4 + n)),
    (1/256, 1/32, 3, lambda n: 0.1 / (2.0e4 + n)),
    (1/512, 1/32, 4, lambda n: 0.1 / (2.5e4 + n)),
]

GAMMA_SA_VAR  = lambda n: 1.0 / (100 + n)
GAMMA_SA_ES   = lambda n: 0.1 / (2.5e4 + n)
GAMMA_NSA_VAR = GAMMA_SA_VAR
GAMMA_NSA_ES  = GAMMA_SA_ES


def n_iters_per_level(eps, L, h0, M=2, beta=1.0):
    base = max(int(np.ceil(eps**(-2.0) * h0)), 100)
    base = min(base, 80_000)
    return [int(round(base / (M ** ell))) for ell in range(L + 1)]


# ---------------------------------------------------------------------------
#  FIGURE 1  -- weak-error linearity
# ---------------------------------------------------------------------------
def figure1(n_runs, n_iter, ckpt):
    h_vals = [1/10, 1/20, 1/50, 1/100, 1/200]
    gamma  = lambda n: 0.1 / (1e4 + n)

    for h in h_vals:
        K = int(round(1.0 / h))
        for run in range(n_runs):
            key = f"fig1::h={h:.4f}::run={run}"
            if ckpt_mod.is_done(ckpt, key):
                continue
            rng = np.random.default_rng(12345 + run)
            xi, chi = algo2_NSA(sim_Xh, n_iter, K, ALPHA, gamma, rng=rng)
            ckpt_mod.mark_done(ckpt, key, dict(xi=xi, chi=chi))
        n_done = sum(1 for r in range(n_runs)
                     if ckpt_mod.is_done(ckpt, f"fig1::h={h:.4f}::run={r}"))
        print(f"  [fig1] h={h:.4f} (K={K})  -- {n_done}/{n_runs} runs done")

    xi_means, chi_means = [], []
    for h in h_vals:
        xis = [ckpt_mod.get(ckpt, f"fig1::h={h:.4f}::run={r}")["xi"]
               for r in range(n_runs)]
        chis = [ckpt_mod.get(ckpt, f"fig1::h={h:.4f}::run={r}")["chi"]
                for r in range(n_runs)]
        xi_means.append(np.mean(xis));  chi_means.append(np.mean(chis))

    xi_diff  = np.array(xi_means) - XI_STAR
    chi_diff = np.array(chi_means) - CHI_STAR
    h_arr    = np.array(h_vals)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h_arr, xi_diff,  '-^', label="Value-at-risk")
    axes[0].plot(h_arr, chi_diff, '-o', label="Expected shortfall")
    axes[0].set_xlabel("Bias parameter $h$"); axes[0].set_ylabel("Centered risk measure")
    axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(h_arr, xi_diff  / h_arr, '-^', label="Value-at-risk")
    axes[1].plot(h_arr, chi_diff / h_arr, '-o', label="Expected shortfall")
    axes[1].set_xlabel("Bias parameter $h$")
    axes[1].set_ylabel(r"$h^{-1}(\cdot)$")
    axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig1_case1.png", dpi=150)
    fig.savefig("figures/fig1_case1.pdf")
    plt.close(fig)
    print("[fig1] saved")


# ---------------------------------------------------------------------------
#  Per-run kernel
# ---------------------------------------------------------------------------
def _run_single(eps, idx, target, run):
    h0_var, L_var, g_var = TABLE1_VAR[idx][1:4]
    h0_es,  L_es,  g_es  = TABLE1_ES [idx][1:4]
    h0, L, g = (h0_var, L_var, g_var) if target == "VaR" else (h0_es, L_es, g_es)

    h_target  = h0 / (M ** L)
    K_target  = int(round(1.0 / h_target))
    N_target  = max(int(np.ceil(eps**(-2.0) / h_target)), 100)
    N_target  = min(N_target, 200_000)
    Ns_levels = n_iters_per_level(eps, L, h0, M, beta=1.0)

    g_sa  = GAMMA_SA_VAR  if target == "VaR" else GAMMA_SA_ES
    g_nsa = GAMMA_NSA_VAR if target == "VaR" else GAMMA_NSA_ES

    out = {}
    rng = np.random.default_rng(2024 + run)
    t0 = time.perf_counter()
    xi, chi = algo1_SA(sim_X0, N_target, ALPHA, g_sa, rng=rng)
    out["SA"]   = dict(xi=xi, chi=chi, time=time.perf_counter() - t0)

    rng = np.random.default_rng(2024 + run)
    t0 = time.perf_counter()
    xi, chi = algo2_NSA(sim_Xh, N_target, K_target, ALPHA, g_nsa, rng=rng)
    out["NSA"]  = dict(xi=xi, chi=chi, time=time.perf_counter() - t0)

    rng = np.random.default_rng(2024 + run)
    t0 = time.perf_counter()
    xi, chi = algo3_MLSA(sim_coupled, sim_Xh, L, h0, M,
                         Ns_levels, ALPHA, g, rng=rng)
    out["MLSA"] = dict(xi=xi, chi=chi, time=time.perf_counter() - t0)
    return out


# ---------------------------------------------------------------------------
#  FIGURE 2 driver
# ---------------------------------------------------------------------------
def figure2(n_runs, ckpt):
    total_tasks = len(EPSILONS) * 2 * n_runs
    done = sum(1 for k in ckpt["tasks"] if k.startswith("fig2::"))
    print(f"  [fig2] starting at {done}/{total_tasks} sub-tasks already done")

    for idx, eps in enumerate(EPSILONS):
        for target in ["VaR", "ES"]:
            for run in range(n_runs):
                key = f"fig2::eps={eps:.5f}::tg={target}::run={run}"
                if ckpt_mod.is_done(ckpt, key):
                    continue
                t0 = time.perf_counter()
                payload = _run_single(eps, idx, target, run)
                ckpt_mod.mark_done(
                    ckpt, key, payload,
                    log_msg=f"eps=1/{int(round(1/eps))} {target} run={run} "
                            f"({time.perf_counter()-t0:.1f}s)")
                done += 1
            print(f"  [fig2] eps=1/{int(round(1/eps))} {target} done -- "
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
                    key = f"fig2::eps={eps:.5f}::tg={target}::run={run}"
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
    fig.suptitle("Fig. 2 - Performance comparison of Algorithms 1, 2 and 3")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig2_case1.png", dpi=150)
    fig.savefig("figures/fig2_case1.pdf")
    plt.close(fig)
    print("[fig2] saved")
    return results


def make_table1(out_path="tables/table1.csv"):
    rows = [("epsilon","VaR h0","VaR L","VaR gamma_n",
                       "ES h0","ES L","ES gamma_n")]
    for v, e in zip(TABLE1_VAR, TABLE1_ES):
        rows.append((f"1/{int(round(1/v[0]))}",
                     f"1/{int(round(1/v[1]))}", v[2], "(see code)",
                     f"1/{int(round(1/e[1]))}", e[2], "(see code)"))
    os.makedirs("tables", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[table1] saved")


def make_table2(results, out_path="tables/table2.csv"):
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
    print(f"[table2] saved")
    for r in rows: print("  " + " | ".join(map(str, r)))


def print_status():
    ckpt = ckpt_mod.load_or_init(CKPT_NAME, restart=False)
    print(ckpt_mod.summary(ckpt))
    fig1_keys = [k for k in ckpt["tasks"] if k.startswith("fig1::")]
    fig2_keys = [k for k in ckpt["tasks"] if k.startswith("fig2::")]
    print(f"  Figure 1 sub-tasks done: {len(fig1_keys)}")
    print(f"  Figure 2 sub-tasks done: {len(fig2_keys)}")
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

    n_runs_fig1 = 50  if args.quick else 200
    n_iter_fig1 = 10**5 if args.quick else 10**6
    n_runs_fig2 = 30  if args.quick else 200

    print(f"xi*  ~ {XI_STAR:.4f}    chi* ~ {CHI_STAR:.4f}")
    ckpt = ckpt_mod.load_or_init(CKPT_NAME, restart=args.restart)
    print(ckpt_mod.summary(ckpt) + "\n")

    print(">>> Table 1");  make_table1()
    print(">>> Figure 1"); figure1(n_runs_fig1, n_iter_fig1, ckpt)
    print(">>> Figure 2 (slow part) -- safe to interrupt; rerun to resume")
    results = figure2(n_runs_fig2, ckpt)
    print(">>> Table 2");  make_table2(results)

    os.makedirs("experiments", exist_ok=True)
    payload = dict(epsilons=EPSILONS, n_runs=n_runs_fig2,
                   xi_star=XI_STAR, chi_star=CHI_STAR, results=results)
    with open("experiments/case1_results.pkl", "wb") as f:
        pickle.dump(payload, f)
    print("\nDONE -- experiments/case1_results.pkl ready")
