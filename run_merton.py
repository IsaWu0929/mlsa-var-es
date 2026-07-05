"""
run_merton.py
==============
Runs SA / NSA / MLSA on the Merton jump-diffusion case study and produces:
    - figures/merton_fig1_weak_error.pdf
    - figures/merton_fig2_time_vs_rmse.pdf
    - figures/merton_fig3_time_vs_eps.pdf
    - experiments/case_merton_results.pkl
    - tables/merton_slopes.csv

USAGE
-----
    python run_merton.py --quick        # ~1 min
    python run_merton.py --paper        # ~minutes (Merton is fast)
"""
from __future__ import annotations
import argparse, os, time, csv, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from mlsa_core         import algo1_SA, algo2_NSA, algo3_MLSA
from case_study_merton import MertonParams, make_simulators, compute_benchmark

rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3,
    "lines.linewidth": 1.6, "lines.markersize": 6,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
P = MertonParams()
EPSILONS = [1/32, 1/64, 1/128, 1/256]
M = 2
H_VALUES_FIG1 = [1/4, 1/8, 1/16, 1/32, 1/64, 1/128]

# Hyperparameter recipes  (h0, L, gamma_factor, gamma_denom_base)
RECIPE_VAR = {
    1/32 : (1/8 , 1, 2.0, 100),
    1/64 : (1/8 , 2, 1.0, 200),
    1/128: (1/8 , 3, 0.5, 500),
    1/256: (1/8 , 4, 0.3, 1000),
}
RECIPE_ES = {
    1/32 : (1/8 , 1, 2.0, 100),
    1/64 : (1/8 , 2, 1.0, 200),
    1/128: (1/8 , 3, 1.0, 500),
    1/256: (1/8 , 4, 0.5, 1000),
}

GAMMA_SA  = lambda n: 1.0 / (100 + n)
GAMMA_NSA = lambda n: 1.0 / (100 + n)


def n_iters_per_level(eps, L, h0, M=2):
    base = max(int(np.ceil(eps**(-2.0) * 0.5)), 200)
    base = min(base, 80_000)            # cap to avoid runaway memory at small eps
    return [int(round(base / (M ** ell))) for ell in range(L + 1)]


SIM_X0, SIM_Xh, SIM_COUPLED, _ = make_simulators(P)


# ---------------------------------------------------------------------------
#  Figure 1: weak error in h
# ---------------------------------------------------------------------------
def figure1(xi_star, chi_star, n_runs, n_iter):
    print("[fig1] Computing weak-error linearity ...")
    xi_means, chi_means = [], []
    for h in H_VALUES_FIG1:
        K = int(round(1.0 / h))
        xis, chis = [], []
        for run in range(n_runs):
            rng = np.random.default_rng(7000 + run)
            xi, chi = algo2_NSA(SIM_Xh, n_iter, K, P.alpha,
                                lambda n: 1.0 / (200 + n), rng=rng)
            xis.append(xi); chis.append(chi)
        xi_means.append(np.mean(xis));  chi_means.append(np.mean(chis))
        print(f"   h={h:.4f}  K={K:3d}  xi_mean={np.mean(xis):+.4f}  "
              f"chi_mean={np.mean(chis):+.4f}")

    xi_diff  = np.array(xi_means)  - xi_star
    chi_diff = np.array(chi_means) - chi_star
    h_arr    = np.array(H_VALUES_FIG1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h_arr, xi_diff,  '-^', color="tab:blue",   label=r"VaR  $\bar\xi^h-\xi^0_\star$")
    axes[0].plot(h_arr, chi_diff, '-o', color="tab:orange", label=r"ES  $\bar\chi^h-\chi^0_\star$")
    axes[0].set_xlabel("Bias parameter $h$"); axes[0].set_ylabel("Centred risk measure")
    axes[0].set_title("Merton: centred risk measures"); axes[0].legend()

    axes[1].plot(h_arr, xi_diff  / h_arr, '-^', color="tab:blue",   label="VaR")
    axes[1].plot(h_arr, chi_diff / h_arr, '-o', color="tab:orange", label="ES")
    axes[1].set_xlabel("Bias parameter $h$")
    axes[1].set_ylabel(r"Rescaled centred risk measure $h^{-1}(\cdot)$")
    axes[1].set_title("Merton: rescaled centred risk measures"); axes[1].legend()

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/merton_fig1_weak_error.pdf")
    fig.savefig("figures/merton_fig1_weak_error.png", dpi=160)
    plt.close(fig)
    print("[fig1] saved")
    return dict(h=H_VALUES_FIG1, xi_mean=list(xi_means), chi_mean=list(chi_means))


def _run_one(eps, target, n_runs, xi_star, chi_star):
    recipe = RECIPE_VAR if target == "VaR" else RECIPE_ES
    h0, L, gf, gd = recipe[eps]
    h_target = h0 / (M ** L)
    K_target = int(round(1.0 / h_target))
    Ns = n_iters_per_level(eps, L, h0, M)
    N_single = sum(Ns)
    g_ml = lambda n, _gf=gf, _gd=gd: _gf / (_gd + n)
    bench = xi_star if target == "VaR" else chi_star

    sa_v, sa_t, ns_v, ns_t, ml_v, ml_t = [], [], [], [], [], []
    for run in range(n_runs):
        rng = np.random.default_rng(60_000 + run)
        t0 = time.perf_counter()
        xi, chi = algo1_SA(SIM_X0, N_single, P.alpha, GAMMA_SA, rng=rng)
        sa_t.append(time.perf_counter() - t0)
        sa_v.append(xi if target == "VaR" else chi)

        rng = np.random.default_rng(60_000 + run)
        t0 = time.perf_counter()
        xi, chi = algo2_NSA(SIM_Xh, N_single, K_target, P.alpha, GAMMA_NSA, rng=rng)
        ns_t.append(time.perf_counter() - t0)
        ns_v.append(xi if target == "VaR" else chi)

        rng = np.random.default_rng(60_000 + run)
        t0 = time.perf_counter()
        xi, chi = algo3_MLSA(SIM_COUPLED, SIM_Xh, L, h0, M, Ns,
                             P.alpha, g_ml, rng=rng)
        ml_t.append(time.perf_counter() - t0)
        ml_v.append(xi if target == "VaR" else chi)

    rmse = lambda v: float(np.sqrt(np.mean((np.array(v) - bench) ** 2)))
    return dict(
        SA  =(rmse(sa_v),  float(np.mean(sa_t))),
        NSA =(rmse(ns_v),  float(np.mean(ns_t))),
        MLSA=(rmse(ml_v),  float(np.mean(ml_t))),
    )


def main_sweep(n_runs, xi_star, chi_star):
    results = {tg: {al: {"rmse": [], "time": []}
                    for al in ["SA", "NSA", "MLSA"]}
               for tg in ["VaR", "ES"]}
    for eps in EPSILONS:
        for target in ["VaR", "ES"]:
            t0 = time.perf_counter()
            r = _run_one(eps, target, n_runs, xi_star, chi_star)
            for alg in ["SA", "NSA", "MLSA"]:
                results[target][alg]["rmse"].append(r[alg][0])
                results[target][alg]["time"].append(r[alg][1])
            print(f"  eps=1/{int(round(1/eps))} {target} ({time.perf_counter()-t0:.1f}s):  "
                  f"SA={r['SA'][0]:.3e}  NSA={r['NSA'][0]:.3e}  MLSA={r['MLSA'][0]:.3e}")
    return results


def figure2_3(results):
    style = dict(SA=("s-", "tab:green",  "SA (Alg.1)"),
                 NSA=("^-", "tab:blue",   "Nested SA (Alg.2)"),
                 MLSA=("o-", "tab:orange", "Multilevel SA (Alg.3)"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for col, target in enumerate(["VaR", "ES"]):
        ax = axes[col]
        for alg in ["SA", "NSA", "MLSA"]:
            ax.loglog(results[target][alg]["rmse"], results[target][alg]["time"],
                      style[alg][0], color=style[alg][1], label=style[alg][2])
        ax.set_xlabel("RMSE"); ax.set_ylabel("Average execution time (s)")
        ax.set_title(f"Merton {target}"); ax.legend()
    fig.suptitle("Merton: execution time vs RMSE")
    fig.tight_layout()
    fig.savefig("figures/merton_fig2_time_vs_rmse.pdf")
    fig.savefig("figures/merton_fig2_time_vs_rmse.png", dpi=160)
    plt.close(fig); print("[fig2] saved")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for col, target in enumerate(["VaR", "ES"]):
        ax = axes[col]
        for alg in ["SA", "NSA", "MLSA"]:
            ax.loglog(EPSILONS, results[target][alg]["time"],
                      style[alg][0], color=style[alg][1], label=style[alg][2])
        ax.set_xlabel(r"Prescribed accuracy $\varepsilon$")
        ax.set_ylabel("Average execution time (s)")
        ax.set_title(f"Merton {target}"); ax.invert_xaxis(); ax.legend()
    fig.suptitle("Merton: execution time vs prescribed accuracy")
    fig.tight_layout()
    fig.savefig("figures/merton_fig3_time_vs_eps.pdf")
    fig.savefig("figures/merton_fig3_time_vs_eps.png", dpi=160)
    plt.close(fig); print("[fig3] saved")


def regress_slopes(results, out_csv="tables/merton_slopes.csv"):
    rows = [("algorithm", "VaR slope (RMSE)", "VaR slope (eps)",
                          "ES slope (RMSE)",  "ES slope (eps)")]
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
        rows.append((alg, f"{s1:+.2f}", f"{s2:+.2f}", f"{s3:+.2f}", f"{s4:+.2f}"))
    os.makedirs("tables", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[slopes] saved to {out_csv}")
    for r in rows: print("   " + " | ".join(map(str, r)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    n_runs_fig1 = 5  if args.quick else 100
    n_iter_fig1 = 20_000 if args.quick else 200_000
    n_runs_main = 10 if args.quick else 200

    print(f"Merton parameters: {P}")
    print("Computing semi-analytical benchmark ...")
    t0 = time.time()
    xi_star, chi_star = compute_benchmark(P, n_samples=200_000 if args.quick else 1_000_000)
    print(f"  xi*  = {xi_star:.4f}    chi* = {chi_star:.4f}    ({time.time()-t0:.1f}s)\n")

    fig1 = figure1(xi_star, chi_star, n_runs_fig1, n_iter_fig1)
    print()

    print("Main sweep ...")
    results = main_sweep(n_runs_main, xi_star, chi_star)
    figure2_3(results)
    regress_slopes(results)

    os.makedirs("experiments", exist_ok=True)
    payload = dict(params=P.__dict__, xi_star=xi_star, chi_star=chi_star,
                   epsilons=EPSILONS, n_runs=n_runs_main,
                   weak_error=fig1, results=results)
    with open("experiments/case_merton_results.pkl", "wb") as f:
        pickle.dump(payload, f)
    print("[pkl ] saved to experiments/case_merton_results.pkl")
    print("\nDONE")
