"""
run_heston.py
==============
Runs SA / NSA / MLSA on the Heston case study and produces:
    - figures/heston_fig1_weak_error.pdf      (weak error linearity)
    - figures/heston_fig2_time_vs_rmse.pdf    (time vs RMSE)
    - figures/heston_fig3_time_vs_eps.pdf     (time vs prescribed accuracy)
    - experiments/case_heston_results.pkl     (raw results)
    - tables/heston_slopes.csv                (regressed log-log slopes)

USAGE
-----
    python run_heston.py --quick            # ~3 min, n_runs=10
    python run_heston.py --paper            # ~hours, n_runs=200

Author: thesis student of Wei-Biao Wu, UChicago.
"""
from __future__ import annotations
import argparse, os, time, csv, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from mlsa_core         import algo1_SA, algo2_NSA, algo3_MLSA
from case_study_heston import HestonParams, make_simulators, compute_benchmark

# ---------------------------------------------------------------------------
#  Plotting style for thesis-quality output
# ---------------------------------------------------------------------------
rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3,
    "lines.linewidth": 1.6, "lines.markersize": 6,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

# ---------------------------------------------------------------------------
#  Experiment configuration
# ---------------------------------------------------------------------------
P = HestonParams()
EPSILONS  = [1/32, 1/64, 1/128, 1/256]    # 4 values; Heston is more expensive
EPS_LABELS = [r"$\frac{1}{32}$", r"$\frac{1}{64}$",
              r"$\frac{1}{128}$", r"$\frac{1}{256}$"]
M = 2
H_VALUES_FIG1 = [1/4, 1/8, 1/16, 1/32, 1/64]    # for weak-error figure

# Hyperparameter recipe by prescribed accuracy
# (h0, L, gamma_factor, gamma_denom_base)  --  gamma_n = factor / (denom_base + n)
RECIPE_VAR = {
    1/32 : (1/8 , 1, 5.0, 100),
    1/64 : (1/8 , 2, 3.0, 200),
    1/128: (1/8 , 3, 2.0, 500),
    1/256: (1/8 , 4, 1.0, 1000),
}
RECIPE_ES = {
    1/32 : (1/8 , 1, 3.0, 100),
    1/64 : (1/8 , 2, 2.0, 300),
    1/128: (1/8 , 3, 1.5, 500),
    1/256: (1/8 , 4, 1.0, 1000),
}

# Step sizes for SA / NSA (single-level)
GAMMA_SA  = lambda n: 5.0 / (100 + n)
GAMMA_NSA = lambda n: 5.0 / (100 + n)


def n_iters_per_level(eps: float, L: int, h0: float, M: int = 2):
    """Heuristic mapping: eps -> (N_0, ..., N_L) decaying geometrically."""
    base = max(int(np.ceil(eps**(-2.0) * 0.3)), 200)
    base = min(base, 15_000)            # cap to avoid blow-up at small eps
    return [int(round(base / (M ** ell))) for ell in range(L + 1)]


# ---------------------------------------------------------------------------
#  Globals (built once)
# ---------------------------------------------------------------------------
SIM_X0, SIM_Xh, SIM_COUPLED, _ = make_simulators(P, n_steps_to_delta=1, n_steps_post_delta=2)


# ---------------------------------------------------------------------------
#  FIGURE 1: weak error linearity in h
# ---------------------------------------------------------------------------
def figure1(xi_star: float, chi_star: float, n_runs: int, n_iter: int):
    """
    Replicates Fig. 1 of the paper: shows that xi^h - xi^0 is linear in h.
    Runs Algorithm 2 for h in {1/4,...,1/64} and averages.
    """
    print("[fig1] Computing weak-error linearity ...")
    xi_means, chi_means = [], []
    for h in H_VALUES_FIG1:
        K = int(round(1.0 / h))
        xis, chis = [], []
        for run in range(n_runs):
            rng = np.random.default_rng(91000 + run)
            xi, chi = algo2_NSA(SIM_Xh, n_iter, K, P.alpha,
                                lambda n: 5.0 / (200 + n), rng=rng)
            xis.append(xi); chis.append(chi)
        xi_means.append(np.mean(xis));  chi_means.append(np.mean(chis))
        print(f"   h={h:.4f} (K={K:3d})  xi_mean={np.mean(xis):+.4f}  "
              f"chi_mean={np.mean(chis):+.4f}")

    xi_diff  = np.array(xi_means)  - xi_star
    chi_diff = np.array(chi_means) - chi_star
    h_arr    = np.array(H_VALUES_FIG1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h_arr, xi_diff,  '-^', color="tab:blue",   label=r"VaR  $\bar\xi^h-\xi^0_\star$")
    axes[0].plot(h_arr, chi_diff, '-o', color="tab:orange", label=r"ES  $\bar\chi^h-\chi^0_\star$")
    axes[0].set_xlabel("Bias parameter $h$"); axes[0].set_ylabel("Centred risk measure")
    axes[0].set_title("Heston: centred risk measures")
    axes[0].legend()

    axes[1].plot(h_arr, xi_diff  / h_arr, '-^', color="tab:blue",   label=r"VaR")
    axes[1].plot(h_arr, chi_diff / h_arr, '-o', color="tab:orange", label=r"ES")
    axes[1].set_xlabel("Bias parameter $h$")
    axes[1].set_ylabel(r"Rescaled centred risk measure $h^{-1}(\cdot)$")
    axes[1].set_title("Heston: rescaled centred risk measures")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/heston_fig1_weak_error.pdf")
    fig.savefig("figures/heston_fig1_weak_error.png", dpi=160)
    plt.close(fig)
    print("[fig1] saved")
    return dict(h=H_VALUES_FIG1, xi_mean=list(xi_means), chi_mean=list(chi_means))


# ---------------------------------------------------------------------------
#  Helper: run one (eps, target) over n_runs replicates for all 3 algos
# ---------------------------------------------------------------------------
def _run_one(eps: float, target: str, n_runs: int,
              xi_star: float, chi_star: float):
    recipe = RECIPE_VAR if target == "VaR" else RECIPE_ES
    h0, L, gf, gd = recipe[eps]
    h_target = h0 / (M ** L)
    K_target = int(round(1.0 / h_target))
    Ns = n_iters_per_level(eps, L, h0, M)
    N_single = sum(Ns)               # comparable budget for SA / NSA

    g_ml = lambda n, _gf=gf, _gd=gd: _gf / (_gd + n)
    bench = xi_star if target == "VaR" else chi_star

    sa_v, sa_t, ns_v, ns_t, ml_v, ml_t = [], [], [], [], [], []
    for run in range(n_runs):
        # ---- SA  (uses high-K_proxy proxy to simulate X_0)
        rng = np.random.default_rng(50_000 + run)
        t0 = time.perf_counter()
        xi, chi = algo1_SA(SIM_X0, N_single, P.alpha, GAMMA_SA, rng=rng)
        sa_t.append(time.perf_counter() - t0)
        sa_v.append(xi if target == "VaR" else chi)

        # ---- NSA  (single-level at finest h)
        rng = np.random.default_rng(50_000 + run)
        t0 = time.perf_counter()
        xi, chi = algo2_NSA(SIM_Xh, N_single, K_target, P.alpha,
                            GAMMA_NSA, rng=rng)
        ns_t.append(time.perf_counter() - t0)
        ns_v.append(xi if target == "VaR" else chi)

        # ---- MLSA
        rng = np.random.default_rng(50_000 + run)
        t0 = time.perf_counter()
        xi, chi = algo3_MLSA(SIM_COUPLED, SIM_Xh, L, h0, M, Ns,
                             P.alpha, g_ml, rng=rng)
        ml_t.append(time.perf_counter() - t0)
        ml_v.append(xi if target == "VaR" else chi)

    rmse = lambda v: float(np.sqrt(np.mean((np.array(v) - bench) ** 2)))
    return dict(
        SA   = (rmse(sa_v),  float(np.mean(sa_t))),
        NSA  = (rmse(ns_v),  float(np.mean(ns_t))),
        MLSA = (rmse(ml_v),  float(np.mean(ml_t))),
    )


# ---------------------------------------------------------------------------
#  FIGURES 2 & 3 (combined sweep)
# ---------------------------------------------------------------------------
def main_sweep(n_runs: int, xi_star: float, chi_star: float):
    """Returns nested dict of results keyed by [target][algo] -> {rmse: list, time: list}."""
    results = {tg: {alg: {"rmse": [], "time": []}
                    for alg in ["SA", "NSA", "MLSA"]}
               for tg in ["VaR", "ES"]}

    for eps in EPSILONS:
        for target in ["VaR", "ES"]:
            t0 = time.perf_counter()
            r = _run_one(eps, target, n_runs, xi_star, chi_star)
            for alg in ["SA", "NSA", "MLSA"]:
                results[target][alg]["rmse"].append(r[alg][0])
                results[target][alg]["time"].append(r[alg][1])
            print(f"  eps=1/{int(round(1/eps)):d} {target} ({time.perf_counter()-t0:.1f}s):  "
                  f"SA={r['SA'][0]:.3e}  NSA={r['NSA'][0]:.3e}  MLSA={r['MLSA'][0]:.3e}")
    return results


def figure2_3(results, n_runs: int):
    """Two separate figures: (2) time vs RMSE, (3) time vs prescribed accuracy."""
    style = dict(SA=("s-", "tab:green",  "SA (Alg.1)"),
                 NSA=("^-", "tab:blue",   "Nested SA (Alg.2)"),
                 MLSA=("o-", "tab:orange", "Multilevel SA (Alg.3)"))

    # ----- Figure 2 (time vs RMSE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for col, target in enumerate(["VaR", "ES"]):
        ax = axes[col]
        for alg in ["SA", "NSA", "MLSA"]:
            ax.loglog(results[target][alg]["rmse"],
                      results[target][alg]["time"],
                      style[alg][0], color=style[alg][1], label=style[alg][2])
        ax.set_xlabel("RMSE"); ax.set_ylabel("Average execution time (s)")
        ax.set_title(f"Heston {target}")
        ax.legend()
    fig.suptitle("Performance: execution time vs RMSE")
    fig.tight_layout()
    fig.savefig("figures/heston_fig2_time_vs_rmse.pdf")
    fig.savefig("figures/heston_fig2_time_vs_rmse.png", dpi=160)
    plt.close(fig)
    print("[fig2] saved")

    # ----- Figure 3 (time vs eps)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for col, target in enumerate(["VaR", "ES"]):
        ax = axes[col]
        for alg in ["SA", "NSA", "MLSA"]:
            ax.loglog(EPSILONS, results[target][alg]["time"],
                      style[alg][0], color=style[alg][1], label=style[alg][2])
        ax.set_xlabel(r"Prescribed accuracy $\varepsilon$")
        ax.set_ylabel("Average execution time (s)")
        ax.set_title(f"Heston {target}")
        ax.invert_xaxis()
        ax.legend()
    fig.suptitle("Performance: execution time vs prescribed accuracy")
    fig.tight_layout()
    fig.savefig("figures/heston_fig3_time_vs_eps.pdf")
    fig.savefig("figures/heston_fig3_time_vs_eps.png", dpi=160)
    plt.close(fig)
    print("[fig3] saved")


def regress_slopes(results, out_csv="tables/heston_slopes.csv"):
    """Regress log time on log RMSE and log eps; save Table 2 analogue."""
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


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()

    n_runs_fig1 = 5  if args.quick else 100
    n_iter_fig1 = 20_000 if args.quick else 200_000
    n_runs_main = 10 if args.quick else 200

    print(f"Heston parameters: {P}")
    print("Computing benchmark (xi*, chi*) ...")
    t0 = time.time()
    xi_star, chi_star = compute_benchmark(P, n_samples=20_000 if args.quick else 50_000)
    print(f"  xi*  = {xi_star:.4f}    chi* = {chi_star:.4f}    (took {time.time()-t0:.1f}s)\n")

    fig1 = figure1(xi_star, chi_star, n_runs_fig1, n_iter_fig1)
    print()

    print("Running main sweep (this is the slow part) ...")
    results = main_sweep(n_runs_main, xi_star, chi_star)
    figure2_3(results, n_runs_main)
    regress_slopes(results)

    # ----- Save raw results
    os.makedirs("experiments", exist_ok=True)
    payload = dict(
        params=P.__dict__, xi_star=xi_star, chi_star=chi_star,
        epsilons=EPSILONS, n_runs=n_runs_main,
        weak_error=fig1, results=results,
    )
    with open("experiments/case_heston_results.pkl", "wb") as f:
        pickle.dump(payload, f)
    print("[pkl ] saved to experiments/case_heston_results.pkl")
    print("\nDONE  -- check ./figures and ./tables")
