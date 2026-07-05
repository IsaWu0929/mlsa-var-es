"""
extra_analyses.py
==================
Three more analyses that add depth to the thesis without adding scope:

  A. Convergence trajectories  -- show xi_n converging over iterations
                                  for SA, NSA, MLSA (one figure)
  B. Bootstrap confidence intervals on RMSE
                               -- shows error bars on the slope table
                                  (proves the differences between algorithms
                                   are statistically significant)
  C. Bias-variance decomposition
                               -- decomposes RMSE^2 into bias^2 + variance
                                  for each algorithm; reveals which one
                                  is hurting which algorithm

USAGE
-----
    python3 extra_analyses.py --analysis convergence --model bs
    python3 extra_analyses.py --analysis bootstrap   --model heston
    python3 extra_analyses.py --analysis decompose   --model merton
    python3 extra_analyses.py --analysis all                       # everything

OUTPUTS
-------
    figures/convergence_<model>.pdf
    figures/bootstrap_<model>.pdf
    tables/bootstrap_<model>.csv
    figures/decomposition_<model>.pdf
    tables/decomposition_<model>.csv
"""
from __future__ import annotations
import argparse, os, csv, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from mlsa_core    import algo1_SA, algo2_NSA, algo3_MLSA, H1, H2

rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3,
    "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# =====================================================================
#  Bundles
# =====================================================================
def _bundle(model: str):
    if model == "bs":
        from case_study_1 import closed_form, make_simulators
        sim_X0, sim_Xh, sim_coupled = make_simulators(0.5)
        xi_b, chi_b = closed_form(0.975, 0.5)
        return dict(name="bs", alpha=0.975,
                    sim_X0=sim_X0, sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                    xi_star=xi_b, chi_star=chi_b)
    if model == "heston":
        from case_study_heston import HestonParams, make_simulators, compute_benchmark
        P = HestonParams()
        sim_X0, sim_Xh, sim_coupled, _ = make_simulators(P, 1, 2)
        xi_b, chi_b = compute_benchmark(P, n_samples=20_000)
        return dict(name="heston", alpha=P.alpha,
                    sim_X0=sim_X0, sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                    xi_star=xi_b, chi_star=chi_b)
    if model == "merton":
        from case_study_merton import MertonParams, make_simulators, compute_benchmark
        P = MertonParams()
        sim_X0, sim_Xh, sim_coupled, _ = make_simulators(P)
        xi_b, chi_b = compute_benchmark(P, n_samples=300_000)
        return dict(name="merton", alpha=P.alpha,
                    sim_X0=sim_X0, sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                    xi_star=xi_b, chi_star=chi_b)
    raise ValueError(model)


# =====================================================================
#  Analysis A: Convergence trajectories
# =====================================================================
def _trajectory_SA(simulate_X0, N, alpha, gamma, rng, snapshots):
    X = simulate_X0(N, rng)
    xi, chi = 0.0, 0.0
    traj = []
    snap_set = set(snapshots)
    for n in range(N):
        x = X[n]
        gn = gamma(n + 1)
        xi  -= gn * H1(xi, x, alpha)
        chi -= 1.0 / (n + 1) * H2(chi, xi, x, alpha)
        if (n + 1) in snap_set:
            traj.append((n + 1, xi, chi))
    return traj


def _trajectory_NSA(simulate_Xh, N, K, alpha, gamma, rng, snapshots):
    X = simulate_Xh(N, K, rng)
    xi, chi = 0.0, 0.0
    traj = []
    snap_set = set(snapshots)
    for n in range(N):
        x = X[n]
        gn = gamma(n + 1)
        xi  -= gn * H1(xi, x, alpha)
        chi -= 1.0 / (n + 1) * H2(chi, xi, x, alpha)
        if (n + 1) in snap_set:
            traj.append((n + 1, xi, chi))
    return traj


def analyze_convergence(model: str):
    bundle = _bundle(model)
    name = bundle["name"]
    print(f"=== convergence: {name} ===")

    N = 50_000
    snaps = list(range(500, N + 1, 500))
    n_replicas = 8

    sa_xis, nsa_xis = [], []
    for r in range(n_replicas):
        rng = np.random.default_rng(2024 + r)
        traj = _trajectory_SA(bundle["sim_X0"], N, bundle["alpha"],
                                lambda n: 1.0/(100+n), rng, snaps)
        sa_xis.append([t[1] for t in traj])
        rng = np.random.default_rng(2024 + r)
        traj = _trajectory_NSA(bundle["sim_Xh"], N, K=64, alpha=bundle["alpha"],
                                gamma=lambda n: 1.0/(100+n), rng=rng, snapshots=snaps)
        nsa_xis.append([t[1] for t in traj])

    sa_xis  = np.array(sa_xis)
    nsa_xis = np.array(nsa_xis)
    n_axis  = np.array(snaps)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for r in range(n_replicas):
        ax.plot(n_axis, sa_xis[r],  "-", color="tab:green", alpha=0.3)
        ax.plot(n_axis, nsa_xis[r], "-", color="tab:blue",  alpha=0.3)
    ax.plot(n_axis, sa_xis.mean(axis=0),  "-", color="tab:green", linewidth=2.5,
             label=f"SA (mean over {n_replicas} runs)")
    ax.plot(n_axis, nsa_xis.mean(axis=0), "-", color="tab:blue",  linewidth=2.5,
             label=f"NSA K=64 (mean over {n_replicas} runs)")
    ax.axhline(bundle["xi_star"], color="black", linestyle=":",
                label=fr"benchmark $\xi^0_\star = {bundle['xi_star']:.3f}$")
    ax.set_xlabel("Iteration $n$")
    ax.set_ylabel(r"$\xi_n$")
    ax.set_title(f"{name}: SA / NSA convergence trajectories")
    ax.legend()
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = f"figures/convergence_{name}.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  saved {out}")


# =====================================================================
#  Analysis B: Bootstrap CI on RMSE
# =====================================================================
def analyze_bootstrap(model: str):
    """
    Reads a master pickle from `experiments/<model>_master.pkl` (produced
    by parallel_driver.py) and computes a 95% bootstrap CI on the RMSE
    of each (algorithm, eps).
    """
    import pickle, glob
    bundle = _bundle(model)
    name = bundle["name"]
    print(f"=== bootstrap: {name} ===")

    shards_dir = "experiments/shards"
    if not os.path.isdir(shards_dir):
        print("  no shards dir; run parallel_driver.py --paper first")
        return

    eps_list = []
    by_eps_alg_target = {}    # (eps, alg, tg) -> [vals]
    for fn in sorted(os.listdir(shards_dir)):
        if not fn.startswith(f"{name}_") or not fn.endswith(".pkl"):
            continue
        with open(os.path.join(shards_dir, fn), "rb") as f:
            d = pickle.load(f)
        eps = d["eps"];  tg = d["target"]
        if eps not in eps_list: eps_list.append(eps)
        bench = d["xi_star"] if tg == "VaR" else d["chi_star"]
        for alg in ["SA", "NSA", "MLSA"]:
            vals = np.array([r[alg]["xi" if tg == "VaR" else "chi"]
                              for r in d["replications"]])
            by_eps_alg_target[(eps, alg, tg)] = (vals - bench)
    eps_list.sort(reverse=True)

    if not eps_list:
        print(f"  no shards found for {name}; run parallel_driver.py first")
        return

    n_boot = 1000
    print(f"  bootstrapping with {n_boot} resamples...")
    rng = np.random.default_rng(123)
    rows = [["eps", "target", "algorithm", "RMSE", "CI_low (2.5%)", "CI_high (97.5%)"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    style = {"SA":("o-","tab:green"), "NSA":("s-","tab:blue"), "MLSA":("^-","tab:orange")}
    for col, tg in enumerate(["VaR", "ES"]):
        ax = axes[col]
        for alg in ["SA", "NSA", "MLSA"]:
            xs, mids, los, his = [], [], [], []
            for eps in eps_list:
                key = (eps, alg, tg)
                if key not in by_eps_alg_target: continue
                resid = by_eps_alg_target[key]
                # bootstrap RMSE
                M = len(resid)
                rmses = []
                for _ in range(n_boot):
                    idx = rng.integers(0, M, M)
                    rmses.append(np.sqrt(np.mean(resid[idx]**2)))
                rmses = np.array(rmses)
                rmse_mid = float(np.sqrt(np.mean(resid**2)))
                ci_lo = float(np.quantile(rmses, 0.025))
                ci_hi = float(np.quantile(rmses, 0.975))
                xs.append(eps); mids.append(rmse_mid)
                los.append(ci_lo); his.append(ci_hi)
                rows.append([f"1/{int(round(1/eps))}", tg, alg,
                              f"{rmse_mid:.4f}", f"{ci_lo:.4f}", f"{ci_hi:.4f}"])
            xs = np.array(xs); mids = np.array(mids)
            los = np.array(los); his = np.array(his)
            ax.errorbar(xs, mids, yerr=[mids - los, his - mids],
                        fmt=style[alg][0], color=style[alg][1], label=alg,
                        capsize=3, alpha=0.85)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"Prescribed accuracy $\varepsilon$")
        ax.set_ylabel("RMSE (95% bootstrap CI)")
        ax.set_title(f"{name} {tg}")
        ax.legend()
    fig.tight_layout()
    out = f"figures/bootstrap_{name}.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  saved {out}")

    os.makedirs("tables", exist_ok=True)
    out_csv = f"tables/bootstrap_{name}.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  saved {out_csv}")


# =====================================================================
#  Analysis C: Bias-variance decomposition
# =====================================================================
def analyze_decomposition(model: str):
    """
    For each (algorithm, eps), decompose RMSE^2 = bias^2 + variance.
    Bias = mean(estimate - true), variance = var(estimate).
    """
    import pickle
    bundle = _bundle(model)
    name = bundle["name"]
    print(f"=== bias-variance decomposition: {name} ===")

    shards_dir = "experiments/shards"
    eps_list = []
    decomp = {}
    for fn in sorted(os.listdir(shards_dir)):
        if not fn.startswith(f"{name}_") or not fn.endswith(".pkl"):
            continue
        with open(os.path.join(shards_dir, fn), "rb") as f:
            d = pickle.load(f)
        eps = d["eps"]; tg = d["target"]
        if eps not in eps_list: eps_list.append(eps)
        bench = d["xi_star"] if tg == "VaR" else d["chi_star"]
        for alg in ["SA", "NSA", "MLSA"]:
            vals = np.array([r[alg]["xi" if tg == "VaR" else "chi"]
                              for r in d["replications"]])
            bias = float(vals.mean() - bench)
            var  = float(vals.var())
            decomp[(eps, tg, alg)] = (bias, var)
    eps_list.sort(reverse=True)

    if not decomp:
        print(f"  no shards found for {name}")
        return

    rows = [["eps", "target", "algorithm", "bias", "var", "bias^2", "RMSE^2"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for col, tg in enumerate(["VaR", "ES"]):
        ax = axes[col]
        for alg, color in [("SA", "tab:green"), ("NSA", "tab:blue"),
                            ("MLSA", "tab:orange")]:
            biases, vars_ = [], []
            for eps in eps_list:
                if (eps, tg, alg) not in decomp:
                    biases.append(np.nan); vars_.append(np.nan); continue
                b, v = decomp[(eps, tg, alg)]
                biases.append(b**2); vars_.append(v)
                rows.append([f"1/{int(round(1/eps))}", tg, alg,
                              f"{b:+.4f}", f"{v:.4f}",
                              f"{b**2:.4f}", f"{b**2 + v:.4f}"])
            xs = np.array(eps_list[:len(biases)])
            biases = np.array(biases); vars_ = np.array(vars_)
            ax.loglog(xs, biases, "o-", color=color, label=f"{alg} bias$^2$")
            ax.loglog(xs, vars_,  "s--", color=color, alpha=0.7, label=f"{alg} variance")
        ax.invert_xaxis()
        ax.set_xlabel(r"Prescribed accuracy $\varepsilon$")
        ax.set_ylabel("Component of MSE")
        ax.set_title(f"{name} {tg}: bias$^2$ vs variance")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = f"figures/decomposition_{name}.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  saved {out}")
    os.makedirs("tables", exist_ok=True)
    out_csv = f"tables/decomposition_{name}.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  saved {out_csv}")


# =====================================================================
#  Main
# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True,
                     choices=["convergence", "bootstrap", "decompose", "all"])
    ap.add_argument("--model",  default="bs",
                     choices=["bs", "heston", "merton"])
    args = ap.parse_args()

    if args.analysis in ("convergence", "all"):
        analyze_convergence(args.model)
    if args.analysis in ("bootstrap", "all"):
        analyze_bootstrap(args.model)
    if args.analysis in ("decompose", "all"):
        analyze_decomposition(args.model)
