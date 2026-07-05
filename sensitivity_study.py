"""
sensitivity_study.py
=====================
Step-size sensitivity study for MLSA -- ANSWERS THE QUESTION:

    "WHY does the Heston VaR MLSA slope deviate from theory?"

We fix the Heston model and the target accuracy eps = 1/128, then sweep
the step-size lead constant gamma_factor over a logarithmic grid. For
each gamma_factor we run n_runs=80 replications of MLSA and record:
  - empirical RMSE (vs benchmark)
  - mean wall time

Output: a plot showing how RMSE varies with gamma_factor, plus a CSV
table for inclusion in the thesis.

USAGE
-----
    python3 sensitivity_study.py            # default: Heston VaR
    python3 sensitivity_study.py --target ES
    python3 sensitivity_study.py --model merton
    python3 sensitivity_study.py --quick    # n_runs=20

OUTPUTS
-------
    figures/sensitivity_<model>_<target>.pdf
    tables/sensitivity_<model>_<target>.csv
"""
from __future__ import annotations
import argparse, os, csv, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from mlsa_core import algo3_MLSA

rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3,
    "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def _bundle(model: str):
    if model == "heston":
        from case_study_heston import HestonParams, make_simulators
        P = HestonParams()
        _, sim_Xh, sim_coupled, _ = make_simulators(
            P, n_steps_to_delta=1, n_steps_post_delta=2
        )
        # benchmark (small but enough for a single fixed eps)
        from case_study_heston import compute_benchmark
        xi_b, chi_b = compute_benchmark(P, n_samples=20_000)
        return dict(name="heston", alpha=P.alpha,
                     sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                     xi_star=xi_b, chi_star=chi_b)

    if model == "merton":
        from case_study_merton import MertonParams, make_simulators, compute_benchmark
        P = MertonParams()
        _, sim_Xh, sim_coupled, _ = make_simulators(P)
        xi_b, chi_b = compute_benchmark(P, n_samples=500_000)
        return dict(name="merton", alpha=P.alpha,
                     sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                     xi_star=xi_b, chi_star=chi_b)

    if model == "bs":
        from case_study_1 import closed_form, make_simulators
        _, sim_Xh, sim_coupled = make_simulators(0.5)
        xi_b, chi_b = closed_form(0.975, 0.5)
        return dict(name="bs", alpha=0.975,
                     sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                     xi_star=xi_b, chi_star=chi_b)
    raise ValueError(model)


def main(args):
    bundle = _bundle(args.model)
    bench  = bundle["xi_star"] if args.target == "VaR" else bundle["chi_star"]

    eps   = 1 / 128
    h0, L, M, denom = 1/8, 3, 2, 500
    Ns    = [4_000, 2_000, 1_000, 500]

    factor_grid = np.logspace(-1.5, 1.5, 11)    # 0.03 to 31.6
    n_runs      = 20 if args.quick else 80

    rmses, times = [], []
    print(f"=== sensitivity study: {bundle['name']} {args.target} "
          f"(eps=1/128, n_runs={n_runs}) ===")
    for gf in factor_grid:
        g = lambda n, _gf=gf: _gf / (denom + n)
        vals, ts = [], []
        for run in range(n_runs):
            rng = np.random.default_rng(2024 + run)
            t0 = time.perf_counter()
            xi, chi = algo3_MLSA(bundle["sim_coupled"], bundle["sim_Xh"],
                                  L, h0, M, Ns, bundle["alpha"], g, rng=rng)
            ts.append(time.perf_counter() - t0)
            vals.append(xi if args.target == "VaR" else chi)
        rmse = float(np.sqrt(np.mean((np.array(vals) - bench) ** 2)))
        mt   = float(np.mean(ts))
        rmses.append(rmse); times.append(mt)
        print(f"  gamma_factor={gf:7.3f}   RMSE={rmse:.4f}   mean time={mt:.3f}s")

    # ----- plot
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.loglog(factor_grid, rmses, "o-", color="tab:red", label="RMSE")
    ax1.set_xlabel(r"Step-size lead constant $\gamma_0$")
    ax1.set_ylabel("RMSE", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title(f"{bundle['name']} {args.target}: "
                  f"MLSA sensitivity to $\\gamma_0$")
    ax2 = ax1.twinx()
    ax2.loglog(factor_grid, times, "s--", color="tab:blue",
               alpha=0.7, label="mean time")
    ax2.set_ylabel("Mean time (s)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.grid(False)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out_pdf = f"figures/sensitivity_{bundle['name']}_{args.target}.pdf"
    fig.savefig(out_pdf); fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"saved -> {out_pdf}")

    # ----- csv
    os.makedirs("tables", exist_ok=True)
    out_csv = f"tables/sensitivity_{bundle['name']}_{args.target}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gamma_factor", "RMSE", "mean_time"])
        for gf, r, t in zip(factor_grid, rmses, times):
            w.writerow([f"{gf:.4f}", f"{r:.6f}", f"{t:.4f}"])
    print(f"saved -> {out_csv}")

    # ----- key insight: best vs worst RMSE
    best, worst = min(rmses), max(rmses)
    print(f"\nKey finding:  RMSE varies from {best:.4f} "
          f"to {worst:.4f} (factor of {worst/best:.1f}x) "
          f"depending on gamma_0 choice.")
    print(f"This quantifies the step-size sensitivity discussed in the thesis.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  choices=["bs", "heston", "merton"], default="heston")
    ap.add_argument("--target", choices=["VaR", "ES"],              default="VaR")
    ap.add_argument("--quick",  action="store_true")
    args = ap.parse_args()
    main(args)
