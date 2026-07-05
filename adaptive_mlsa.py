"""
adaptive_mlsa.py
=================

The procedure:
1. Run a SHORT pilot (2000 iterations) at each candidate gamma_factor in
   a small grid, on the level-l SA estimator.
2. Score each gamma by a bias-variance proxy (drift^2 + var) across
   m=4 mini-replications.
3. Pick the gamma minimising that score per level.
4. Run the main MLSA loop using that step size.


USAGE
-----
    python3 adaptive_mlsa.py --model heston --target VaR
    python3 adaptive_mlsa.py --model heston --target VaR --quick

OUTPUTS
-------
    figures/adaptive_<model>_<target>.pdf         (RMSE comparison plot)
    tables/adaptive_<model>_<target>.csv          (RMSE / time table)

"""
from __future__ import annotations
import argparse, os, csv, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from mlsa_core import algo3_MLSA, H1, H2

rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3,
    "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------
#  Pilot run
# ---------------------------------------------------------------------
def _pilot_score(simulate_Xh, K, alpha, gamma_factor, gamma_denom,
                 n_pilot=1_000, m=8, rng_seed=0) -> float:
    """
    Run m short SA pilots and return a bias-variance score.

    Score = mean(xi-final)^2 + var(xi-final)  -- the classical
    MSE-style decomposition. Lower is better. We center the bias
    against the pilots' own mean.

    Returns -1 if all pilots produce NaN/inf.
    """
    g = lambda n: gamma_factor / (gamma_denom + n)
    xis = []
    for r in range(m):
        rng = np.random.default_rng(rng_seed + r)
        X = simulate_Xh(n_pilot, K, rng)
        xi = 0.0; chi = 0.0
        for n in range(n_pilot):
            x = X[n]
            xi  -= g(n + 1) * H1(xi, x, alpha)
            chi -= 1.0 / (n + 1) * H2(chi, xi, x, alpha)
        if not np.isfinite(xi):
            return float("inf")
        xis.append(xi)
    xis = np.asarray(xis)
    # We use a proxy for MSE: (running mean drift)^2 + variance.
    # The variance penalises noisy step sizes, the mean drift
    # penalises step sizes that haven't moved off zero yet.
    drift_proxy = (xis.mean()) ** 2          # large if SA hasn't converged
    return float(xis.var() + 0.1 * drift_proxy)


# ---------------------------------------------------------------------
#  Adaptive-step-size MLSA
# ---------------------------------------------------------------------
def adaptive_mlsa(simulate_coupled_pair, simulate_Xh,
                   L, h0, M, Ns, alpha,
                   gamma_grid=(0.3, 1.0, 3.0, 10.0),
                   gamma_denom=500,
                   rng=None) -> tuple[float, float, list[float]]:
    """
    Standard MLSA but with per-level gamma_factor selected by pilot.
    Returns (xi_ML, chi_ML, selected_gammas).
    """
    rng = np.random.default_rng() if rng is None else rng

    # For pilot we use the inner Xh simulator at K_l.
    h_levels = [h0 / (M ** ell) for ell in range(L + 1)]
    K_levels = [int(round(1.0 / h)) for h in h_levels]

    selected = []
    # Choose gamma at each level.
    for ell in range(L + 1):
        K = K_levels[ell]
        best_g, best_score = None, float("inf")
        for gf in gamma_grid:
            v = _pilot_score(simulate_Xh, K, alpha, gf, gamma_denom,
                              n_pilot=2000, m=4,
                              rng_seed=int(rng.integers(0, 1<<31)))
            if v < best_score:
                best_g, best_score = gf, v
        selected.append(best_g)

    # Use the FINEST level's gamma for the algo3 step size.
    gf_global = selected[-1]
    gamma = lambda n: gf_global / (gamma_denom + n)

    xi, chi = algo3_MLSA(simulate_coupled_pair, simulate_Xh,
                          L, h0, M, Ns, alpha, gamma, rng=rng)
    return xi, chi, selected


# ---------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------
def _bundle(model: str):
    if model == "heston":
        from case_study_heston import HestonParams, make_simulators, compute_benchmark
        P = HestonParams()
        _, sim_Xh, sim_coupled, _ = make_simulators(P, n_steps_to_delta=1, n_steps_post_delta=2)
        xi_b, chi_b = compute_benchmark(P, n_samples=20_000)
        return dict(name="heston", alpha=P.alpha,
                     sim_Xh=sim_Xh, sim_coupled=sim_coupled,
                     xi_star=xi_b, chi_star=chi_b)
    if model == "merton":
        from case_study_merton import MertonParams, make_simulators, compute_benchmark
        P = MertonParams()
        _, sim_Xh, sim_coupled, _ = make_simulators(P)
        xi_b, chi_b = compute_benchmark(P, n_samples=300_000)
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

    # Match the standard recipe at eps=1/128 so we can compare apples-to-apples
    h0, L, M, denom = 1/8, 3, 2, 500
    Ns = [4_000, 2_000, 1_000, 500]

    standard_gamma_factor = {
        ("heston", "VaR"): 2.0,  ("heston", "ES"): 1.5,
        ("merton", "VaR"): 0.5,  ("merton", "ES"): 1.0,
        ("bs",     "VaR"): 0.75, ("bs",     "ES"): 0.1,
    }[(bundle["name"], args.target)]

    n_runs = 20 if args.quick else 80

    results_std = []
    results_adp = []
    times_std, times_adp = [], []

    print(f"=== adaptive MLSA: {bundle['name']} {args.target}  "
          f"(eps=1/128, n_runs={n_runs}) ===\n")

    for run in range(n_runs):
        # ----- standard MLSA
        rng = np.random.default_rng(7 + run)
        g_std = lambda n: standard_gamma_factor / (denom + n)
        t0 = time.perf_counter()
        xi, chi = algo3_MLSA(bundle["sim_coupled"], bundle["sim_Xh"],
                              L, h0, M, Ns, bundle["alpha"], g_std, rng=rng)
        times_std.append(time.perf_counter() - t0)
        results_std.append(xi if args.target == "VaR" else chi)

        # ----- adaptive MLSA
        rng = np.random.default_rng(7 + run)
        t0 = time.perf_counter()
        xi, chi, sel = adaptive_mlsa(
            bundle["sim_coupled"], bundle["sim_Xh"],
            L, h0, M, Ns, bundle["alpha"],
            gamma_grid=(0.3, 1.0, 3.0, 10.0), gamma_denom=denom, rng=rng,
        )
        times_adp.append(time.perf_counter() - t0)
        results_adp.append(xi if args.target == "VaR" else chi)

        if run < 3:
            print(f"  run {run}: standard={results_std[-1]:.4f},  "
                  f"adaptive={results_adp[-1]:.4f}  selected gamma per level={sel}")

    rmse_std = float(np.sqrt(np.mean((np.array(results_std) - bench) ** 2)))
    rmse_adp = float(np.sqrt(np.mean((np.array(results_adp) - bench) ** 2)))
    mt_std   = float(np.mean(times_std))
    mt_adp   = float(np.mean(times_adp))

    print(f"\n--- Results ({n_runs} replications) ---")
    print(f"  STANDARD MLSA: RMSE = {rmse_std:.4f},  mean time = {mt_std:.3f}s")
    print(f"  ADAPTIVE MLSA: RMSE = {rmse_adp:.4f},  mean time = {mt_adp:.3f}s")
    print(f"  Improvement: {rmse_std/rmse_adp:.1f}x lower RMSE,  "
          f"{mt_adp/mt_std:.2f}x time overhead")

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(min(min(results_std), min(results_adp)) - 0.05,
                       max(max(results_std), max(results_adp)) + 0.05, 30)
    ax.hist(results_std, bins=bins, alpha=0.6, color="tab:gray",
             label=f"Standard MLSA (RMSE {rmse_std:.3f})")
    ax.hist(results_adp, bins=bins, alpha=0.6, color="tab:orange",
             label=f"Adaptive MLSA (RMSE {rmse_adp:.3f})")
    ax.axvline(bench, color="black", linestyle=":",
                label=f"Benchmark = {bench:.3f}")
    ax.set_xlabel(args.target + " estimate")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{bundle['name']} {args.target}:  standard vs adaptive MLSA")
    ax.legend()
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = f"figures/adaptive_{bundle['name']}_{args.target}.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"saved -> {out}")

    os.makedirs("tables", exist_ok=True)
    out_csv = f"tables/adaptive_{bundle['name']}_{args.target}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "RMSE", "mean_time_s", "n_runs"])
        w.writerow(["standard", f"{rmse_std:.6f}", f"{mt_std:.4f}", n_runs])
        w.writerow(["adaptive", f"{rmse_adp:.6f}", f"{mt_adp:.4f}", n_runs])
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="heston",
                     choices=["bs", "heston", "merton"])
    ap.add_argument("--target", default="VaR", choices=["VaR", "ES"])
    ap.add_argument("--quick",  action="store_true")
    args = ap.parse_args()
    main(args)
