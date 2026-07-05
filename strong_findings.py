"""
strong_findings.py
====================
Five new analyses on the existing 18,000-run experiment matrix.
None of these need new experiments; they re-extract value from data
already in experiments/shards/*.pkl and experiments/*_master.pkl.

Each analysis answers a question the paper does NOT answer:

  1. Pareto frontier (time vs RMSE)
       Q: "In a fixed compute budget B, which algorithm is best?"
       Output: figures/pareto_<model>.pdf, tables/pareto_<model>.csv

  2. Cross-model crossover threshold
       Q: "Beyond what eps does MLSA beat NSA in each model?"
       Output: tables/crossover_thresholds.csv, figures/crossover.pdf

  3. Per-model verdict on MLSA
       Q: "How much does MLSA actually save in each (model, target)?"
       Output: tables/mlsa_verdict.csv

  4. Runtime tail risk
       Q: "What's the 95th-percentile runtime of each algorithm?"
       Output: tables/runtime_p95.csv, figures/tail_risk.pdf

  5. Cost-equivalent accuracy comparison
       Q: "Given fixed wall-time T, which algorithm has the smallest RMSE?"
       Output: figures/iso_budget.pdf, tables/iso_budget.csv

USAGE
-----
    python3 strong_findings.py                          # all five analyses
    python3 strong_findings.py --analysis pareto        # one only
    python3 strong_findings.py --models bs heston       # subset of models
"""
from __future__ import annotations
import argparse, csv, os, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3,
    "lines.linewidth": 1.5, "lines.markersize": 6,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

EPS_LABELS = {
    1/32: "1/32",  1/64: "1/64", 1/128: "1/128",
    1/256: "1/256", 1/512: "1/512",
}
ALGS = ["SA", "NSA", "MLSA"]
ALG_COLORS = {"SA": "tab:green", "NSA": "tab:blue", "MLSA": "tab:orange"}
ALG_MARKERS = {"SA": "o", "NSA": "s", "MLSA": "^"}


# =====================================================================
#  Helpers — load all shards into a tidy dict
# =====================================================================
def load_all_shards(model: str, shards_dir: str = "experiments/shards") -> list[dict]:
    """Returns a list of all shard pickles for a model, sorted by eps then target."""
    rows = []
    for fn in sorted(os.listdir(shards_dir)):
        if not fn.startswith(f"{model}_") or not fn.endswith(".pkl"):
            continue
        path = os.path.join(shards_dir, fn)
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            rows.append(d)
        except Exception as e:
            print(f"  warn: failed to read {fn}: {e}")
    return rows


def per_replication_stats(shard: dict) -> dict:
    """For one shard, returns {alg: {'errs': array, 'times': array}}."""
    bench = shard["xi_star"] if shard["target"] == "VaR" else shard["chi_star"]
    key = "xi" if shard["target"] == "VaR" else "chi"
    out = {}
    for alg in ALGS:
        vals  = np.array([r[alg][key]   for r in shard["replications"]])
        times = np.array([r[alg]["time"] for r in shard["replications"]])
        out[alg] = {"errs": vals - bench, "times": times}
    return out


# =====================================================================
#  Analysis 1 — Pareto frontier (RMSE vs mean time)
# =====================================================================
def analyze_pareto(model: str) -> None:
    print(f"\n=== Pareto frontier: {model} ===")
    shards = load_all_shards(model)
    if not shards:
        print(f"  no shards for {model}, skipping"); return

    # Build (alg, eps, target) -> (mean_time, rmse) records
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    rows = [["target", "algorithm", "eps", "mean_time", "RMSE"]]
    for col, target in enumerate(["VaR", "ES"]):
        ax = axes[col]
        for alg in ALGS:
            xs, ys, eps_lbl = [], [], []
            for s in shards:
                if s["target"] != target: continue
                stats = per_replication_stats(s)
                t = stats[alg]["times"].mean()
                r = float(np.sqrt(np.mean(stats[alg]["errs"] ** 2)))
                xs.append(t); ys.append(r); eps_lbl.append(EPS_LABELS[s["eps"]])
                rows.append([target, alg, EPS_LABELS[s["eps"]],
                              f"{t:.4f}", f"{r:.4f}"])
            xs = np.array(xs); ys = np.array(ys)
            order = np.argsort(xs)
            ax.loglog(xs[order], ys[order],
                       ALG_MARKERS[alg] + "-",
                       color=ALG_COLORS[alg], label=alg, alpha=0.85)
            for x, y, e in zip(xs, ys, eps_lbl):
                ax.annotate(e, (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, alpha=0.6)

        # Compute and shade the Pareto frontier
        all_pts = []
        for alg in ALGS:
            for s in shards:
                if s["target"] != target: continue
                stats = per_replication_stats(s)
                t = stats[alg]["times"].mean()
                r = float(np.sqrt(np.mean(stats[alg]["errs"] ** 2)))
                all_pts.append((t, r, alg))
        all_pts.sort()
        pareto = []
        best_r = float("inf")
        for t, r, alg in all_pts:
            if r < best_r:
                pareto.append((t, r, alg))
                best_r = r
        if pareto:
            pt = np.array([(t, r) for t, r, _ in pareto])
            ax.fill_between(pt[:, 0], pt[:, 1], pt[:, 1].max() * 2,
                            color="gray", alpha=0.08, label="dominated")
        ax.set_xlabel("Mean wall time (s, log)")
        ax.set_ylabel("RMSE (log)")
        ax.set_title(f"{model} {target}: Pareto frontier")
        ax.legend(fontsize=9)

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = f"figures/pareto_{model}.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  saved {out}")

    Path("tables").mkdir(exist_ok=True)
    out_csv = f"tables/pareto_{model}.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  saved {out_csv}")

    # Print the dominated/dominates summary
    pareto_algs = [alg for _, _, alg in pareto]
    counter = {alg: pareto_algs.count(alg) for alg in ALGS}
    print(f"  Pareto-optimal points by algorithm: {counter}")


# =====================================================================
#  Analysis 2 — Crossover thresholds: where MLSA starts beating NSA
# =====================================================================
def analyze_crossover(models: list[str]) -> None:
    print(f"\n=== Crossover thresholds (MLSA vs NSA) ===")
    rows = [["model", "target", "crossover_eps_estimate",
             "interp_method", "MLSA_wins_below"]]
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.5 * len(models)),
                              squeeze=False)
    for r_idx, model in enumerate(models):
        shards = load_all_shards(model)
        if not shards:
            print(f"  {model}: no shards"); continue
        for c_idx, target in enumerate(["VaR", "ES"]):
            ax = axes[r_idx, c_idx]
            xs, nsa_t, mlsa_t = [], [], []
            for s in sorted(shards, key=lambda x: -x["eps"]):
                if s["target"] != target: continue
                stats = per_replication_stats(s)
                xs.append(s["eps"])
                nsa_t.append(stats["NSA"]["times"].mean())
                mlsa_t.append(stats["MLSA"]["times"].mean())
            xs = np.array(xs); nsa_t = np.array(nsa_t); mlsa_t = np.array(mlsa_t)
            ratio = mlsa_t / nsa_t   # < 1  means MLSA faster
            ax.semilogx(xs, ratio, "o-", color=ALG_COLORS["MLSA"])
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
            ax.invert_xaxis()
            ax.set_xlabel(r"$\varepsilon$")
            ax.set_ylabel("MLSA / NSA mean time")
            ax.set_title(f"{model} {target}")
            ax.set_yscale("log")

            # Find first eps where MLSA < NSA  (crossover)
            cross_eps = None
            for j in range(len(xs) - 1):
                if (ratio[j] - 1) * (ratio[j+1] - 1) < 0:
                    # log-linear interpolation
                    a = np.log(xs[j]);    b = np.log(xs[j+1])
                    p = np.log(ratio[j]); q = np.log(ratio[j+1])
                    cross_eps = float(np.exp(a - p * (b - a) / (q - p)))
                    break
            if cross_eps is None and (ratio < 1).all():
                cross_eps = xs.max()
            if cross_eps is None and (ratio > 1).all():
                cross_eps = xs.min()
            cross_lbl = f"1/{int(round(1/cross_eps))}" if cross_eps else "n/a"
            rows.append([model, target, cross_lbl, "log-linear",
                          str((ratio < 1).any())])
            if cross_eps:
                ax.axvline(cross_eps, color="red", linestyle="--", alpha=0.7,
                            label=f"crossover ε ≈ {cross_lbl}")
                ax.legend()

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = "figures/crossover.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  saved {out}")

    Path("tables").mkdir(exist_ok=True)
    out_csv = "tables/crossover_thresholds.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  saved {out_csv}")
    for r in rows[1:]:
        print(f"    {r[0]:8s} {r[1]:5s} crossover at ε ≈ {r[2]}")


# =====================================================================
#  Analysis 3 — Per-model MLSA verdict
# =====================================================================
def analyze_verdict(models: list[str]) -> None:
    """For each (model, target), report best MLSA speedup over NSA at the
    smallest ε where both produce comparable RMSE."""
    print(f"\n=== Per-model MLSA verdict ===")
    rows = [["model", "target", "smallest_eps",
             "MLSA_time", "NSA_time", "speedup",
             "MLSA_RMSE", "NSA_RMSE", "verdict"]]
    for model in models:
        shards = load_all_shards(model)
        if not shards:
            continue
        for target in ["VaR", "ES"]:
            sels = sorted([s for s in shards if s["target"] == target],
                          key=lambda x: x["eps"])
            if not sels: continue
            # use the smallest eps  (largest workload, most informative)
            s = sels[0]
            stats = per_replication_stats(s)
            mlsa_t = stats["MLSA"]["times"].mean()
            nsa_t  = stats["NSA"]["times"].mean()
            mlsa_r = float(np.sqrt(np.mean(stats["MLSA"]["errs"] ** 2)))
            nsa_r  = float(np.sqrt(np.mean(stats["NSA"]["errs"] ** 2)))
            speedup = nsa_t / mlsa_t
            if speedup > 2 and mlsa_r < nsa_r * 2:
                verdict = "USE MLSA (>=2x speedup)"
            elif speedup > 1.2:
                verdict = "MLSA slightly faster"
            elif speedup > 0.8:
                verdict = "ROUGHLY EQUAL"
            else:
                verdict = "USE NSA (MLSA slower)"
            rows.append([model, target, EPS_LABELS[s["eps"]],
                          f"{mlsa_t:.3f}", f"{nsa_t:.3f}", f"{speedup:.2f}x",
                          f"{mlsa_r:.4f}", f"{nsa_r:.4f}", verdict])
            print(f"  {model:8s} {target:3s}  speedup={speedup:5.2f}x  "
                  f"MLSA_RMSE={mlsa_r:.4f}  NSA_RMSE={nsa_r:.4f}  -> {verdict}")
    Path("tables").mkdir(exist_ok=True)
    out_csv = "tables/mlsa_verdict.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  saved {out_csv}")


# =====================================================================
#  Analysis 4 — Runtime tail risk (P50 / P95 / P99)
# =====================================================================
def analyze_tail_risk(models: list[str]) -> None:
    print(f"\n=== Runtime tail risk ===")
    rows = [["model", "target", "eps", "algorithm",
             "mean", "p50", "p95", "p99", "max", "p95_over_mean"]]
    fig, axes = plt.subplots(len(models), 2,
                              figsize=(11, 3.5 * len(models)), squeeze=False)
    for r_idx, model in enumerate(models):
        shards = load_all_shards(model)
        if not shards:
            continue
        for c_idx, target in enumerate(["VaR", "ES"]):
            ax = axes[r_idx, c_idx]
            sels = sorted([s for s in shards if s["target"] == target],
                          key=lambda x: -x["eps"])
            for alg in ALGS:
                xs, p50s, p95s, p99s = [], [], [], []
                for s in sels:
                    stats = per_replication_stats(s)
                    t = stats[alg]["times"]
                    xs.append(s["eps"])
                    p50s.append(np.percentile(t, 50))
                    p95s.append(np.percentile(t, 95))
                    p99s.append(np.percentile(t, 99))
                    rows.append([model, target, EPS_LABELS[s["eps"]], alg,
                                  f"{t.mean():.4f}",
                                  f"{p50s[-1]:.4f}",
                                  f"{p95s[-1]:.4f}",
                                  f"{p99s[-1]:.4f}",
                                  f"{t.max():.4f}",
                                  f"{p95s[-1]/t.mean():.2f}"])
                xs = np.array(xs)
                p50s = np.array(p50s); p95s = np.array(p95s)
                ax.loglog(xs, p50s, "-",  color=ALG_COLORS[alg],
                            alpha=0.4, label=f"{alg} P50")
                ax.loglog(xs, p95s, "--", color=ALG_COLORS[alg],
                            alpha=1.0, label=f"{alg} P95")
            ax.invert_xaxis()
            ax.set_xlabel(r"$\varepsilon$")
            ax.set_ylabel("Wall time (s)")
            ax.set_title(f"{model} {target}")
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = "figures/tail_risk.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  saved {out}")
    Path("tables").mkdir(exist_ok=True)
    out_csv = "tables/runtime_p95.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  saved {out_csv}")
    # Print headline finding
    print("  Highlights (P95 / mean ratio, larger means heavier tail):")
    for model in models:
        for target in ["VaR", "ES"]:
            shards = load_all_shards(model)
            for s in sorted(shards, key=lambda x: -x["eps"])[:1]:
                if s["target"] != target: continue
                for alg in ALGS:
                    stats = per_replication_stats(s)
                    t = stats[alg]["times"]
                    print(f"    {model:8s} {target:3s}  {alg:5s}  "
                          f"P95/mean = {np.percentile(t,95)/t.mean():.2f}")
                break


# =====================================================================
#  Analysis 5 — Cost-equivalent accuracy
# =====================================================================
def analyze_iso_budget(models: list[str]) -> None:
    """For each fixed wall-time T, find which (alg, eps) gives smallest RMSE."""
    print(f"\n=== Iso-budget RMSE comparison ===")
    fig, axes = plt.subplots(len(models), 2,
                              figsize=(11, 3.5 * len(models)), squeeze=False)
    rows = [["model", "target", "budget_seconds",
             "best_alg", "best_eps", "best_RMSE"]]
    for r_idx, model in enumerate(models):
        shards = load_all_shards(model)
        if not shards: continue
        for c_idx, target in enumerate(["VaR", "ES"]):
            ax = axes[r_idx, c_idx]
            # Collect all (alg, eps, time, rmse) records
            data = []
            for s in shards:
                if s["target"] != target: continue
                stats = per_replication_stats(s)
                for alg in ALGS:
                    t = stats[alg]["times"].mean()
                    r = float(np.sqrt(np.mean(stats[alg]["errs"] ** 2)))
                    data.append((alg, s["eps"], t, r))
            if not data: continue

            # For a grid of budgets, find best RMSE achievable per algorithm
            min_t = min(d[2] for d in data)
            max_t = max(d[2] for d in data)
            T_grid = np.logspace(np.log10(min_t * 0.9),
                                  np.log10(max_t * 1.1), 50)

            for alg in ALGS:
                alg_data = [(t, r, eps) for (a, eps, t, r) in data if a == alg]
                alg_data.sort()  # by time
                best_rmse_at_T = []
                for T in T_grid:
                    candidates = [(r, eps) for (t, r, eps) in alg_data if t <= T]
                    best_rmse_at_T.append(min(c[0] for c in candidates) if candidates else np.nan)
                ax.loglog(T_grid, best_rmse_at_T,
                          color=ALG_COLORS[alg], label=alg, linewidth=2)

            # Find global best at a few key budgets
            for budget_T in [1.0, 10.0, 60.0]:
                feas = [(r, alg, eps) for (alg, eps, t, r) in data if t <= budget_T]
                if not feas: continue
                feas.sort()
                best_r, best_alg, best_eps = feas[0]
                rows.append([model, target, f"{budget_T:.0f}s",
                              best_alg, EPS_LABELS[best_eps],
                              f"{best_r:.4f}"])
                ax.axvline(budget_T, color="gray", alpha=0.3, linewidth=0.7)
                ax.text(budget_T, ax.get_ylim()[1] * 0.7,
                        f"T={budget_T:g}s\nbest={best_alg}",
                        fontsize=7, ha="center", alpha=0.6)

            ax.set_xlabel("Wall-time budget (s, log)")
            ax.set_ylabel("Best achievable RMSE")
            ax.set_title(f"{model} {target}")
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = "figures/iso_budget.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  saved {out}")

    Path("tables").mkdir(exist_ok=True)
    out_csv = "tables/iso_budget.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  saved {out_csv}")


# =====================================================================
#  Main
# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="all",
                     choices=["all", "pareto", "crossover", "verdict",
                              "tail_risk", "iso_budget"])
    ap.add_argument("--models", nargs="+",
                     default=["bs", "merton", "heston"])
    args = ap.parse_args()

    if args.analysis in ("pareto", "all"):
        for m in args.models:
            analyze_pareto(m)
    if args.analysis in ("crossover", "all"):
        analyze_crossover(args.models)
    if args.analysis in ("verdict", "all"):
        analyze_verdict(args.models)
    if args.analysis in ("tail_risk", "all"):
        analyze_tail_risk(args.models)
    if args.analysis in ("iso_budget", "all"):
        analyze_iso_budget(args.models)

    print("\n=== ALL DONE ===")
