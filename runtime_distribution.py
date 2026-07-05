"""
runtime_distribution.py
========================
Loads the per-replication results from `experiments/<model>_master.pkl`
and plots a boxplot of execution times for SA / NSA / MLSA at each
prescribed accuracy. This addresses the question:

    "OK, MLSA is faster on AVERAGE, but is it RELIABLY faster, or does
    it occasionally take ages to run?"

Boxplots reveal whether MLSA's distribution is fat-tailed or compact.

USAGE
-----
    python3 runtime_distribution.py --model heston --target VaR
    python3 runtime_distribution.py --model bs --target ES

Requires that ``parallel_driver.py`` has been run for the given model.
"""
from __future__ import annotations
import argparse, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

def main(args):
    # We need PER-REPLICATION times, which live in the SHARDS, not the
    # master.  So walk through the shards directly.
    shards_dir = "experiments/shards"
    if not os.path.isdir(shards_dir):
        raise SystemExit("Run parallel_driver.py first to generate shards.")

    all_times = {alg: [] for alg in ["SA", "NSA", "MLSA"]}
    eps_list = []
    for fn in sorted(os.listdir(shards_dir)):
        if not fn.startswith(f"{args.model}_") or not fn.endswith(".pkl"):
            continue
        if not fn.endswith(f"{args.target}.pkl"):
            continue
        with open(os.path.join(shards_dir, fn), "rb") as f:
            d = pickle.load(f)
        eps_list.append(d["eps"])
        for alg in ["SA", "NSA", "MLSA"]:
            ts = [r[alg]["time"] for r in d["replications"]]
            all_times[alg].append(ts)

    # sort by eps decreasing (so plot reads left-to-right from large to small)
    order = np.argsort([-e for e in eps_list])
    eps_list = [eps_list[i] for i in order]
    for alg in all_times:
        all_times[alg] = [all_times[alg][i] for i in order]

    # --- 3 panels, one per algorithm
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    colors = {"SA": "tab:green", "NSA": "tab:blue", "MLSA": "tab:orange"}
    labels = [f"1/{int(round(1/e))}" for e in eps_list]

    for ax, alg in zip(axes, ["SA", "NSA", "MLSA"]):
        bp = ax.boxplot(all_times[alg], labels=labels, patch_artist=True,
                         medianprops=dict(color="black"))
        for b in bp["boxes"]:
            b.set_facecolor(colors[alg]); b.set_alpha(0.6)
        ax.set_yscale("log")
        ax.set_xlabel(r"Prescribed accuracy $\varepsilon$")
        ax.set_title(f"{alg}")
    axes[0].set_ylabel("Execution time (s, log scale)")
    fig.suptitle(f"{args.model.title()} {args.target}: "
                 f"distribution of execution times across 200 replications")
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = f"figures/runtime_dist_{args.model}_{args.target}.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"saved -> {out}")

    # ---- print a summary
    print("\nVariance summary (coefficient of variation = std/mean):")
    print(f"{'eps':>8s}  {'SA cv':>8s}  {'NSA cv':>8s}  {'MLSA cv':>8s}")
    for i, eps in enumerate(eps_list):
        print(f"  1/{int(round(1/eps)):>4d}  ", end="")
        for alg in ["SA", "NSA", "MLSA"]:
            ts = np.array(all_times[alg][i])
            cv = ts.std() / ts.mean() if ts.mean() > 0 else 0
            print(f"{cv:>8.3f}", end="  ")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True,
                     choices=["bs", "heston", "merton"])
    ap.add_argument("--target", default="VaR", choices=["VaR", "ES"])
    args = ap.parse_args()
    main(args)
