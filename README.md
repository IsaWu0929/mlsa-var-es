# Multilevel Stochastic Approximation for VaR & Expected Shortfall

Reproduction **and extension** of the multilevel stochastic-approximation (MLSA)
estimator for Value-at-Risk (VaR) and Expected Shortfall (ES) of
Crépey, Frikha & Louzi (2025), developed for my M.S. Statistics thesis at the
University of Chicago.

Starting from the paper's Black–Scholes benchmark, this project:

1. **Reproduces** the published algorithms and validates every implementation
   against eight independent closed-form / analytic benchmarks.
2. **Generalizes** the estimator from Black–Scholes to the **Heston**
   (stochastic-volatility) and **Merton** (jump-diffusion) models.
3. Builds an **18,000-run parallelized simulation framework** to compare
   algorithms across models, accuracy targets, and replications.
4. Diagnoses a step-size–driven instability in the VaR estimator and adds an
   **adaptive, pilot-based step-size selector** that restores robustness.

## Highlights

- **Correctness** — every estimator is checked against 8 analytic benchmarks
  (e.g. the Black–Scholes closed form ξ⋆⁰ = 2.0119, χ⋆⁰ = 2.9011, matching the
  paper's 2.012 / 2.901).
- **Efficiency** — the Heston regime with a Quadratic-Exponential inner
  Monte-Carlo scheme is the strongest MLSA setting, giving **1.73–1.80×
  wall-clock speedup at matched accuracy** (2.37–2.48× at matched RMSE) over
  nested SA.
- **Robustness** — under step-size misspecification the VaR estimator's RMSE
  inflates by **7.7–10.7×**; the adaptive selector cuts that RMSE by
  **2.3–3.7×** in the affected settings, turning a known failure mode into a
  controlled one.
- **Scale** — 3 models × 5 accuracy targets × 3 algorithms × 2 risk measures ×
  200 replications, run in parallel across all cores with crash-safe
  checkpointing.

## Repository layout

| Path | What it is |
|---|---|
| `mlsa_core.py` | Core algorithms: SA (Alg. 1), nested SA (Alg. 2), multilevel SA (Alg. 3) and the H₁/H₂ gradient operators |
| `case_study_1.py`, `case_study_2.py` | The paper's Black–Scholes / swap benchmarks, with closed-form checks |
| `case_study_heston.py`, `case_study_merton.py` | Extensions to the Heston and Merton models |
| `heston_fourier.py`, `heston_fourier_benchmark.py` | Fourier-transform Heston pricing used as an analytic benchmark |
| `parallel_driver.py` | 18,000-run experiment driver (joblib parallelism, per-shard checkpointing) |
| `adaptive_mlsa.py` | Adaptive pilot-based step-size selector |
| `sensitivity_study.py` | Step-size sensitivity / instability diagnosis |
| `run_case1.py`, `run_case2.py`, `run_heston.py`, `run_merton.py` | One-shot reproduction entry points |
| `bootstrap_ci.py`, `equal_rmse_table.py`, `matched_walltime_adaptive.py`, `runtime_distribution.py`, `strong_findings.py`, `extra_analyses.py` | Bootstrapped CIs, figure/table generation, and supporting analysis |
| `tests_correctness.py`, `smoke_test.py` | Analytic-benchmark checks and a ~1-minute end-to-end smoke test |
| `figures/`, `tables/` | Generated figures and result tables |
| `RESULTS_INTERPRETATION.md`, `REAL_RESULTS_SUMMARY.md` | Write-ups of the empirical findings |
| `scratch/` | Development/iteration scripts (not needed to run the project) |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ~1-minute sanity check
python smoke_test.py

# quick reproductions (minutes)
python run_case1.py --quick
python run_heston.py --quick

# full 18,000-run study (hours; parallel, checkpointed)
python parallel_driver.py --model all --paper
```

Outputs are written to `figures/` and `tables/`.

## Method in one paragraph

VaR and ES are estimated *jointly* by stochastic approximation on the H₁/H₂
gradient operators from the paper; the multilevel scheme couples nested
Monte-Carlo levels to cut the cost of the inner conditional expectation. The
extensions replace the Black–Scholes inner simulator with Heston dynamics
(using a Quadratic-Exponential discretization) and Merton jump-diffusion, and
add a short pilot that scores candidate step sizes by a bias–variance proxy
before committing to the main run.

## Tech stack

Python · NumPy · SciPy · joblib (parallelism) · Matplotlib · tqdm

## Attribution

Algorithms 1–3 and the two base case studies are from:

> S. Crépey, N. Frikha, A. Louzi (2025). *A multilevel stochastic approximation
> algorithm for value-at-risk and expected shortfall estimation.* Finance and
> Stochastics 29: 1015–1074. https://doi.org/10.1007/s00780-025-00573-5

Reference implementation for the base algorithms:
https://github.com/azarlouzi/mlsa. The Heston/Merton generalization, the
parallelized experiment framework, the step-size sensitivity analysis, and the
adaptive step-size selector are my own contributions for this thesis.

## License

MIT — see [`LICENSE`](LICENSE).
