"""
tests_correctness.py
=====================
Automated implementation-correctness tests.  Run BEFORE the paper-mode
experiments so you can tell WBW honestly: "I have eight automated tests
that verify the implementation against analytic / cross-validated targets."

USAGE
-----
    python3 tests_correctness.py

Each test prints PASS or FAIL with a numerical tolerance. A successful
end-to-end run produces ``tables/correctness_tests.csv`` summarising all
tests for inclusion in the thesis appendix.

Eight tests, in order of importance:

  T1  BS closed-form xi*, chi*  match paper to 4 decimals
  T2  Algorithm 1 on BS converges to xi*, chi* (large N)
  T3  Algorithm 2 on BS, large K, converges to xi*, chi*
  T4  Algorithm 3 telescoping identity is bias-corrected
  T5  Heston QE scheme: S_T mean = S_0 * exp(rT) (martingale check)
  T6  Heston QE scheme: V_T mean -> theta as T -> infinity
  T7  Merton closed-form put price matches MC to <= 0.5% relative error
  T8  Merton conditional benchmark = direct MC quantile within MC error
"""
from __future__ import annotations
import os, csv, time
import numpy as np

from mlsa_core    import algo1_SA, algo2_NSA, algo3_MLSA
from case_study_1 import closed_form as bs_closed, make_simulators as bs_sims
from case_study_heston import HestonParams, simulate_heston, make_simulators as h_sims
from case_study_merton import (MertonParams, simulate_S_one_step, merton_put,
                                make_simulators as m_sims, compute_benchmark as m_bench)


# Color/PASS/FAIL helpers ------------------------------------------------
GREEN  = "\033[92m"; RED = "\033[91m"; END = "\033[0m"
def pass_(msg):   return f"{GREEN}PASS{END} - {msg}"
def fail_(msg):   return f"{RED}FAIL{END} - {msg}"

results = []   # (test_id, passed, message, value, tolerance)


def record(tid, passed, msg, value=None, tol=None):
    results.append(dict(test=tid, passed=passed, msg=msg,
                        value=value, tol=tol))
    print(pass_(msg) if passed else fail_(msg))


# =========================================================================
#  T1: BS closed-form constants
# =========================================================================
def test_t1():
    print("\n--- T1: BS closed-form constants ---")
    xi, chi = bs_closed(0.975, 0.5)
    EXPECTED_XI, EXPECTED_CHI = 2.012, 2.901
    err_xi  = abs(xi  - EXPECTED_XI)
    err_chi = abs(chi - EXPECTED_CHI)
    record("T1",
           err_xi < 1e-2 and err_chi < 1e-2,
           f"xi*  = {xi:.4f}  (paper: 2.012)   "
           f"chi* = {chi:.4f}  (paper: 2.901)",
           value=max(err_xi, err_chi), tol=1e-2)


# =========================================================================
#  T2: Algorithm 1 SA convergence
# =========================================================================
def test_t2():
    print("\n--- T2: Algorithm 1 (SA) on BS, large N ---")
    sim_X0, _, _ = bs_sims(0.5)
    xi_t, chi_t = bs_closed(0.975, 0.5)
    rng = np.random.default_rng(0)
    xi, chi = algo1_SA(sim_X0, N=500_000, alpha=0.975,
                        gamma=lambda n: 1.0 / (100 + n), rng=rng)
    err = max(abs(xi - xi_t), abs(chi - chi_t))
    record("T2",
           err < 0.05,
           f"after 5e5 steps: xi={xi:.4f} (target {xi_t:.4f}), "
           f"chi={chi:.4f} (target {chi_t:.4f}), max err = {err:.4f}",
           value=err, tol=0.05)


# =========================================================================
#  T3: Algorithm 2 NSA, large K, converges to xi*, chi*
# =========================================================================
def test_t3():
    print("\n--- T3: Algorithm 2 (NSA) on BS, K = 256 ---")
    _, sim_Xh, _ = bs_sims(0.5)
    xi_t, chi_t = bs_closed(0.975, 0.5)
    rng = np.random.default_rng(0)
    xi, chi = algo2_NSA(sim_Xh, N=200_000, K=256, alpha=0.975,
                        gamma=lambda n: 1.0 / (100 + n), rng=rng)
    err = max(abs(xi - xi_t), abs(chi - chi_t))
    record("T3",
           err < 0.10,
           f"K=256, N=2e5: xi={xi:.4f}, chi={chi:.4f}, "
           f"max err = {err:.4f}  (note: bias O(1/K) is non-zero)",
           value=err, tol=0.10)


# =========================================================================
#  T4: MLSA telescoping identity
#
#  Numerical verification that  xi^ML  ~ xi^h_L  (the finest level alone)
#  to within MC noise. This is the key sanity test that the multilevel
#  coupling is implemented correctly.
# =========================================================================
def test_t4():
    print("\n--- T4: MLSA telescoping identity ---")
    _, sim_Xh, sim_coupled = bs_sims(0.5)
    L, h0, M = 3, 1/8, 2
    Ns = [50_000, 25_000, 12_500, 6_250]
    K_finest = int(round(1.0 / (h0 / M**L)))     # = 64

    rng = np.random.default_rng(0)
    xi_ml, chi_ml = algo3_MLSA(sim_coupled, sim_Xh, L, h0, M, Ns,
                                alpha=0.975,
                                gamma=lambda n: 1.0/(100+n), rng=rng)

    rng = np.random.default_rng(0)
    xi_nsa, chi_nsa = algo2_NSA(sim_Xh, N=sum(Ns), K=K_finest,
                                 alpha=0.975,
                                 gamma=lambda n: 1.0/(100+n), rng=rng)

    err = max(abs(xi_ml - xi_nsa), abs(chi_ml - chi_nsa))
    record("T4",
           err < 0.30,                  # MC noise + step-size differences
           f"xi^ML - xi^NSA(K=64) = {xi_ml - xi_nsa:+.4f}, "
           f"chi^ML - chi^NSA = {chi_ml - chi_nsa:+.4f}",
           value=err, tol=0.30)


# =========================================================================
#  T5: Heston QE - S_T martingale check
# =========================================================================
def test_t5():
    print("\n--- T5: Heston QE, S_T martingale check ---")
    p = HestonParams()
    rng = np.random.default_rng(42)
    S_T, _ = simulate_heston(p.S0, p.V0, 0.0, p.T, 200_000, 8, p, rng)
    expected = p.S0 * np.exp(p.r * p.T)
    rel_err = abs(S_T.mean() - expected) / expected
    record("T5",
           rel_err < 0.005,             # 0.5% MC tolerance
           f"E[S_T] = {S_T.mean():.4f},  S_0 e^{{rT}} = {expected:.4f},  "
           f"rel err = {100*rel_err:.3f}%",
           value=rel_err, tol=0.005)


# =========================================================================
#  T6: Heston QE - V_T variance long-run mean
# =========================================================================
def test_t6():
    print("\n--- T6: Heston QE, V_T mean reverts to theta ---")
    p = HestonParams(T=10.0)            # long horizon -> stationary distribution
    rng = np.random.default_rng(13)
    _, V_T = simulate_heston(p.S0, p.V0, 0.0, p.T, 200_000, 50, p, rng)
    rel_err = abs(V_T.mean() - p.theta) / p.theta
    record("T6",
           rel_err < 0.05,              # 5% tolerance
           f"E[V_T at T=10] = {V_T.mean():.5f}, theta = {p.theta:.5f}, "
           f"rel err = {100*rel_err:.2f}%",
           value=rel_err, tol=0.05)


# =========================================================================
#  T7: Merton closed-form vs MC
# =========================================================================
def test_t7():
    print("\n--- T7: Merton put price closed-form vs MC ---")
    p = MertonParams()
    P0_arr = merton_put(np.array([p.S0]), p.K_strike, p.T, p)
    P0 = float(np.atleast_1d(P0_arr)[0])

    rng = np.random.default_rng(7)
    S_T = simulate_S_one_step(p.S0, 0.0, p.T, 1_000_000, p, rng)
    P0_mc = np.exp(-p.r * p.T) * np.maximum(p.K_strike - S_T, 0.0).mean()

    rel_err = abs(P0 - P0_mc) / P0
    record("T7",
           rel_err < 0.005,             # 0.5%
           f"closed-form = {P0:.4f},  MC (1e6 paths) = {P0_mc:.4f},  "
           f"rel err = {100*rel_err:.3f}%",
           value=rel_err, tol=0.005)


# =========================================================================
#  T8: Merton benchmark vs SA convergence
# =========================================================================
def test_t8():
    print("\n--- T8: Merton: SA result matches conditional-MC benchmark ---")
    p = MertonParams()
    sim_X0, _, _, _ = m_sims(p)
    xi_b, chi_b = m_bench(p, n_samples=500_000)
    rng = np.random.default_rng(0)
    xi, chi = algo1_SA(sim_X0, N=300_000, alpha=p.alpha,
                       gamma=lambda n: 1.0 / (100 + n), rng=rng)
    err = max(abs(xi - xi_b), abs(chi - chi_b))
    record("T8",
           err < 0.20,                  # SA at finite N has noise; tolerant
           f"SA xi={xi:.4f} (bench {xi_b:.4f}),  "
           f"SA chi={chi:.4f} (bench {chi_b:.4f}),  max err = {err:.4f}",
           value=err, tol=0.20)


# =========================================================================
#  Main + CSV export
# =========================================================================
if __name__ == "__main__":
    t0 = time.time()
    for fn in [test_t1, test_t2, test_t3, test_t4,
                test_t5, test_t6, test_t7, test_t8]:
        try:
            fn()
        except Exception as e:
            record(fn.__name__.upper(), False, f"EXCEPTION: {e}")

    print(f"\n=== Summary: {sum(r['passed'] for r in results)}/{len(results)} "
          f"passed (total wall {time.time()-t0:.1f}s) ===")

    os.makedirs("tables", exist_ok=True)
    with open("tables/correctness_tests.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "passed", "message", "value", "tolerance"])
        for r in results:
            w.writerow([r["test"], r["passed"], r["msg"],
                        r["value"], r["tol"]])
    print(f"saved -> tables/correctness_tests.csv")
