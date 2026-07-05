"""
mlsa_core.py
=============
Core implementations of Algorithms 1, 2, 3 from Crepey, Frikha & Louzi (2025),
"A multilevel stochastic approximation algorithm for value-at-risk
and expected shortfall estimation", Finance and Stochastics 29: 1015-1074.

Notation follows the paper:
- alpha : confidence level in (0, 1)
- xi    : VaR iterate
- chi   : ES iterate
- gamma_n : step-size for the (xi) update
- (chi) update uses 1/(n+1) (Polyak-Ruppert style averaging weight)

H1 and H2 are the gradient operators in (2.4)-(2.5):
    H1(xi, x)         = 1 - 1/(1-alpha) * 1{x >= xi}
    H2(chi, xi, x)    = chi - (xi + (x - xi)^+ / (1-alpha))

Heartbeat / progress
--------------------
All three algorithms accept an optional ``progress`` argument which is
a callable ``(message: str) -> None``. If supplied, it is invoked with
short status strings every ``heartbeat_every`` iterations (default
50_000), letting long runs surface progress without slowing the inner
loop appreciably. Pass ``progress=print`` to dump to stdout, or wire it
to a logger / tqdm bar.
"""
from __future__ import annotations

import sys, time
import numpy as np
from typing import Callable, Optional, Tuple

# Default no-op progress callback
_NO_PROGRESS: Callable[[str], None] = lambda _msg: None


def _make_heartbeat(progress: Optional[Callable[[str], None]],
                    every: int,
                    label: str):
    """Returns a (call, last_t) pair that respects an `every` interval."""
    if progress is None:
        return (lambda n, xi, chi: None)
    t0 = time.perf_counter()
    state = {"last_n": 0, "last_t": t0}
    def _beat(n: int, xi: float, chi: float):
        if n - state["last_n"] >= every:
            now = time.perf_counter()
            rate = (n - state["last_n"]) / max(now - state["last_t"], 1e-9)
            progress(f"  [{label}] iter {n:>9,d}  "
                     f"xi={xi:+.4f}  chi={chi:+.4f}  "
                     f"({rate:.0f} it/s)")
            state["last_n"] = n
            state["last_t"] = now
    return _beat


# ---------------------------------------------------------------------------
#  Operators H1, H2  (eqs. (2.4), (2.5) in the paper)
# ---------------------------------------------------------------------------

def H1(xi: float, x: np.ndarray | float, alpha: float) -> np.ndarray | float:
    return 1.0 - (x >= xi) / (1.0 - alpha)

def H2(chi: float, xi: float, x: np.ndarray | float,
       alpha: float) -> np.ndarray | float:
    return chi - (xi + np.maximum(x - xi, 0.0) / (1.0 - alpha))


# ---------------------------------------------------------------------------
#  Algorithm 1 :  Standard SA (Robbins-Monro) when X_0 is directly simulatable
# ---------------------------------------------------------------------------

def algo1_SA(simulate_X0: Callable[[int, np.random.Generator], np.ndarray],
             N: int,
             alpha: float,
             gamma: Callable[[int], float],
             xi0: float = 0.0,
             chi0: float = 0.0,
             rng: np.random.Generator | None = None,
             progress: Optional[Callable[[str], None]] = None,
             heartbeat_every: int = 50_000,
             ) -> Tuple[float, float]:
    """
    Algorithm 1 of the paper.

    Parameters
    ----------
    simulate_X0 : (N, rng) -> ndarray of length N drawn i.i.d. from the law of X_0
    N           : total number of SA iterations
    alpha       : confidence level
    gamma       : function n -> learning-rate gamma_n (n>=1)
    progress    : optional callable for periodic heartbeat messages
    heartbeat_every : iterations between heartbeats
    """
    rng = np.random.default_rng() if rng is None else rng
    X = simulate_X0(N, rng)
    xi, chi = xi0, chi0
    beat = _make_heartbeat(progress, heartbeat_every, "Alg1")
    for n in range(N):
        x = X[n]
        gn = gamma(n + 1)
        xi  -= gn * H1(xi, x, alpha)
        chi -= 1.0 / (n + 1) * H2(chi, xi, x, alpha)
        beat(n + 1, xi, chi)
    return xi, chi


# ---------------------------------------------------------------------------
#  Algorithm 2 :  Nested SA  (NSA)
# ---------------------------------------------------------------------------

def algo2_NSA(simulate_Xh: Callable[[int, int, np.random.Generator], np.ndarray],
              N: int,
              K: int,
              alpha: float,
              gamma: Callable[[int], float],
              xi0: float = 0.0,
              chi0: float = 0.0,
              rng: np.random.Generator | None = None,
              progress: Optional[Callable[[str], None]] = None,
              heartbeat_every: int = 50_000,
              ) -> Tuple[float, float]:
    """
    Algorithm 2 of the paper. The inner-layer estimator is

         X_h^{(n+1)}  =  (1/K) * sum_{k=1..K} phi(Y^{(n+1)}, Z^{(n+1,k)}).

    The user must supply a function `simulate_Xh(N, K, rng)` returning a
    1-D array of length N, each entry being one simulated X_h.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = simulate_Xh(N, K, rng)
    xi, chi = xi0, chi0
    beat = _make_heartbeat(progress, heartbeat_every, f"Alg2 K={K}")
    for n in range(N):
        x = X[n]
        gn = gamma(n + 1)
        xi  -= gn * H1(xi, x, alpha)
        chi -= 1.0 / (n + 1) * H2(chi, xi, x, alpha)
        beat(n + 1, xi, chi)
    return xi, chi


# ---------------------------------------------------------------------------
#  Algorithm 3 :  Multilevel SA  (MLSA)
# ---------------------------------------------------------------------------

def algo3_MLSA(
    simulate_coupled_pair: Callable[
        [int, int, int, np.random.Generator],
        Tuple[np.ndarray, np.ndarray]
    ],
    simulate_Xh0: Callable[[int, int, np.random.Generator], np.ndarray],
    L: int,
    h0: float,
    M: int,
    Ns: list[int],
    alpha: float,
    gamma: Callable[[int], float],
    rng: np.random.Generator | None = None,
    progress: Optional[Callable[[str], None]] = None,
    heartbeat_every: int = 50_000,
) -> Tuple[float, float]:
    """
    Algorithm 3 of the paper.

    Parameters
    ----------
    simulate_coupled_pair(N, K_coarse, K_fine, rng) -> (X_coarse, X_fine)
        Returns two N-vectors using *the same* outer Y_n and consistent
        Z's so that X_coarse and X_fine come from the multilevel coupling
        described in Eqs. (4.1)-(4.2) of the paper.

    simulate_Xh0(N, K_coarse, rng) -> X_h0
        Returns an N-vector of i.i.d. X_h0 (level 0).

    L      : number of refinement levels (>=1)
    h0     : 1/K0   (coarse bias)
    M      : geometric refinement factor (>=2)
    Ns     : list/tuple of length L+1 with the number of iterations N_l per level
    """
    rng = np.random.default_rng() if rng is None else rng
    assert len(Ns) == L + 1

    # ----- bias parameters and inner-MC sample sizes
    h_levels = [h0 / (M ** ell) for ell in range(L + 1)]
    K_levels = [int(round(1.0 / h)) for h in h_levels]   # K_l = K0 * M^l

    if progress is not None:
        progress(f"  [Alg3] L={L} levels, K_levels={K_levels}, "
                 f"Ns={Ns}, total inner samples={sum(N*K for N,K in zip(Ns, K_levels)):,}")

    # ============= level 0 =============
    xi_h0, chi_h0 = 0.0, 0.0
    X0 = simulate_Xh0(Ns[0], K_levels[0], rng)
    beat0 = _make_heartbeat(progress, heartbeat_every, "Alg3 L=0")
    for n in range(Ns[0]):
        x  = X0[n]
        gn = gamma(n + 1)
        xi_h0  -= gn * H1(xi_h0, x, alpha)
        chi_h0 -= 1.0 / (n + 1) * H2(chi_h0, xi_h0, x, alpha)
        beat0(n + 1, xi_h0, chi_h0)

    # We iteratively store level-l minus level-(l-1) increments
    xi_increments  = []
    chi_increments = []

    for ell in range(1, L + 1):
        # Coupled pair: (X_{h_{l-1}}^{(n)}, X_{h_l}^{(n)})  i.i.d. across n.
        # The fine estimator reuses the coarse Z's, then adds (M-1)*K_{l-1}
        # extra Z samples to refine.
        Xc, Xf = simulate_coupled_pair(
            Ns[ell], K_levels[ell - 1], K_levels[ell], rng
        )

        xi_c, chi_c = 0.0, 0.0
        xi_f, chi_f = 0.0, 0.0
        beat_l = _make_heartbeat(progress, heartbeat_every, f"Alg3 L={ell}")
        for n in range(Ns[ell]):
            xc, xf = Xc[n], Xf[n]
            gn = gamma(n + 1)
            xi_c  -= gn * H1(xi_c, xc, alpha)
            chi_c -= 1.0 / (n + 1) * H2(chi_c, xi_c, xc, alpha)
            xi_f  -= gn * H1(xi_f, xf, alpha)
            chi_f -= 1.0 / (n + 1) * H2(chi_f, xi_f, xf, alpha)
            beat_l(n + 1, xi_f, chi_f)

        xi_increments.append(xi_f - xi_c)
        chi_increments.append(chi_f - chi_c)

    xi_ML  = xi_h0  + sum(xi_increments)
    chi_ML = chi_h0 + sum(chi_increments)
    return xi_ML, chi_ML
