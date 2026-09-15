#!/usr/bin/env python3
"""
strategic_switching_experiments.py

Complete Python numerical pipeline for the manuscript

    Sparse One-Step-Ahead Optimal Control of Time-Varying
    Affine Opinion Networks: Tracking and Competitive Games

The script reproduces every numerical artifact consumed by the LaTeX
manuscript.  By default it writes to

    ./strategic_switching_outputs/

so it can be run from the manuscript directory without changing any LaTeX
paths.

Dependencies
------------
Python >= 3.10
numpy
matplotlib

No optimization package is required.  Small exact benchmarks use explicit
support enumeration and NumPy linear solves.

Important reproducibility choice
--------------------------------
The five-node Wang et al. matrices are copied exactly from the manuscript.
The 8- and 10-node benchmarks are deterministic Python-generated periodic
row-stochastic networks with fixed seeds.  Their realized matrices are also
exported to W8_matrices.csv and W10_matrices.csv.  This deliberately avoids
trying to reproduce Julia-specific MersenneTwister search trajectories.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np

# Matplotlib is imported lazily by save_figures() so the numerical CSV/TeX
# outputs can still be generated on a machine without a graphical backend.

GLOBAL_SEED = 20260820
NETWORK_SEED_8 = 142008
NETWORK_SEED_10 = 410010
SETTLE_EPS = 1e-6
CYCLE_ZERO_TOL = 1e-9
ACTIVATION_VALUE_TOL = 1e-24
BUDGET_MC_SEED = 20260902
BUDGET_MC_SAMPLES = 50
COMP_MC_BASE_SEED = 510000
COMP_MC_SAMPLES = 10
RANDOM_BASELINE_SEEDS = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all manuscript numerical outputs.")
    p.add_argument(
        "--out",
        default="strategic_switching_outputs",
        help="output directory (default: strategic_switching_outputs)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="reduce timing/correctness repetitions for a fast smoke test",
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="skip PDF/PNG figure generation",
    )
    return p.parse_args()


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def fmt(x: object, digits: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    xx = float(x)
    if math.isnan(xx):
        return "NA"
    return f"{xx:.{digits}f}"


def fmt_sci_tex(x: float, digits: int = 3) -> str:
    x = float(x)
    if x == 0.0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = x / (10.0**e)
    return f"{m:.{digits}f}\\times 10^{{{e}}}"


def combinations_of(v: Sequence[int], s: int):
    return itertools.combinations(v, s)


def support_vectors(n: int, targets: Sequence[int], r: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for s in range(0, min(r, len(targets)) + 1):
        for S in combinations_of(targets, s):
            b = np.zeros(n, dtype=float)
            if S:
                b[list(S)] = 1.0
            out.append(b)
    return out


def support_indices_1based(b: np.ndarray, atol: float = 0.5) -> np.ndarray:
    return np.flatnonzero(b > atol) + 1


# ---------------------------------------------------------------------------
# Deterministic periodic row-stochastic benchmark networks
# ---------------------------------------------------------------------------


def deterministic_periodic_network(n: int, seed: int, size: int = 4) -> list[np.ndarray]:
    """Generate a fixed heterogeneous periodic row-stochastic network.

    Each row retains a positive self-weight and mixes with three additional
    neighbors.  The generator is used with fixed PCG64 seeds and the realized
    matrices are exported, so every reported experiment is fully auditable.
    """
    rng = np.random.default_rng(seed)
    Wset: list[np.ndarray] = []
    for phase in range(size):
        W = np.zeros((n, n), dtype=float)
        for i in range(n):
            pool = [j for j in range(n) if j != i]
            nbrs = rng.choice(pool, size=min(3, n - 1), replace=False)
            cols = np.array([i, *nbrs], dtype=int)
            # Phase-dependent but strictly positive weights.
            raw = rng.uniform(0.15, 1.0, size=len(cols))
            raw[0] += 0.35 + 0.05 * phase
            raw /= raw.sum()
            W[i, cols] = raw
        Wset.append(W)
    return Wset


def write_matrix_sequence_csv(path: Path, Wseq: Sequence[np.ndarray]) -> None:
    rows = []
    for phase, W in enumerate(Wseq, start=1):
        n = W.shape[0]
        for i in range(n):
            for j in range(n):
                rows.append((phase, i + 1, j + 1, f"{W[i, j]:.17g}"))
    write_csv(path, ["phase", "row", "col", "value"], rows)


# ---------------------------------------------------------------------------
# Exact sparse OSAOC and exhaustive validation
# ---------------------------------------------------------------------------


@dataclass
class SparseSolution:
    b: np.ndarray
    u: float
    S: list[int]
    value: float


def sparse_osaoc(
    c: np.ndarray,
    targets: Sequence[int],
    r: int,
    gamma: float,
    activation_tol: float = 0.0,
) -> SparseSolution:
    """Exact shared-amplitude sparse OSAOC.

    activation_tol is a numerical objective-improvement dead-band.  The
    mathematical optimizer corresponds to activation_tol=0.  Manuscript
    simulations use ACTIVATION_VALUE_TOL only to suppress round-off supports.
    """
    N = len(targets)
    ruse = min(r, N)
    sorted_nodes = sorted(targets, key=lambda i: (-float(c[i]), i))
    sorted_vals = [float(c[i]) for i in sorted_nodes]

    p = 0.0
    q = 0.0
    bestV = float(activation_tol)
    bestS: list[int] = []

    for s in range(1, ruse + 1):
        p += sorted_vals[s - 1]
        q += sorted_vals[N - s]
        Vp = p * p / (s + gamma)
        Vq = q * q / (s + gamma)
        if Vp > bestV:
            bestV = Vp
            bestS = sorted(sorted_nodes[:s])
        if Vq > bestV:
            bestV = Vq
            bestS = sorted(sorted_nodes[N - s :])

    b = np.zeros(len(c), dtype=float)
    if bestS:
        b[bestS] = 1.0
        u = float(np.sum(c[bestS]) / (len(bestS) + gamma))
    else:
        u = 0.0
    return SparseSolution(b=b, u=u, S=bestS, value=float(bestV if bestS else 0.0))

def exhaustive_sparse(c: np.ndarray, targets: Sequence[int], r: int, gamma: float):
    bestV = 0.0
    bestS: list[int] = []
    for s in range(0, min(r, len(targets)) + 1):
        for S_tuple in combinations_of(targets, s):
            S = list(S_tuple)
            val = 0.0 if not S else float(np.sum(c[S]) ** 2 / (s + gamma))
            if val > bestV:
                bestV = val
                bestS = S
    return bestS, bestV


def sorting_validation(outdir: Path, quick: bool = False):
    rng = np.random.default_rng(GLOBAL_SEED)
    random_tests = 250 if quick else 2000
    maxerr = 0.0

    for _ in range(random_tests):
        N = int(rng.integers(4, 15))
        r = int(rng.integers(1, min(5, N) + 1))
        gamma = 10.0 ** (2.0 * rng.random() - 1.0)
        c = rng.standard_normal(N)
        ss = sparse_osaoc(c, list(range(N)), r, gamma)
        _, ev = exhaustive_sparse(c, list(range(N)), r, gamma)
        maxerr = max(maxerr, abs(ss.value - ev))

    tied = [
        np.array([2.0, 2.0, 1.0, -1.0, -2.0, -2.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([3.0, -3.0, 3.0, -3.0, 0.0]),
        np.zeros(4),
    ]
    for c in tied:
        r = min(3, len(c))
        ss = sparse_osaoc(c, list(range(len(c))), r, 0.5)
        _, ev = exhaustive_sparse(c, list(range(len(c))), r, 0.5)
        maxerr = max(maxerr, abs(ss.value - ev))

    validation_rows = []
    reps = 5 if quick else 20
    for N in [8, 10, 12, 14, 16, 18]:
        r = min(5, N)
        tsort, tenum = [], []
        rowerr = 0.0
        c0 = rng.standard_normal(N)
        sparse_osaoc(c0, list(range(N)), r, 0.5)
        exhaustive_sparse(c0, list(range(N)), r, 0.5)
        for _ in range(reps):
            c = rng.standard_normal(N)
            t0 = time.perf_counter()
            ss = sparse_osaoc(c, list(range(N)), r, 0.5)
            tsort.append(1e3 * (time.perf_counter() - t0))
            t0 = time.perf_counter()
            _, ev = exhaustive_sparse(c, list(range(N)), r, 0.5)
            tenum.append(1e3 * (time.perf_counter() - t0))
            rowerr = max(rowerr, abs(ss.value - ev))
        maxerr = max(maxerr, rowerr)
        validation_rows.append((N, r, median(tsort), median(tenum), rowerr))

    write_csv(
        outdir / "sorting_validation.csv",
        ["N", "r", "median_sort_ms", "median_exhaustive_ms", "max_objective_error"],
        validation_rows,
    )

    timing_rows = []
    for N in [10, 50, 100, 500, 1000, 5000, 10000]:
        r = min(20, N)
        reps2 = (10 if N <= 1000 else 5) if quick else (100 if N <= 1000 else 30)
        times = []
        for _ in range(reps2):
            c = rng.standard_normal(N)
            t0 = time.perf_counter()
            sparse_osaoc(c, list(range(N)), r, 0.5)
            times.append(1e3 * (time.perf_counter() - t0))
        timing_rows.append((N, r, median(times)))
    write_csv(outdir / "sorting_scaling.csv", ["N", "r", "median_sort_ms"], timing_rows)

    return {
        "total_tests": random_tests + len(tied),
        "maxerr": maxerr,
        "validation_rows": validation_rows,
        "timing_rows": timing_rows,
    }


# ---------------------------------------------------------------------------
# Five-node periodic benchmark
# ---------------------------------------------------------------------------


def five_node_matrices() -> list[np.ndarray]:
    W1 = np.array([
        [1/2, 1/2, 0, 0, 0],
        [1/2, 1/2, 0, 0, 0],
        [1/2, 1/2, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1/2, 1/2, 0],
        [0, 0, 1/2, 1/2, 0],
        [0, 0, 1/2, 1/2, 0],
    ], dtype=float)
    W3 = np.array([
        [1/2, 0, 0, 0, 1/2],
        [1/2, 0, 0, 0, 1/2],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1/2, 0, 0, 0, 1/2],
    ], dtype=float)
    W4 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1/2, 1/2, 0, 0],
        [0, 1/2, 1/2, 0, 0],
        [0, 1/2, 1/2, 0, 0],
        [0, 0, 0, 0, 1],
    ], dtype=float)
    W5 = np.array([
        [0, 0, 0, 1/2, 1/2],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1/2, 1/2],
        [0, 0, 0, 1/2, 1/2],
    ], dtype=float)
    return [W1, W2, W3, W4, W5]


def simulate_open(Wseq: Sequence[np.ndarray], x0: np.ndarray, K: int) -> np.ndarray:
    X = np.zeros((len(x0), K + 1))
    X[:, 0] = x0
    for k in range(K):
        X[:, k + 1] = Wseq[k % len(Wseq)] @ X[:, k]
    return X


def simulate_single(Wseq, x0, targets, g, r, gamma, K, activation_tol=ACTIVATION_VALUE_TOL):
    n = len(x0)
    X = np.zeros((n, K + 1))
    X[:, 0] = x0
    U = np.zeros(K)
    card = np.zeros(K, dtype=int)
    Bsel = np.zeros((n, K))
    Et = np.zeros(K + 1)
    Et[0] = np.sum((X[targets, 0] - g[targets]) ** 2)
    for k in range(K):
        W = Wseq[k % len(Wseq)]
        z = W @ X[:, k]
        sol = sparse_osaoc(g - z, targets, r, gamma, activation_tol=activation_tol)
        X[:, k + 1] = z + sol.b * sol.u
        U[k] = sol.u
        card[k] = int(round(np.sum(sol.b)))
        Bsel[:, k] = sol.b
        Et[k + 1] = np.sum((X[targets, k + 1] - g[targets]) ** 2)
    return {"X": X, "U": U, "card": card, "Bsel": Bsel, "Et": Et}

def sparse_osaoc_exact_cardinality(c: np.ndarray, targets: Sequence[int], r: int, gamma: float, activation_tol: float = 0.0) -> SparseSolution:
    """Exact shared-amplitude OSAOC with |S| fixed to r (apart from dead-band)."""
    N = len(targets)
    if r <= 0 or r > N:
        raise ValueError("exact cardinality requires 1 <= r <= number of targets")
    sorted_nodes = sorted(targets, key=lambda i: (-float(c[i]), i))
    top = sorted_nodes[:r]
    bot = sorted_nodes[N-r:]
    atop = float(np.sum(c[top])); abot = float(np.sum(c[bot]))
    Vtop = atop * atop / (r + gamma); Vbot = abot * abot / (r + gamma)
    if max(Vtop, Vbot) <= activation_tol:
        return SparseSolution(np.zeros(len(c)), 0.0, [], 0.0)
    S = sorted(top if Vtop >= Vbot else bot)
    b = np.zeros(len(c)); b[S] = 1.0
    u = float(np.sum(c[S]) / (r + gamma))
    return SparseSolution(b, u, S, max(Vtop, Vbot))


def simulate_single_exact_cardinality(Wseq, x0, targets, g, r, gamma, K, activation_tol=ACTIVATION_VALUE_TOL):
    n = len(x0); X = np.zeros((n, K + 1)); X[:, 0] = x0
    U = np.zeros(K); card = np.zeros(K, dtype=int); Et = np.zeros(K + 1)
    Et[0] = np.sum((x0[targets] - g[targets]) ** 2)
    for k in range(K):
        z = Wseq[k % len(Wseq)] @ X[:, k]
        sol = sparse_osaoc_exact_cardinality(g-z, targets, r, gamma, activation_tol)
        X[:, k+1] = z + sol.b * sol.u
        U[k] = sol.u; card[k] = len(sol.S)
        Et[k+1] = np.sum((X[targets, k+1] - g[targets]) ** 2)
    return {"X": X, "U": U, "card": card, "Et": Et}


def mean_cardinality_to_settle(sim, tset):
    if tset is None:
        return float(np.mean(sim["card"]))
    stop = max(1, min(int(tset), len(sim["card"])))
    return float(np.mean(sim["card"][:stop]))


def settling_time(Et: np.ndarray, eps: float = SETTLE_EPS):
    """Finite-horizon settling time.

    Returns the first k for which Et[j] <= eps for every remaining simulated
    state j=k,...,K.  None means that the trajectory does not settle within
    the simulation horizon.
    """
    for k in range(len(Et)):
        if np.all(Et[k:] <= eps):
            return k
    return None

def last_active_time(U: np.ndarray, tol: float = 0.0) -> int:
    inds = np.flatnonzero(np.abs(U) > tol)
    return -1 if len(inds) == 0 else int(inds[-1])

def realized_cycle_gains(X: np.ndarray, g: np.ndarray, period: int):
    ncycles = (X.shape[1] - 1) // period
    chi = np.full(ncycles, np.nan)
    den = np.zeros(ncycles)
    for m in range(ncycles):
        e0 = X[:, m * period] - g
        e1 = X[:, (m + 1) * period] - g
        d0 = np.linalg.norm(e0)
        den[m] = d0
        if d0 > CYCLE_ZERO_TOL:
            chi[m] = np.linalg.norm(e1) / d0
    return chi, den

def five_node_experiment(outdir: Path):
    W = five_node_matrices()
    Wcons = W
    Wosc = [W[0], W[4], W[3], W[2], W[1]]
    x0 = np.array([0.43, 0.18, 0.90, 0.97, 0.45])
    targets = [0, 2, 4]
    g = 0.2 * np.ones(5)
    r, gamma, K, period = 3, 0.5, 150, 5
    open_cons = simulate_open(Wcons, x0, K)
    open_osc = simulate_open(Wosc, x0, K)
    cons = simulate_single(Wcons, x0, targets, g, r, gamma, K)
    osc = simulate_single(Wosc, x0, targets, g, r, gamma, K)
    chi_cons, den_cons = realized_cycle_gains(cons["X"], g, period)
    chi_osc, den_osc = realized_cycle_gains(osc["X"], g, period)
    aobs_cons = float(np.nanmax(chi_cons))
    aobs_osc = float(np.nanmax(chi_osc))
    write_csv(
        outdir / "five_node_cycle_gains.csv",
        ["cycle", "cycle_start_error_consensus", "chi_consensus", "cycle_start_error_oscillatory", "chi_oscillatory"],
        [(m, den_cons[m], chi_cons[m], den_osc[m], chi_osc[m]) for m in range(max(len(chi_cons), len(chi_osc)))],
    )
    tail = open_osc[:, -100:]
    rows = [
        ("open_consensus_limit", float(np.mean(open_cons[:, -1]))),
        ("open_osc_min_last100", float(np.min(tail))),
        ("open_osc_max_last100", float(np.max(tail))),
        ("last_active_k_consensus", last_active_time(cons["U"])),
        ("last_active_k_oscillatory", last_active_time(osc["U"])),
        ("aobs_consensus", aobs_cons),
        ("aobs_oscillatory", aobs_osc),
    ]
    write_csv(outdir / "five_node_summary.csv", ["metric", "value"], rows)
    write_matrix_sequence_csv(outdir / "W5_matrices.csv", W)
    return locals()

# ---------------------------------------------------------------------------
# Friedkin--Johnsen specialization
# ---------------------------------------------------------------------------


def simulate_single_affine(Wseq, Lambda, prejudice, x0, targets, g, r, gamma, K, activation_tol=ACTIVATION_VALUE_TOL):
    n = len(x0)
    Fseq = [Lambda @ W for W in Wseq]
    h = (np.eye(n) - Lambda) @ prejudice
    X = np.zeros((n, K + 1)); X[:, 0] = x0
    U = np.zeros(K); card = np.zeros(K, dtype=int); Bsel = np.zeros((n, K))
    Et = np.zeros(K + 1); Et[0] = np.sum((X[targets, 0] - g[targets]) ** 2)
    for k in range(K):
        F = Fseq[k % len(Fseq)]
        z = F @ X[:, k] + h
        sol = sparse_osaoc(g - z, targets, r, gamma, activation_tol=activation_tol)
        X[:, k + 1] = z + sol.b * sol.u
        U[k] = sol.u; card[k] = int(round(np.sum(sol.b))); Bsel[:, k] = sol.b
        Et[k + 1] = np.sum((X[targets, k + 1] - g[targets]) ** 2)
    return {"X": X, "U": U, "card": card, "Bsel": Bsel, "Et": Et, "Fseq": Fseq, "h": h}

def affine_cycle_diagnostics(sim, g, gamma, period):
    n = len(g); ncycles = (sim["X"].shape[1] - 1) // period
    chi = np.full(ncycles, np.nan)
    actual_ratio = np.full(ncycles, np.nan)
    eta_norm = np.zeros(ncycles)
    cycle_start_error = np.zeros(ncycles)
    for m in range(ncycles):
        Aprod = np.eye(n); eta = np.zeros(n)
        for j in range(period):
            k = m * period + j
            F = sim["Fseq"][k % period]
            b = sim["Bsel"][:, k]; s = np.sum(b)
            Phi = np.outer(b, b) / (s + gamma) if s > 0 else np.zeros((n, n))
            A = (np.eye(n) - Phi) @ F
            d = (np.eye(n) - Phi) @ (F @ g + sim["h"] - g)
            Aprod = A @ Aprod
            eta = A @ eta + d
        e0 = sim["X"][:, m * period] - g
        e1 = sim["X"][:, (m + 1) * period] - g
        d0 = np.linalg.norm(e0)
        cycle_start_error[m] = d0
        if d0 > CYCLE_ZERO_TOL:
            chi[m] = np.linalg.norm(Aprod @ e0) / d0
            actual_ratio[m] = np.linalg.norm(e1) / d0
        eta_norm[m] = np.linalg.norm(eta)
    return {"chi": chi, "actual_ratio": actual_ratio, "eta_norm": eta_norm, "cycle_start_error": cycle_start_error}

def fj_specialization_experiment(outdir: Path):
    Wseq = five_node_matrices()
    x0 = np.array([0.43, 0.18, 0.90, 0.97, 0.45])
    targets = [0, 2, 4]; g = 0.2 * np.ones(5); r, gamma, K, period = 3, 0.5, 150, 5
    lam = np.array([0.55, 0.65, 0.70, 0.60, 0.75]); Lambda = np.diag(lam)
    s_invariant = g.copy(); s_mismatch = np.array([0.05, 0.35, -0.10, 0.45, 0.00])
    invariant = simulate_single_affine(Wseq, Lambda, s_invariant, x0, targets, g, r, gamma, K)
    mismatch = simulate_single_affine(Wseq, Lambda, s_mismatch, x0, targets, g, r, gamma, K)
    dinv = affine_cycle_diagnostics(invariant, g, gamma, period)
    dmis = affine_cycle_diagnostics(mismatch, g, gamma, period)
    fnorms = [np.linalg.norm(F, 2) for F in invariant["Fseq"]]
    beta = float(max(fnorms)); abar_bound = float(np.prod(fnorms)); G = sum(beta**j for j in range(period))
    eps_inv = float(max(np.linalg.norm(F @ g + invariant["h"] - g) for F in invariant["Fseq"]))
    eps_mis = float(max(np.linalg.norm(F @ g + mismatch["h"] - g) for F in mismatch["Fseq"]))
    ub_inv = eps_inv * G / (1 - abar_bound); ub_mis = eps_mis * G / (1 - abar_bound)
    state_slice = slice(K - period, K + 1)
    max_target_inv = float(np.max(invariant["Et"][state_slice])); max_target_mis = float(np.max(mismatch["Et"][state_slice]))
    max_full_inv = float(max(np.linalg.norm(invariant["X"][:, j] - g) for j in range(K - period, K + 1)))
    max_full_mis = float(max(np.linalg.norm(mismatch["X"][:, j] - g) for j in range(K - period, K + 1)))
    final_cycle_start_inv = float(np.linalg.norm(invariant["X"][:, K] - g))
    final_cycle_start_mis = float(np.linalg.norm(mismatch["X"][:, K] - g))
    maxchi_inv = float(np.nanmax(dinv["chi"])); maxchi_mis = float(np.nanmax(dmis["chi"]))
    conservatism_ratio = float(ub_mis / final_cycle_start_mis)
    rows = [
        ("invariant", eps_inv, abar_bound, maxchi_inv, ub_inv, final_cycle_start_inv, max_full_inv, max_target_inv, last_active_time(invariant["U"])),
        ("mismatched", eps_mis, abar_bound, maxchi_mis, ub_mis, final_cycle_start_mis, max_full_mis, max_target_mis, "throughout"),
    ]
    write_csv(outdir / "fj_specialization_summary.csv", ["case", "epsilon_g", "uniform_cycle_bound", "max_homogeneous_chi", "theorem_cycle_ultimate_bound", "final_cycle_start_full_error", "max_full_error_final_cycle", "max_target_error_final_cycle", "activity"], rows)
    write_csv(outdir / "fj_cycle_gains.csv", ["cycle", "chi_hom_invariant", "actual_ratio_invariant", "chi_hom_mismatched", "actual_ratio_mismatched", "eta_norm_invariant", "eta_norm_mismatched", "cycle_start_error_invariant", "cycle_start_error_mismatched"], [(m, dinv["chi"][m], dinv["actual_ratio"][m], dmis["chi"][m], dmis["actual_ratio"][m], dinv["eta_norm"][m], dmis["eta_norm"][m], dinv["cycle_start_error"][m], dmis["cycle_start_error"][m]) for m in range(len(dinv["chi"]))])
    return locals()

# ---------------------------------------------------------------------------
# Eight-node budget and baseline experiments
# ---------------------------------------------------------------------------


def eight_node_network() -> list[np.ndarray]:
    return deterministic_periodic_network(8, NETWORK_SEED_8, size=4)


def budget_experiment(outdir: Path):
    W = eight_node_network()
    x0 = np.array([-1.6, -1.5, -0.4, 0.2, 0.4, -0.1, -0.3, -1.2])
    targets = [2, 3, 6, 7]
    g = 0.5 * np.ones(8); gamma = 0.1; K = 250
    results = {}; rows = []
    for r in range(1, 5):
        sim = simulate_single(W, x0, targets, g, r, gamma, K)
        results[r] = sim
        ts = settling_time(sim["Et"])
        rows.append((r, ts, float(np.sum(sim["U"]**2)), float(np.sum(sim["Et"][:-1])), float(np.mean(sim["card"])), mean_cardinality_to_settle(sim, ts)))
    write_csv(outdir / "budget_results.csv", ["r", "settling_time_to_horizon", "control_energy", "cumulative_target_error", "mean_cardinality_all_K", "mean_cardinality_to_settle"], rows)

    gamma_grid = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0]; gamma_rows = []
    for gamma_i in gamma_grid:
        sim = simulate_single(W, x0, targets, g, 2, gamma_i, K)
        ts = settling_time(sim["Et"])
        gamma_rows.append((gamma_i, ts, float(np.sum(sim["U"]**2)), float(np.sum(sim["Et"][:-1])), float(np.mean(sim["card"])), mean_cardinality_to_settle(sim, ts)))
    write_csv(outdir / "gamma_sweep.csv", ["gamma", "settling_time_to_horizon", "control_energy", "cumulative_target_error", "mean_cardinality_all_K", "mean_cardinality_to_settle"], gamma_rows)

    rng = np.random.default_rng(BUDGET_MC_SEED)
    mc_initials = rng.uniform(-1.5, 1.0, size=(BUDGET_MC_SAMPLES, 8))
    mc_raw = []
    mc_summary = []
    for r in range(1, 5):
        tsvals=[]; evals=[]; errvals=[]; cardvals=[]
        for q, x0q in enumerate(mc_initials):
            sim = simulate_single(W, x0q, targets, g, r, gamma, K)
            ts = settling_time(sim["Et"])
            mc_raw.append((q, r, np.nan if ts is None else ts, float(np.sum(sim["U"]**2)), float(np.sum(sim["Et"][:-1])), mean_cardinality_to_settle(sim, ts)))
            if ts is not None: tsvals.append(float(ts))
            evals.append(float(np.sum(sim["U"]**2))); errvals.append(float(np.sum(sim["Et"][:-1]))); cardvals.append(mean_cardinality_to_settle(sim, ts))
        success=len(tsvals)/BUDGET_MC_SAMPLES
        q25, med, q75 = (np.quantile(tsvals, [0.25,0.5,0.75]) if tsvals else [np.nan]*3)
        mc_summary.append((r, success, med, q25, q75, float(np.mean(evals)), float(np.std(evals,ddof=1)), float(np.mean(errvals)), float(np.std(errvals,ddof=1)), float(np.mean(cardvals)), float(np.std(cardvals,ddof=1))))
    write_csv(outdir / "budget_monte_carlo.csv", ["sample", "r", "settling_time", "control_energy", "cumulative_target_error", "mean_cardinality_to_settle"], mc_raw)
    write_csv(outdir / "budget_monte_carlo_summary.csv", ["r", "settle_fraction", "median_tset", "q25_tset", "q75_tset", "mean_energy", "sd_energy", "mean_error", "sd_error", "mean_card_to_settle", "sd_card_to_settle"], mc_summary)
    write_csv(outdir / "budget_monte_carlo_initial_conditions.csv", ["sample", *[f"x{i+1}" for i in range(8)]], [(q, *x) for q,x in enumerate(mc_initials)])
    write_matrix_sequence_csv(outdir / "W8_matrices.csv", W)
    return locals()

def simulate_fixed_support(Wseq, x0, targets, g, S, gamma, K):
    n = len(x0); X = np.zeros((n, K + 1)); X[:, 0] = x0; U = np.zeros(K); Et = np.zeros(K + 1)
    Et[0] = np.sum((x0[targets] - g[targets])**2); b = np.zeros(n); b[S] = 1.0
    for k in range(K):
        z = Wseq[k % len(Wseq)] @ X[:, k]; c = g - z
        u = 0.0 if len(S) == 0 else float(b @ c / (len(S) + gamma))
        X[:, k + 1] = z + b * u; U[k] = u; Et[k + 1] = np.sum((X[targets, k + 1] - g[targets])**2)
    return {"X": X, "U": U, "Et": Et}


def simulate_random_support(Wseq, x0, targets, g, r, gamma, K, rng):
    n = len(x0); X = np.zeros((n, K + 1)); X[:, 0] = x0; U = np.zeros(K); Et = np.zeros(K + 1)
    Et[0] = np.sum((x0[targets] - g[targets])**2)
    for k in range(K):
        z = Wseq[k % len(Wseq)] @ X[:, k]; c = g - z
        S = sorted(rng.choice(targets, size=r, replace=False).tolist()); b = np.zeros(n); b[S] = 1.0
        u = float(b @ c / (r + gamma)); X[:, k + 1] = z + b * u; U[k] = u
        Et[k + 1] = np.sum((X[targets, k + 1] - g[targets])**2)
    return {"X": X, "U": U, "Et": Et}


def baseline_experiment(outdir: Path, budget):
    W, x0, targets, g = budget["W"], budget["x0"], budget["targets"], budget["g"]
    r, gamma, K = 2, budget["gamma"], budget["K"]
    dyn = budget["results"][r]
    exact_r = simulate_single_exact_cardinality(W, x0, targets, g, r, gamma, K)
    c0 = g - W[0] @ x0
    static_sol = sparse_osaoc(c0, targets, r, gamma, activation_tol=0.0)
    staticS = static_sol.S
    sta = simulate_fixed_support(W, x0, targets, g, staticS, gamma, K)
    unconstrained = simulate_single(W, x0, targets, g, len(targets), gamma, K)

    random_err, random_energy, random_settle = [], [], []
    random_points = []
    for seed in range(1, RANDOM_BASELINE_SEEDS + 1):
        rr = simulate_random_support(W, x0, targets, g, r, gamma, K, np.random.default_rng(seed))
        er = float(np.sum(rr["Et"][:-1])); en = float(np.sum(rr["U"]**2)); st = settling_time(rr["Et"])
        random_err.append(er); random_energy.append(en); random_settle.append(np.nan if st is None else float(st)); random_points.append((seed, er, en, np.nan if st is None else st))
    valid = np.asarray(random_settle)[~np.isnan(random_settle)]
    mean_settle = float(np.mean(valid)) if len(valid) else np.nan; std_settle = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

    def metrics(sim):
        return float(np.sum(sim["Et"][:-1])), float(np.sum(sim["U"]**2)), settling_time(sim["Et"])
    rows = [
        ("adaptive_leq_r", *metrics(dyn)),
        ("adaptive_exact_r", *metrics(exact_r)),
        ("static_initial_optimal", *metrics(sta)),
        ("random_mean", float(np.mean(random_err)), float(np.mean(random_energy)), mean_settle),
        ("random_std", float(np.std(random_err, ddof=1)), float(np.std(random_energy, ddof=1)), std_settle),
        ("unconstrained_r_eq_T", *metrics(unconstrained)),
    ]
    write_csv(outdir / "baseline_results.csv", ["method", "cumulative_target_error", "control_energy", "settling_time_to_horizon"], rows)
    write_csv(outdir / "random_baseline_runs.csv", ["seed", "cumulative_target_error", "control_energy", "settling_time_to_horizon"], random_points)
    return locals()

# ---------------------------------------------------------------------------
# Competitive sparse game
# ---------------------------------------------------------------------------


def ten_node_network() -> list[np.ndarray]:
    return deterministic_periodic_network(10, NETWORK_SEED_10, size=4)


def player_cost(z, bs, u, m, Tms, goals, gammas):
    y = z.copy()
    for ell in range(len(bs)):
        y += bs[ell] * u[ell]
    inds = Tms[m]
    return float(np.sum((y[inds] - goals[m][inds])**2) + gammas[m] * u[m]**2)


def social_cost(z, bs, u, Tms, goals, gammas):
    return sum(player_cost(z, bs, u, m, Tms, goals, gammas) for m in range(len(bs)))


def fixed_support_ne(z, bs, goals, gammas):
    B = np.column_stack(bs); Gamma = np.diag(gammas); v = np.array([bs[m] @ goals[m] for m in range(len(bs))])
    S = B.T @ B + Gamma
    return np.linalg.solve(S, v - B.T @ z)


def potential_value(z, bs, u, goals, gammas):
    B = np.column_stack(bs); y = z + B @ u
    return float(y @ y - 2 * sum(u[m] * (bs[m] @ goals[m]) for m in range(len(bs))) + sum(gammas[m] * u[m]**2 for m in range(len(bs))))


def social_data(Tms, goals, gammas, n):
    dcount = np.zeros(n); gsum = np.zeros(n)
    for m, inds in enumerate(Tms):
        dcount[inds] += 1.0; gsum[inds] += goals[m][inds]
    return np.diag(dcount), gsum, np.diag(gammas)


def fixed_support_so(z, bs, Tms, goals, gammas):
    n = len(z); Dt, gsum, Gamma = social_data(Tms, goals, gammas, n); B = np.column_stack(bs)
    H = B.T @ Dt @ B + Gamma
    return np.linalg.solve(H, B.T @ (gsum - Dt @ z))


def exact_ibre(z, supports_by_player, Tms, goals, gammas):
    bestP = math.inf; bestbs = None; bestu = None
    for tup in itertools.product(*supports_by_player):
        bs = [b.copy() for b in tup]; u = fixed_support_ne(z, bs, goals, gammas); P = potential_value(z, bs, u, goals, gammas)
        if P < bestP - 1e-12:
            bestP, bestbs, bestu = P, bs, u.copy()
    return bestbs, bestu, bestP


def centralized_sparse(z, supports_by_player, Tms, goals, gammas):
    bestJ = math.inf; bestbs = None; bestu = None
    for tup in itertools.product(*supports_by_player):
        bs = [b.copy() for b in tup]; u = fixed_support_so(z, bs, Tms, goals, gammas); J = social_cost(z, bs, u, Tms, goals, gammas)
        if J < bestJ - 1e-12:
            bestJ, bestbs, bestu = J, bs, u.copy()
    return bestbs, bestu, bestJ


def sparse_best_response(z, bs, u, m, Tms, r, goals, gammas):
    y_other = z.copy()
    for ell in range(len(bs)):
        if ell != m:
            y_other += bs[ell] * u[ell]
    eff = np.zeros(len(z)); inds = Tms[m]; eff[inds] = goals[m][inds] - y_other[inds]
    return sparse_osaoc(eff, inds, r[m], gammas[m])


def target_exposure(bm, Tell):
    return float(np.sum(bm[Tell]))


def simulate_multiplayer(protocol, Wseq, x0, Tms, goals, r, gammas, K, supports_by_player):
    M = len(Tms); n = len(x0); X = np.zeros((n, K + 1)); X[:, 0] = x0
    Jstage = np.zeros(K); Energy = np.zeros(K); G12 = np.zeros(K); C2to1 = np.zeros(K); C1to2 = np.zeros(K)
    Bs, Us, Zs = [], [], []
    prevbs = [np.zeros(n) for _ in range(M)]; prevu = np.zeros(M)
    for k in range(K):
        z = Wseq[k % len(Wseq)] @ X[:, k]
        if protocol == "zero":
            bs, u = [], np.zeros(M)
            for m in range(M):
                eff = np.zeros(n); inds = Tms[m]; eff[inds] = goals[m][inds] - z[inds]
                sol = sparse_osaoc(eff, inds, r[m], gammas[m]); bs.append(sol.b); u[m] = sol.u
        elif protocol == "sequential":
            bs = [b.copy() for b in prevbs]; u = prevu.copy()
            for m in range(M):
                sol = sparse_best_response(z, bs, u, m, Tms, r, goals, gammas); bs[m] = sol.b; u[m] = sol.u
        elif protocol == "ibre":
            bs, u, _ = exact_ibre(z, supports_by_player, Tms, goals, gammas)
        elif protocol == "social":
            bs, u, _ = centralized_sparse(z, supports_by_player, Tms, goals, gammas)
        else:
            raise ValueError(protocol)
        y = z.copy()
        for m in range(M): y += bs[m] * u[m]
        X[:, k + 1] = y
        Jstage[k] = social_cost(z, bs, u, Tms, goals, gammas); Energy[k] = float(np.sum(u**2)); G12[k] = bs[0] @ bs[1]
        C2to1[k] = target_exposure(bs[1], Tms[0]); C1to2[k] = target_exposure(bs[0], Tms[1])
        Bs.append([b.copy() for b in bs]); Us.append(u.copy()); Zs.append(z.copy()); prevbs = [b.copy() for b in bs]; prevu = u.copy()
    return {"X": X, "Jstage": Jstage, "Energy": Energy, "G12": G12, "C2to1": C2to1, "C1to2": C1to2, "Bs": Bs, "Us": Us, "Zs": Zs}


def welfare_decomposition(z, bsNE, uNE, supports_by_player, Tms, goals, gammas):
    uSame = fixed_support_so(z, bsNE, Tms, goals, gammas)
    JNE = social_cost(z, bsNE, uNE, Tms, goals, gammas); Jsame = social_cost(z, bsNE, uSame, Tms, goals, gammas)
    _, _, Jglobal = centralized_sparse(z, supports_by_player, Tms, goals, gammas)
    dact = JNE - Jsame; dsup = Jsame - Jglobal
    return dact, dsup, dact + dsup


def competitive_experiment(outdir: Path):
    W = ten_node_network()
    x0 = np.array([-0.85, 0.75, -0.25, 0.95, -1.10, 0.35, 0.80, -0.65, 0.15, -0.45])
    T1 = [0, 1, 5, 6, 8]; T2 = [2, 3, 4, 6, 9]; Tms = [T1, T2]
    goals = [0.5 * np.ones(10), -0.5 * np.ones(10)]; r = [2, 2]; gammas = [0.5, 0.5]; K = 80
    supports_by_player = [support_vectors(10, T1, r[0]), support_vectors(10, T2, r[1])]
    zero = simulate_multiplayer("zero", W, x0, Tms, goals, r, gammas, K, supports_by_player)
    seq = simulate_multiplayer("sequential", W, x0, Tms, goals, r, gammas, K, supports_by_player)
    ibre = simulate_multiplayer("ibre", W, x0, Tms, goals, r, gammas, K, supports_by_player)
    social = simulate_multiplayer("social", W, x0, Tms, goals, r, gammas, K, supports_by_player)
    methods = [("zero_opponent", zero), ("sequential_one_sweep", seq), ("exact_IBRE", ibre), ("centralized_sparse", social)]
    summary_rows = [(name, float(np.sum(rr["Jstage"])), float(np.sum(rr["Energy"])), float(np.mean(rr["G12"]))) for name, rr in methods]
    write_csv(outdir / "competitive_summary.csv", ["method", "cumulative_social_cost", "cumulative_control_energy", "mean_G12"], summary_rows)
    dact = np.zeros(K); dsup = np.zeros(K); dsp = np.zeros(K); max_ne_residual = 0.0
    for k in range(K):
        z, bs, u = ibre["Zs"][k], ibre["Bs"][k], ibre["Us"][k]
        dact[k], dsup[k], dsp[k] = welfare_decomposition(z, bs, u, supports_by_player, Tms, goals, gammas)
        for m in range(2):
            current = player_cost(z, bs, u, m, Tms, goals, gammas); br = sparse_best_response(z, bs, u, m, Tms, r, goals, gammas)
            bs2 = [b.copy() for b in bs]; u2 = u.copy(); bs2[m] = br.b; u2[m] = br.u
            best = player_cost(z, bs2, u2, m, Tms, goals, gammas); max_ne_residual = max(max_ne_residual, max(current - best, 0.0))
    write_csv(outdir / "welfare_decomposition.csv", ["k", "Delta_act", "Delta_sup", "Delta_sp", "G12", "C_2to1", "C_1to2"], [(k, dact[k], dsup[k], dsp[k], ibre["G12"][k], ibre["C2to1"][k], ibre["C1to2"][k]) for k in range(K)])
    write_matrix_sequence_csv(outdir / "W10_matrices.csv", W)

    mc_rows=[]
    for q in range(COMP_MC_SAMPLES):
        seed=COMP_MC_BASE_SEED+q
        Wq=deterministic_periodic_network(10, seed, size=4)
        zq=simulate_multiplayer("zero", Wq, x0, Tms, goals, r, gammas, K, supports_by_player)
        iq=simulate_multiplayer("ibre", Wq, x0, Tms, goals, r, gammas, K, supports_by_player)
        sq=simulate_multiplayer("social", Wq, x0, Tms, goals, r, gammas, K, supports_by_player)
        da=ds=0.0
        for k in range(K):
            a,b,_=welfare_decomposition(iq["Zs"][k], iq["Bs"][k], iq["Us"][k], supports_by_player, Tms, goals, gammas)
            da += a; ds += b
        mc_rows.append((seed, float(np.sum(zq["Jstage"])), float(np.sum(iq["Jstage"])), float(np.sum(sq["Jstage"])), float(np.sum(zq["Energy"])), float(np.sum(iq["Energy"])), float(np.sum(sq["Energy"])), da, ds, da+ds))
    write_csv(outdir / "competitive_monte_carlo.csv", ["network_seed", "zero_cum_social", "ibre_cum_social", "central_cum_social", "zero_energy", "ibre_energy", "central_energy", "cum_Delta_act", "cum_Delta_sup", "cum_Delta_sp"], mc_rows)
    arr=np.asarray([row[1:] for row in mc_rows], dtype=float)
    mc_mean=np.mean(arr,axis=0); mc_sd=np.std(arr,axis=0,ddof=1)
    ibre_worse_count=sum(row[2] > row[1] for row in mc_rows)
    return locals()

# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------


def write_resulttodo_tex(outdir, sortres, five, fj, budget, baseline, comp):
    with (outdir / "RESULTTODO_values.tex").open("w", encoding="utf-8") as io:
        io.write("% Auto-generated by strategic_switching_experiments.py\n% Do not edit by hand; rerun the script instead.\n\n")
        def cmd(name, value): io.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")
        cmd("NumSortingTests", sortres["total_tests"]); cmd("MaxSortingError", fmt_sci_tex(sortres["maxerr"])); cmd("SettleTol", "10^{-6}"); cmd("CycleZeroTol", "10^{-9}"); cmd("ActivationValueTol", "10^{-24}")
        cmd("BudgetGamma", fmt(budget["gamma"], 2)); cmd("BudgetHorizon", budget["K"]); cmd("GammaGrid", "0.05,0.10,0.25,0.50,1,2,5"); cmd("NumRandomSeeds", RANDOM_BASELINE_SEEDS); cmd("BudgetMCSamples", BUDGET_MC_SAMPLES); cmd("CompetitiveMCSamples", COMP_MC_SAMPLES)
        cmd("OpenConsensusLimit", fmt(np.mean(five["open_cons"][:, -1]))); tail = five["open_osc"][:, -100:]; cmd("OpenOscMin", fmt(np.min(tail))); cmd("OpenOscMax", fmt(np.max(tail)))
        cmd("LastActiveConsensus", last_active_time(five["cons"]["U"])); cmd("LastActiveOscillatory", last_active_time(five["osc"]["U"])); cmd("AObsConsensus", fmt(five["aobs_cons"])); cmd("AObsOscillatory", fmt(five["aobs_osc"]))
        cmd("FJBeta", fmt(fj["beta"])); cmd("FJUniformCycleBound", fmt(fj["abar_bound"])); cmd("FJEpsilonInvariant", fmt(fj["eps_inv"])); cmd("FJEpsilonMismatch", fmt(fj["eps_mis"])); cmd("FJAObsInvariant", fmt(np.nanmax(fj["dinv"]["chi"]))); cmd("FJAObsMismatch", fmt(np.nanmax(fj["dmis"]["chi"])))
        cmd("FJUltimateInvariant", fmt(fj["ub_inv"])); cmd("FJUltimateMismatch", fmt(fj["ub_mis"])); cmd("FJFinalCycleInvariant", fmt(fj["final_cycle_start_inv"])); cmd("FJFinalCycleMismatch", fmt(fj["final_cycle_start_mis"])); cmd("FJMaxTargetInvariant", fmt(fj["max_target_inv"])); cmd("FJMaxTargetMismatch", fmt(fj["max_target_mis"])); cmd("FJLastActiveInvariant", last_active_time(fj["invariant"]["U"])); cmd("FJLastActiveMismatch", "throughout"); cmd("FJBoundConservatism", fmt(fj["conservatism_ratio"],2))
        for idx, row in enumerate(budget["rows"]):
            suffix = ["One", "Two", "Three", "Four"][idx]; cmd(f"BudgetSettle{suffix}", fmt(row[1])); cmd(f"BudgetEnergy{suffix}", fmt(row[2])); cmd(f"BudgetError{suffix}", fmt(row[3])); cmd(f"BudgetCard{suffix}", fmt(row[4], 3)); cmd(f"BudgetCardPre{suffix}", fmt(row[5],3))
        prefixes=["Adaptive","AdaptiveExact","Static","RandomMean","RandomStd","Unconstrained"]
        for prefix,row in zip(prefixes, baseline["rows"]): cmd(f"{prefix}BaselineError",fmt(row[1])); cmd(f"{prefix}BaselineEnergy",fmt(row[2])); cmd(f"{prefix}BaselineSettle",fmt(row[3],2))
        cmd("TargetSetOne", "\\{1,2,6,7,9\\}"); cmd("TargetSetTwo", "\\{3,4,5,7,10\\}")
        for pref, row in zip(["Zero", "Sequential", "IBRE", "Central"], comp["summary_rows"]):
            cmd(f"{pref}CumSocial", fmt(row[1])); cmd(f"{pref}CumEnergy", fmt(row[2])); cmd(f"{pref}MeanG", fmt(row[3], 4))
        cmd("MaxNEResidual", fmt_sci_tex(comp["max_ne_residual"])); cmd("CumDeltaAct", fmt(np.sum(comp["dact"]))); cmd("CumDeltaSup", fmt(np.sum(comp["dsup"]))); cmd("CumDeltaSp", fmt(np.sum(comp["dsp"]))); cmd("MCIBREWorseCount", comp["ibre_worse_count"])

def write_generated_tables(outdir, sortres, fj, budget, baseline, comp):
    blocks=[]; names=[]
    def add(name, lines): names.append(name); blocks.append("\n".join(lines)+"\n")
    lines=[r"\begin{table}[t]",r"\caption{Validation and timing of exact sparse support selection.}",r"\label{tab:sorting_validation}",r"\centering",r"\resizebox{\columnwidth}{!}{%",r"\begin{tabular}{r r r r r}",r"\hline",r"\(N\) & \(r\) & full sort (ms) & enum. (ms) & max. \(\varepsilon_V\)\\",r"\hline"]
    for row in sortres["validation_rows"]: lines.append(f"{row[0]} & {row[1]} & {row[2]:.6f} & {row[3]:.6f} & {row[4]:.2e}\\\\")
    lines += [r"\hline",r"\end{tabular}",r"}%",r"\end{table}"]; add("table_sorting_validation.tex",lines)

    lines=[r"\begin{table}[t]",r"\caption{Friedkin--Johnsen specialization. The reported \(\chi_m\) is the homogeneous cycle gain, not the full affine cycle-error ratio.}",r"\label{tab:fj_specialization}",r"\centering",r"\begin{tabular}{l r r}",r"\hline",r"Metric & invariant & mismatched\\",r"\hline",rf"\(\varepsilon_g\) & {fmt(fj['eps_inv'])} & {fmt(fj['eps_mis'])}\\",rf"\(\max_m\chi_m\) & {fmt(np.nanmax(fj['dinv']['chi']))} & {fmt(np.nanmax(fj['dmis']['chi']))}\\",f"cycle UUB & {fmt(fj['ub_inv'])} & {fmt(fj['ub_mis'])}\\\\",rf"final cycle-start \(\|e\|_2\) & {fmt(fj['final_cycle_start_inv'])} & {fmt(fj['final_cycle_start_mis'])}\\",rf"max. final-cycle \(E_{{\mathcal T}}\) & {fmt(fj['max_target_inv'])} & {fmt(fj['max_target_mis'])}\\",rf"activity & last \(k={last_active_time(fj['invariant']['U'])}\) & throughout horizon\\",r"\hline",r"\end{tabular}",r"\end{table}"]; add("table_fj_specialization.tex",lines)

    rows=budget["rows"]
    lines=[r"\begin{table}[t]",r"\caption{Effect of the cardinality budget in the deterministic eight-node benchmark (\(\gamma=0.1\), \(K=250\)).}",r"\label{tab:budget_results}",r"\centering",r"\resizebox{\columnwidth}{!}{%",r"\begin{tabular}{c|cccc}",r"\hline",r"\(r\) & 1 & 2 & 3 & 4\\",r"\hline",r"\(t_{\rm set}\) & "+" & ".join(fmt(x[1]) for x in rows)+r"\\",r"\(\mathcal E_u\) & "+" & ".join(fmt(x[2]) for x in rows)+r"\\",r"\(\sum_k E_{\mathcal T}(k)\) & "+" & ".join(fmt(x[3]) for x in rows)+r"\\",r"mean \(s(k),\ k<t_{\rm set}\) & "+" & ".join(fmt(x[5],3) for x in rows)+r"\\",r"\hline",r"\end{tabular}",r"}%",r"\end{table}"]; add("table_budget_results.tex",lines)

    lines=[r"\begin{table}[t]",r"\caption{Monte-Carlo budget sensitivity over 50 initial conditions, \(x_i(0)\sim\mathcal U[-1.5,1]\).}",r"\label{tab:budget_mc_results}",r"\centering",r"\resizebox{\columnwidth}{!}{%",r"\begin{tabular}{c r r r}",r"\hline",r"\(r\) & median \(t_{\rm set}\) [IQR] & mean \(\sum E_{\mathcal T}\) & mean \(\mathcal E_u\)\\",r"\hline"]
    for row in budget["mc_summary"]: lines.append(f"{row[0]} & {row[2]:.0f} [{row[3]:.0f},{row[4]:.0f}] & {row[7]:.3f} $\\pm$ {row[8]:.3f} & {row[5]:.3f} $\\pm$ {row[6]:.3f}\\\\")
    lines += [r"\hline",r"\end{tabular}",r"}%",r"\end{table}"]; add("table_budget_monte_carlo.tex",lines)

    lines=[r"\begin{table}[t]",r"\caption{Regularization sensitivity for \(r=2\) in the deterministic eight-node benchmark.}",r"\label{tab:gamma_sweep_results}",r"\centering",r"\resizebox{\columnwidth}{!}{%",r"\begin{tabular}{c r r r r}",r"\hline",r"\(\gamma\) & \(t_{\rm set}\) & \(\mathcal E_u\) & \(\sum E_{\mathcal T}\) & mean \(s\), pre-settle\\",r"\hline"]
    for row in budget["gamma_rows"]: lines.append(f"{fmt(row[0],2)} & {fmt(row[1])} & {fmt(row[2])} & {fmt(row[3])} & {fmt(row[5],3)}\\\\")
    lines += [r"\hline",r"\end{tabular}",r"}%",r"\end{table}"]; add("table_gamma_sweep.tex",lines)

    labels=[r"Adaptive \(|S|\le r\)",r"Adaptive \(|S|=r\)","Static initial optimum",r"Random exact-\(r\), mean $\pm$ sd",r"Unconstrained \(r=|\mathcal T|\)"]
    br=baseline["rows"]
    table_rows=[br[0],br[1],br[2],("random",br[3][1],br[3][2],br[3][3]),br[5]]
    lines=[r"\begin{table}[t]",r"\caption{Adaptive support selection and baselines (nominal eight-node trajectory, \(r=2\) where applicable).}",r"\label{tab:baseline_results}",r"\centering",r"\resizebox{\columnwidth}{!}{%",r"\begin{tabular}{l r r r}",r"\hline",r"Method & cumulative target error & control energy & \(t_{\rm set}\)\\",r"\hline"]
    for idx,(lab,row) in enumerate(zip(labels,table_rows)):
        if idx==3: lines.append(f"{lab} & {row[1]:.3f} $\\pm$ {br[4][1]:.3f} & {row[2]:.3f} $\\pm$ {br[4][2]:.3f} & {row[3]:.1f} $\\pm$ {br[4][3]:.1f}\\\\")
        else: lines.append(f"{lab} & {fmt(row[1])} & {fmt(row[2])} & {fmt(row[3],2)}\\\\")
    lines += [r"\hline",r"\end{tabular}",r"}%",r"\end{table}"]; add("table_baselines.tex",lines)

    labels=["Zero-opponent myopic","One-sweep sequential BR","Exact IBRE","Centralized sparse"]
    lines=[r"\begin{table}[t]",r"\caption{Competitive one-step benchmarks for the deterministic ten-node network.}",r"\label{tab:competitive_summary}",r"\centering",r"\resizebox{\columnwidth}{!}{%",r"\begin{tabular}{l r r r}",r"\hline",r"Method & cum. social cost & cum. energy & mean \(G_{12}\)\\",r"\hline"]
    for lab,row in zip(labels,comp["summary_rows"]): lines.append(f"{lab} & {fmt(row[1])} & {fmt(row[2])} & {fmt(row[3],4)}\\\\")
    lines += [r"\hline",r"\end{tabular}",r"}%",r"\end{table}"]; add("table_competitive_summary.tex",lines)

    lines=[r"\begin{table}[t]",r"\caption{Same-state welfare decomposition along the exact-IBRE trajectory.}",r"\label{tab:welfare_summary}",r"\centering",r"\begin{tabular}{l r}",r"\hline",r"Quantity & cumulative value\\",r"\hline",rf"\(\sum_k\Delta_{{\rm act}}(k)\) & {fmt(np.sum(comp['dact']))}\\",rf"\(\sum_k\Delta_{{\rm sup}}(k)\) & {fmt(np.sum(comp['dsup']))}\\",rf"\(\sum_k\Delta_{{\rm sp}}(k)\) & {fmt(np.sum(comp['dsp']))}\\",f"max. unilateral IBRE residual & {comp['max_ne_residual']:.3e}\\\\",r"\hline",r"\end{tabular}",r"\end{table}"]; add("table_welfare_summary.tex",lines)

    mm=comp["mc_mean"]; sd=comp["mc_sd"]
    lines=[r"\begin{table}[t]",r"\caption{Competitive sensitivity over ten independently generated periodic networks; entries are mean $\pm$ standard deviation over network realizations.}",r"\label{tab:competitive_mc}",r"\centering",r"\resizebox{\columnwidth}{!}{%",r"\begin{tabular}{l r}",r"\hline",r"Metric & mean $\pm$ sd\\",r"\hline",f"zero-opponent cumulative social cost & {mm[0]:.3f} $\\pm$ {sd[0]:.3f}\\\\",f"exact-IBRE cumulative social cost & {mm[1]:.3f} $\\pm$ {sd[1]:.3f}\\\\",f"centralized cumulative social cost & {mm[2]:.3f} $\\pm$ {sd[2]:.3f}\\\\",rf"\(\sum\Delta_{{\rm act}}\) & {mm[6]:.3f} $\pm$ {sd[6]:.3f}\\",rf"\(\sum\Delta_{{\rm sup}}\) & {mm[7]:.3f} $\pm$ {sd[7]:.3f}\\",r"\hline",r"\end{tabular}",r"}%",r"\end{table}"]; add("table_competitive_monte_carlo.tex",lines)

    with (outdir / "generated_tables.tex").open("w",encoding="utf-8") as f: f.write("% Auto-generated numerical tables\n\n"+"\n".join(blocks))
    for name,block in zip(names,blocks): (outdir/name).write_text("% Auto-generated table fragment\n"+block,encoding="utf-8")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_figures(outdir, five, fj, budget, baseline, comp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Figures skipped: matplotlib unavailable ({exc})")
        return

    # Five-node composite
    fig, ax = plt.subplots(3, 2, figsize=(8.4, 9.0), constrained_layout=True)
    for row in five["cons"]["X"]: ax[0,0].plot(range(five["K"]+1), row)
    ax[0,0].axhline(0.2, linestyle="--"); ax[0,0].set(title="consensus order", xlabel="k", ylabel="state")
    for row in five["osc"]["X"]: ax[0,1].plot(range(five["K"]+1), row)
    ax[0,1].axhline(0.2, linestyle="--"); ax[0,1].set(title="oscillatory order", xlabel="k", ylabel="state")
    ax[1,0].semilogy(range(five["K"]+1), np.maximum(five["cons"]["Et"],1e-18), label="cons."); ax[1,0].semilogy(range(five["K"]+1), np.maximum(five["osc"]["Et"],1e-18), label="osc."); ax[1,0].set(xlabel="k", ylabel="target error"); ax[1,0].legend()
    ax[1,1].plot(range(five["K"]), five["cons"]["U"], label="cons."); ax[1,1].plot(range(five["K"]), five["osc"]["U"], label="osc."); ax[1,1].set(xlabel="k", ylabel="u(k)"); ax[1,1].legend()
    ax[2,0].step(range(five["K"]), five["cons"]["card"], where="pre", label="cons."); ax[2,0].step(range(five["K"]), five["osc"]["card"], where="pre", label="osc."); ax[2,0].set(xlabel="k", ylabel="s(k)"); ax[2,0].legend()
    ax[2,1].plot(range(len(five["chi_cons"])), five["chi_cons"], marker="o", label="cons."); ax[2,1].plot(range(len(five["chi_osc"])), five["chi_osc"], marker="s", label="osc."); ax[2,1].axhline(1.0, linestyle="--"); ax[2,1].set(xlabel="cycle m", ylabel=r"$\chi_m$"); ax[2,1].legend()
    for ext in ("pdf", "png"): fig.savefig(outdir / f"Figure_periodic_tracking.{ext}", dpi=220); plt.close(fig) if ext == "png" else None

    # FJ composite
    fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.5), constrained_layout=True)
    for row in fj["invariant"]["X"]: ax[0,0].plot(range(fj["K"]+1), row)
    ax[0,0].axhline(0.2, linestyle="--"); ax[0,0].set(title="FJ: invariant target", xlabel="k", ylabel="state")
    for row in fj["mismatch"]["X"]: ax[0,1].plot(range(fj["K"]+1), row)
    ax[0,1].axhline(0.2, linestyle="--"); ax[0,1].set(title="FJ: mismatched prejudices", xlabel="k", ylabel="state")
    ax[1,0].semilogy(range(fj["K"]+1), np.maximum(fj["invariant"]["Et"],1e-18), label="invariant"); ax[1,0].semilogy(range(fj["K"]+1), np.maximum(fj["mismatch"]["Et"],1e-18), label="mismatched"); ax[1,0].set(xlabel="k", ylabel="target error"); ax[1,0].legend()
    ax[1,1].plot(range(fj["K"]), fj["invariant"]["U"], label="invariant"); ax[1,1].plot(range(fj["K"]), fj["mismatch"]["U"], label="mismatched"); ax[1,1].set(xlabel="k", ylabel="u(k)"); ax[1,1].legend()
    fig.savefig(outdir / "Figure_fj_affine_tracking.pdf"); fig.savefig(outdir / "Figure_fj_affine_tracking.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    for r in range(1,5): ax.semilogy(range(budget["K"]+1), np.maximum(budget["results"][r]["Et"],1e-18), label=f"r={r}")
    ax.set(xlabel="k", ylabel="target error"); ax.legend(); fig.savefig(outdir / "Figure_budget_target_error.pdf"); fig.savefig(outdir / "Figure_budget_target_error.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    gam = np.array([row[0] for row in budget["gamma_rows"]]); err = np.array([row[3] for row in budget["gamma_rows"]]); en = np.array([row[2] for row in budget["gamma_rows"]])
    ax.plot(gam, err, marker="o", label="target error"); ax.plot(gam, en, marker="s", label="control energy"); ax.set_xscale("log"); ax.set(xlabel=r"$\gamma$", ylabel="cumulative metric"); ax.legend(); fig.savefig(outdir / "Figure_gamma_sweep.pdf"); fig.savefig(outdir / "Figure_gamma_sweep.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    labels = ["adaptive <=r", "adaptive =r", "static", "random mean", "unconstrained"]
    rr = baseline["rows"]
    pts = [rr[0], rr[1], rr[2], rr[3], rr[5]]
    offsets = [(4,4), (4,-10), (4,4), (4,4), (4,4)]
    for lab, row, off in zip(labels, pts, offsets):
        ax.scatter(row[2], row[1], s=34)
        ax.annotate(lab, (row[2], row[1]), xytext=off, textcoords="offset points", fontsize=7)
    ax.scatter(baseline["random_energy"], baseline["random_err"], s=6, alpha=0.18)
    ax.set(xlabel="cumulative control energy", ylabel="cumulative target error")
    fig.savefig(outdir / "Figure_baseline_error.pdf"); fig.savefig(outdir / "Figure_baseline_error.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.2), constrained_layout=True)
    for j, (title, rr) in enumerate([("zero-opponent", comp["zero"]), ("one-sweep sequential", comp["seq"]), ("exact IBRE", comp["ibre"])]):
        for row in rr["X"]: ax[0,j].plot(range(comp["K"]+1), row)
        ax[0,j].set(title=title, xlabel="k", ylabel="state")
        for player, marker in [(0,"o"),(1,"s")]:
            tt, nn = [], []
            for k, bs in enumerate(rr["Bs"]):
                inds = support_indices_1based(bs[player]); tt.extend([k]*len(inds)); nn.extend(inds.tolist())
            ax[1,j].scatter(tt, nn, marker=marker, s=10, label=f"P{player+1}")
        ax[1,j].set(xlabel="k", ylabel="node"); ax[1,j].legend()
    fig.savefig(outdir / "Figure_competitive_protocols.pdf"); fig.savefig(outdir / "Figure_competitive_protocols.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(6.3, 5.6), constrained_layout=True)
    k = np.arange(comp["K"]); ax[0].step(k, comp["ibre"]["G12"], where="pre", label=r"$G_{12}$"); ax[0].step(k, comp["ibre"]["C2to1"], where="pre", label=r"$C_{2\to1}$"); ax[0].step(k, comp["ibre"]["C1to2"], where="pre", label=r"$C_{1\to2}$"); ax[0].set(xlabel="k", ylabel="count"); ax[0].legend()
    ax[1].plot(k, comp["dact"], label=r"$\Delta_{\rm act}$"); ax[1].plot(k, comp["dsup"], label=r"$\Delta_{\rm sup}$"); ax[1].plot(k, comp["dsp"], label=r"$\Delta_{\rm sp}$"); ax[1].set(xlabel="k", ylabel="welfare loss"); ax[1].legend()
    fig.savefig(outdir / "Figure_competitive_welfare.pdf"); fig.savefig(outdir / "Figure_competitive_welfare.png", dpi=220); plt.close(fig)


def write_provenance(outdir, five, fj, budget, comp):
    import numpy
    try:
        import matplotlib
        mplver = matplotlib.__version__
    except Exception:
        mplver = "not installed"
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu_model=line.split(":",1)[1].strip(); break
    except Exception:
        pass
    with (outdir / "network_provenance.txt").open("w", encoding="utf-8") as f:
        f.write(f"Python version: {platform.python_version()}\nNumPy version: {numpy.__version__}\nMatplotlib version: {mplver}\nPlatform: {platform.platform()}\nCPU: {cpu_model}\n")
        f.write(f"Global sorting-test seed: {GLOBAL_SEED}\nSorting timing protocol: c~N(0,I), gamma=0.5, r=5, 20 repetitions per table row; median wall-clock perf_counter time.\n\n")
        f.write("Five-node matrices: Wang et al. representative matrices; exact matrices in W5_matrices.csv and in the main manuscript.\n")
        f.write("Five-node order 1: W1,W2,W3,W4,W5\nFive-node order 2: W1,W5,W4,W3,W2\n")
        f.write("Five-node T={1,3,5}, g=0.2*ones(5), r=3, gamma=0.5\n\n")
        f.write(f"FJ lambda: {fj['lam'].tolist()}\nFJ invariant prejudice: {fj['s_invariant'].tolist()}\nFJ mismatched prejudice: {fj['s_mismatch'].tolist()}\n")
        f.write("FJ direct actuation: bu is applied after the free response and is not premultiplied by Lambda.\n")
        f.write(f"FJ beta={fj['beta']:.17g}; product norm bound={fj['abar_bound']:.17g}\n\n")
        f.write(f"Eight-node periodic benchmark: deterministic Python generator, seed={NETWORK_SEED_8}, 4 phases.\n")
        f.write(f"Eight-node x0={budget['x0'].tolist()}\nEight-node T={{3,4,7,8}}, gamma={budget['gamma']}, horizon K={budget['K']}\n")
        f.write(f"Budget Monte Carlo: {BUDGET_MC_SAMPLES} initial states, seed={BUDGET_MC_SEED}, iid Uniform[-1.5,1.0].\n")
        f.write("Realized matrices are stored in W8_matrices.csv.\n\n")
        f.write(f"Ten-node periodic benchmark: deterministic Python generator, seed={NETWORK_SEED_10}, 4 phases.\n")
        f.write(f"Ten-node x0={comp['x0'].tolist()}\nT1={{1,2,6,7,9}}; T2={{3,4,5,7,10}}\n")
        f.write("goals=+/-0.5; r=(2,2); gammas=(0.5,0.5); horizon K=80\nRealized matrices are stored in W10_matrices.csv.\n")
        f.write(f"Competitive Monte Carlo: {COMP_MC_SAMPLES} periodic networks with seeds {COMP_MC_BASE_SEED},...,{COMP_MC_BASE_SEED+COMP_MC_SAMPLES-1}.\n\n")
        f.write(f"Finite-horizon settling threshold epsilon={SETTLE_EPS}.\n")
        f.write(f"Support objective-improvement dead-band: {ACTIVATION_VALUE_TOL}.\n")
        f.write(f"Cycle diagnostic zero threshold: {CYCLE_ZERO_TOL}.\nRandom-baseline seeds: 1,...,{RANDOM_BASELINE_SEEDS}.\n")

def matrix_to_latex(W: np.ndarray, digits: int = 6) -> str:
    rows = [" & ".join(f"{float(x):.{digits}f}" for x in row) + r"\\" for row in W]
    return "\\begin{bmatrix}\n" + "\n".join(rows) + "\n\\end{bmatrix}"


def write_reproducibility_extras(outdir: Path, five, budget, comp) -> None:
    """Write manuscript matrix fragment, supplementary TeX, README, and manifest."""
    # Compact five-node matrix fragment used directly by the manuscript.
    W = five["W"]
    mats = [matrix_to_latex(M, 1) for M in W]
    fragment = r"""\begin{figure*}[t]
\centering
\begingroup
\setlength{\arraycolsep}{3pt}
\small
\[
\begin{array}{c@{\qquad}c@{\qquad}c}
W_1=%s & W_2=%s & W_3=%s\\[2mm]
W_4=%s & W_5=%s &
\end{array}
\]
\endgroup
\caption{Five representative stochastic matrices from the Wang et al. benchmark used in the DeGroot and FJ numerical experiments.}
\label{fig:five_node_matrices}
\end{figure*}
""" % tuple(mats)
    (outdir / "five_node_matrices_fragment.tex").write_text(fragment, encoding="utf-8")

    # Supplementary network-matrix source. Exact double-precision values remain
    # in the CSV files; larger matrices are rounded here only for readability.
    W8 = budget["W"]; W10 = comp["W"]
    parts = [r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,graphicx,pdflscape,geometry}
\geometry{margin=18mm}
\title{Supplementary Network Matrices\\Sparse One-Step-Ahead Optimal Control of Time-Varying Affine Opinion Networks}
\author{Gabriel Gentil and Amit Bhaya}
\date{}
\begin{document}
\maketitle
\noindent This supplementary document prints the realized matrices used in the numerical experiments. Entries for the eight- and ten-node matrices are rounded here to six decimal places for readability; the accompanying files \texttt{W8\_matrices.csv} and \texttt{W10\_matrices.csv} contain the full double-precision values used by the Python script. The five-node matrices are exact half-weight representatives and are also printed in the main paper.
\section*{S1. Five-node Wang representative matrices}
"""]
    for j,M in enumerate(W,1): parts.append(rf"\[W_{j}={matrix_to_latex(M,1)}\]")
    parts.append(r"\clearpage\begin{landscape}\section*{S2. Eight-node four-phase benchmark}")
    for j,M in enumerate(W8,1):
        parts += [rf"\subsection*{{Phase {j}}}", rf"\resizebox{{0.96\linewidth}}{{!}}{{$W^{{(8)}}_{j}={matrix_to_latex(M,6)}$}}"]
    parts.append(r"\clearpage\section*{S3. Ten-node four-phase competitive benchmark}")
    for j,M in enumerate(W10,1):
        parts += [rf"\subsection*{{Phase {j}}}", rf"\resizebox{{0.96\linewidth}}{{!}}{{$W^{{(10)}}_{j}={matrix_to_latex(M,6)}$}}"]
    parts.append(r"\end{landscape}\end{document}")
    (outdir / "Supplementary_Network_Matrices.tex").write_text("\n\n".join(parts), encoding="utf-8")

    readme = f"""Reproducibility package for the numerical experiments

Run from the manuscript directory:
    python strategic_switching_experiments.py

Required packages:
    numpy
    matplotlib

The default output directory is strategic_switching_outputs/.

Numerical safeguards:
    objective-improvement dead-band = {ACTIVATION_VALUE_TOL:g}
    cycle diagnostic zero threshold = {CYCLE_ZERO_TOL:g}
    finite-horizon settling threshold = {SETTLE_EPS:g}

Monte Carlo:
    budget initial-condition samples = {BUDGET_MC_SAMPLES}, seed {BUDGET_MC_SEED}
    competitive network realizations = {COMP_MC_SAMPLES}, seeds {COMP_MC_BASE_SEED}..{COMP_MC_BASE_SEED+COMP_MC_SAMPLES-1}
    random-support baseline seeds = 1..{RANDOM_BASELINE_SEEDS}

Exact realized network matrices are W5_matrices.csv, W8_matrices.csv,
and W10_matrices.csv.  Supplementary_Network_Matrices.tex prints them
in human-readable form (the 8/10-node printed values are rounded; the CSVs
are authoritative).
"""
    (outdir / "README_REPRODUCIBILITY.txt").write_text(readme, encoding="utf-8")

    # Write manifest last, excluding itself until the final line.
    names = sorted(x.name for x in outdir.iterdir() if x.is_file() and x.name != "MANIFEST.txt")
    (outdir / "MANIFEST.txt").write_text("\n".join(names + ["MANIFEST.txt"]) + "\n", encoding="utf-8")


def main():
    args = parse_args(); outdir = Path(args.out).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    print("Output directory:", outdir)
    print("[1/7] Sorting correctness and timing"); sortres = sorting_validation(outdir, quick=args.quick); print(f"  correctness tests: {sortres['total_tests']}; max discrepancy={sortres['maxerr']:.3e}")
    print("[2/7] Five-node DeGroot periodic benchmark"); five = five_node_experiment(outdir); print(f"  open-loop consensus limit={np.mean(five['open_cons'][:,-1]):.6f}; a_obs={five['aobs_cons']:.6f}/{five['aobs_osc']:.6f}")
    print("[3/7] FJ invariant and mismatched cases"); fj = fj_specialization_experiment(outdir); print(f"  product-norm bound={fj['abar_bound']:.6f}; epsilon_g={fj['eps_inv']:.6f}/{fj['eps_mis']:.6f}")
    print("[4/7] Eight-node budget, gamma, and initial-condition sweeps"); budget = budget_experiment(outdir); [print(f"  r={row[0]}: t_set={row[1]}, energy={row[2]:.6f}, cum.error={row[3]:.6f}, mean s(pre)={row[5]:.3f}") for row in budget['rows']]
    print("[5/7] Selection baselines"); baseline = baseline_experiment(outdir, budget); print("  static support (1-based):", [i+1 for i in baseline['staticS']])
    print("[6/7] Competitive sparse game and network-realization sweep"); comp = competitive_experiment(outdir); [print(f"  {row[0]:22s} J={row[1]:.6f}, energy={row[2]:.6f}, mean G12={row[3]:.4f}") for row in comp['summary_rows']]; print(f"  max IBRE residual={comp['max_ne_residual']:.3e}")
    print("[7/7] TeX, provenance, and figures"); write_resulttodo_tex(outdir, sortres, five, fj, budget, baseline, comp); write_generated_tables(outdir, sortres, fj, budget, baseline, comp); write_provenance(outdir, five, fj, budget, comp); write_reproducibility_extras(outdir, five, budget, comp)
    if not args.no_figures: save_figures(outdir, five, fj, budget, baseline, comp)
    # Refresh the manifest after figure generation so it inventories the complete output tree.
    names = sorted(x.name for x in outdir.iterdir() if x.is_file() and x.name != "MANIFEST.txt")
    (outdir / "MANIFEST.txt").write_text("\n".join(names + ["MANIFEST.txt"]) + "\n", encoding="utf-8")
    print("Done. Numerical CSV/TeX/figure outputs are in:", outdir)


if __name__ == "__main__":
    main()
