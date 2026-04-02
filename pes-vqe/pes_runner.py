
# Author: Riya Kayal
# Created: 30/11/2025

"""
pes_runner.py
=============
Run VQE at a single geometry point for PES scanning.
Called by the SLURM array job script for each R value.

Usage in claude01.py — replace Step 1/2 with:

    from pes_runner import run_pes_point, PES_ANSATZE

At the top of claude01.py, set:
    R = float(os.environ["PES_R"])      # passed from SLURM array
    MOL_NAME = os.environ["MOL_NAME"]   # e.g. "H2"
    ANSATZ_LABEL = os.environ["ANSATZ"] # e.g. "UCCSD"
"""

import os
import json
import time
import numpy as np

from molecule_setup import setup_molecule, ccsd_initial_point, MOLECULES
from estimator_class import get_estimator
# =============================================================================
# WHICH ANSATZE TO RUN AT EACH POINT
# Set True/False to control which are computed
# =============================================================================

PES_ANSATZE = {
    "UCCSD":       True,
    "SUCCD":       True,
    "PUCCD":       True,
    "UCCGSD":      False,   # slow — enable if needed
    "UCCSD_reps2": False,
}


def build_geometry(mol_name, R):
    """
    Build geometry string for a molecule with bond length R (Angstrom).
    Supports H2, LiH, HF, and linear triatomics.
    """
    templates = {
        "H2":  f"H 0.000 0.000 0.000; H 0.000 0.000 {R:.6f}",
        "LiH": f"Li 0.000 0.000 0.000; H 0.000 0.000 {R:.6f}",
        "HF":  f"H 0.000 0.000 0.000; F 0.000 0.000 {R:.6f}",
        "BeH2": (
            f"Be 0.000 0.000 0.000; "
            f"H 0.000 0.000 {R:.6f}; "
            f"H 0.000 0.000 {-R:.6f}"
        ),
    }
    if mol_name not in templates:
        raise ValueError(
            f"No geometry template for '{mol_name}'. "
            f"Add it to pes_runner.build_geometry() or use molecule_setup.py."
        )
    return templates[mol_name]


def run_pes_point(
    mol_name,
    R,
    ansatz_label,
    basis="sto-3g",
    output_dir="pes_results",
    seed_simulator=42,
    spsa_maxiter=300,
    spsa_lr=0.005,
    spsa_pert=0.005,
    use_lbfgs=True,
    lbfgs_maxiter=200,
):
    """
    Run VQE + CCSD + FCI at a single geometry point R.

    Parameters
    ----------
    mol_name : str
        Molecule name matching build_geometry templates.
    R : float
        Bond length in Angstrom.
    ansatz_label : str
        One of: UCCSD, SUCCD, PUCCD, UCCGSD, UCCSD_reps2.
    basis : str
        Basis set.
    output_dir : str
        Directory to save JSON results.
    seed_simulator : int
        AerSimulator seed for reproducibility.
    spsa_maxiter : int
        SPSA maximum iterations.
    spsa_lr : float
        SPSA learning rate.
    spsa_pert : float
        SPSA perturbation.
    use_lbfgs : bool
        Use L-BFGS-B instead of SPSA.
    lbfgs_maxiter : int
        L-BFGS-B maximum iterations.

    Returns
    -------
    dict with all results for this (R, ansatz) point.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["GOMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import Estimator
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SPSA, L_BFGS_B
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import ParityMapper
    from qiskit_nature.second_q.circuit.library import (
        UCCSD, PUCCD, SUCCD, HartreeFock
    )

    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{mol_name}_{ansatz_label}_R{R:.4f}.json")

    # Skip if already done
    if os.path.exists(outfile):
        print(f"  [Skip] {outfile} already exists.", flush=True)
        with open(outfile) as f:
            return json.load(f)

    print(f"\n{'='*60}", flush=True)
    print(f"  {mol_name} / {ansatz_label} / R={R:.4f} Å / {basis}", flush=True)
    print(f"{'='*60}", flush=True)

    t_total = time.time()

    # ── Classical reference ──────────────────────────────────────────────────
    geom = build_geometry(mol_name, R)
    mol_data = setup_molecule(
        "custom",
        geometry=geom,
        basis=basis,
        run_fci=True,
        verbose=True,
    )
    e_hf   = mol_data["e_hf"]
    e_ccsd = mol_data["e_ccsd"]
    e_fci  = mol_data["e_fci"]
    t1     = mol_data["t1"]
    t2     = mol_data["t2"]
    nocc   = mol_data["nocc"]
    nvirt  = mol_data["nvirt"]
    mf     = mol_data["mf"]

    print(f"  E(HF)   = {e_hf:.8f} Ha")
    print(f"  E(CCSD) = {e_ccsd:.8f} Ha")
    print(f"  E(FCI)  = {e_fci:.8f} Ha" if e_fci else "  E(FCI) = N/A")

    # ── Qiskit problem setup ─────────────────────────────────────────────────
    clean_atom_list = mol_data["clean_atom_list"]
    qiskit_geom = "; ".join(
        [f"{a} {p[0]} {p[1]} {p[2]}" for a, p in clean_atom_list]
    )

    driver  = PySCFDriver(atom=qiskit_geom, basis=basis, charge=0, spin=0)
    problem = driver.run()
    NUC     = problem.nuclear_repulsion_energy
    mapper  = ParityMapper(num_particles=problem.num_particles)
    tmapper = problem.get_tapered_mapper(mapper)
    qubit_op = tmapper.map(problem.second_q_ops()[0])
    hf_state = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        tmapper
    )

    # ── Build ansatz ─────────────────────────────────────────────────────────
    if ansatz_label == "UCCSD":
        ansatz = UCCSD(
            problem.num_spatial_orbitals, problem.num_particles, tmapper,
            initial_state=hf_state
        )
    elif ansatz_label == "UCCGSD":
        ansatz = UCCSD(
            problem.num_spatial_orbitals, problem.num_particles, tmapper,
            initial_state=hf_state, generalized=True
        )
    elif ansatz_label == "UCCSD_reps2":
        ansatz = UCCSD(
            problem.num_spatial_orbitals, problem.num_particles, tmapper,
            initial_state=hf_state, reps=2
        )
    elif ansatz_label == "PUCCD":
        ansatz = PUCCD(
            problem.num_spatial_orbitals, problem.num_particles, tmapper,
            initial_state=hf_state, include_singles=(True,True)
        )
    elif ansatz_label == "SUCCD":
        ansatz = SUCCD(
            problem.num_spatial_orbitals, problem.num_particles, tmapper,
            initial_state=hf_state, include_singles=(True,True)
        )
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_label}")

    # ── Transpile ────────────────────────────────────────────────────────────
    backend  = AerSimulator(method='statevector', seed_simulator=seed_simulator)
    ansatz_t = transpile(ansatz, backend=backend, optimization_level=1)
    n_params = ansatz_t.num_parameters
    print(f"  Qubits={qubit_op.num_qubits}  Params={n_params}  "
          f"Depth={ansatz_t.depth()}", flush=True)

    # ── CCSD initial point ────────────────────────────────────────────────────
    initial_point = ccsd_initial_point(
        t1, t2, nocc, nvirt, ansatz, problem.num_spatial_orbitals
    )
    _n = ansatz_t.num_parameters
    if len(initial_point) >= _n:
        initial_point = initial_point[:_n]
    else:
        initial_point = np.concatenate(
            [initial_point, np.zeros(_n - len(initial_point))]
        )

    # ── Estimator ────────────────────────────────────────────────────────────
    estimator = get_estimator()

    # ── Optimizer ────────────────────────────────────────────────────────────
    #if use_lbfgs and n_params <= 60:
    if use_lbfgs:
        from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
        optimizer = L_BFGS_B(maxiter=lbfgs_maxiter,
                             options={'ftol': 1e-10, 'gtol': 1e-6})
        opt_name  = f"L-BFGS-B(maxiter={lbfgs_maxiter})"
        gradient  = ParamShiftEstimatorGradient(estimator)
    else:
        optimizer = SPSA(
            maxiter=spsa_maxiter,
            learning_rate=spsa_lr,
            perturbation=spsa_pert,
            last_avg=20,
            resamplings=1,
        )
        opt_name = f"SPSA(maxiter={spsa_maxiter},lr={spsa_lr},pert={spsa_pert})"
        gradient = None

    print(f"  Optimizer: {opt_name}", flush=True)

    # ── VQE ──────────────────────────────────────────────────────────────────
    counts, values = [], []
    best_energy    = np.inf

    def callback(ec, params, mean, meta):
        nonlocal best_energy
        e = mean + NUC
        counts.append(ec)
        values.append(e)
        if e < best_energy:
            best_energy = e
        if ec % 20 == 0:
            print(f"    Eval {ec:4}: {e:.8f} Ha  Best: {best_energy:.8f} Ha",
                  flush=True)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        gradient=gradient,
        callback=callback,
        initial_point=initial_point,
    )

    t_vqe    = time.time()
    result   = vqe.compute_minimum_eigenvalue(qubit_op)
    wall_vqe = time.time() - t_vqe

    e_vqe      = result.eigenvalue.real + NUC
    error_ccsd = (e_vqe - e_ccsd) * 1000
    error_fci  = (e_vqe - e_fci)  * 1000 if e_fci else None
    corr_pct   = (e_vqe - e_hf) / (e_ccsd - e_hf) * 100

    print(f"\n  VQE  = {e_vqe:.8f} Ha")
    print(f"  CCSD = {e_ccsd:.8f} Ha")
    print(f"  FCI  = {e_fci:.8f} Ha" if e_fci else "  FCI = N/A")
    print(f"  Error vs CCSD: {error_ccsd:+.3f} mHa")
    print(f"  Corr recovered: {corr_pct:.2f}%")
    print(f"  Wall time: {wall_vqe/60:.1f} min", flush=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    data = {
        "mol_name":     mol_name,
        "ansatz":       ansatz_label,
        "basis":        basis,
        "R_angstrom":   R,
        "optimizer":    opt_name,
        "seed":         seed_simulator,
        "n_qubits":     int(qubit_op.num_qubits),
        "n_params":     int(n_params),
        "circuit_depth": int(ansatz_t.depth()),
        "e_hf":         float(e_hf),
        "e_ccsd":       float(e_ccsd),
        "e_fci":        float(e_fci) if e_fci else None,
        "e_vqe":        float(e_vqe),
        "e_vqe_best":   float(best_energy),
        "error_ccsd_mha": float(error_ccsd),
        "error_fci_mha":  float(error_fci) if error_fci else None,
        "corr_pct":     float(corr_pct),
        "wall_min":     round(wall_vqe / 60, 2),
        "eval_counts":  [int(c) for c in counts],
        "eval_energies": [float(v) for v in values],
        "opt_params":   result.optimal_point.tolist(),
    }

    with open(outfile, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {outfile}", flush=True)

    return data
