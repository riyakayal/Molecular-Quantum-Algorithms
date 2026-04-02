

# Author: Riya Kayal
# Created: 20/09/2025


import os
import sys
import numpy as np
from pyscf import gto, dft, scf, cc
from pyscf.geomopt import berny_solver
import time
from datetime import datetime

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_algorithms import VQE
from qiskit_aer.primitives import Estimator


#---------------------------------------------------------------
# STEP 1: Choose ansatz
#---------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1" ## for reproducibility
os.environ["GOMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# change to: UCCGSD, PUCCD, SUCCD, UCCSD_reps2, EfficientSU2
if len(sys.argv) > 1:
    ANSATZ_LABEL = sys.argv[1]
else:
    ANSATZ_LABEL = "UCCSD" # Fallback default

MAXITER = 10
LR = 0.0001
PERT = 0.0001
LAST_AVG = 20
RES = 2
print("="*50)
print(f"      INITIALIZING VQE RUN: {ANSATZ_LABEL} ")
print("="*50)

# Record the start time
start_wall = datetime.now()
start_perf = time.perf_counter()

print(f"   JOB STARTED  : {start_wall.strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)    

#---------------------------------------------------------------
# STEP 2: Get Geometry
#---------------------------------------------------------------
from molecule_setup import setup_molecule, ccsd_initial_point, MOLECULES

# ── Configure here ────────────────────────────────────────────────────────────
MOL_NAME = "LiH"    # or "H2_HF", "H2O", "LiH", "BeH2", "HF", "custom"
BASIS = ["sto-3g", "cc-pvdz"][0]
mol_data = setup_molecule(MOL_NAME, run_fci=True)
print(f"Molecule: {MOL_NAME}, basis: {BASIS}")

# Unpack for downstream use
clean_atom_list = mol_data["clean_atom_list"]
mf_vqe          = mol_data["mf"]
e_ccsd          = mol_data["e_ccsd"]
e_fci_total     = mol_data["e_fci"]
t1              = mol_data["t1"]
t2              = mol_data["t2"]

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA

#---------------------------------------------------------------
# STEP 3: Prepare Qubit Operator
#---------------------------------------------------------------
qiskit_geom = "; ".join([f"{a} {p[0]} {p[1]} {p[2]}" for a, p in clean_atom_list])
print("Building driver...", flush=True)
driver = PySCFDriver(atom=qiskit_geom, basis='sto-3g', charge=0, spin=0)
driver = PySCFDriver(atom=qiskit_geom, basis=BASIS, charge=0, spin=0)
problem = driver.run()

print("Driver done.", flush=True)

mapper = ParityMapper(num_particles=problem.num_particles)
print("Mapper done.", flush=True)

tapered_mapper = problem.get_tapered_mapper(mapper)
print("Tapering done.", flush=True)

qubit_op = tapered_mapper.map(problem.second_q_ops()[0])

# No tapering (don't use)
#qubit_op = mapper.map(problem.second_q_ops()[0])

print(f"Qubit op done. Qubits: {qubit_op.num_qubits}", flush=True)


qubit_op = tapered_mapper.map(problem.second_q_ops()[0])

# ── CRITICAL DIAGNOSTIC ───────────────────────────────────────────────────────
print(f"\nqubit_op num_qubits:        {qubit_op.num_qubits}")
print(f"nuclear_repulsion_energy:   {problem.nuclear_repulsion_energy:.10f} Ha")

H_matrix = qubit_op.to_matrix()
print(f"H_matrix shape:             {H_matrix.shape}")
exact_eigs = np.sort(np.linalg.eigvalsh(H_matrix).real)
print(f"Exact eigenvalues of qubit H:")
for i, e in enumerate(exact_eigs):
    print(f"  eig[{i}] = {e:.10f}  →  total = {e + problem.nuclear_repulsion_energy:.10f} Ha")

print(f"\nExpected FCI total:         {e_fci_total:.10f} Ha")
print(f"Lowest eig + nuc_rep:       {exact_eigs[0] + problem.nuclear_repulsion_energy:.10f} Ha")

# ── Test 1: PySCF FCI direct ─────────────────────────────────────────────────
from pyscf import fci as pyscf_fci
cisolver = pyscf_fci.FCI(mf_vqe)
cisolver.verbose = 0
e_fci_pyscf, _ = cisolver.kernel()
print(f"PySCF FCI direct:        {e_fci_pyscf:.10f} Ha")
print(f"molecule_setup FCI:      {e_fci_total:.10f} Ha")
print(f"Difference:              {(e_fci_pyscf - e_fci_total)*1000:+.4f} mHa")

# ── Test 2: Qiskit NumPy exact diag ──────────────────────────────────────────
# Run after qubit_op is built
from qiskit_algorithms import NumPyMinimumEigensolver
numpy_solver  = NumPyMinimumEigensolver()
fci_result    = numpy_solver.compute_minimum_eigenvalue(qubit_op)
e_fci_qiskit  = fci_result.eigenvalue.real + problem.nuclear_repulsion_energy
print(f"Qiskit FCI (exact diag): {e_fci_qiskit:.10f} Ha")
print(f"PySCF FCI:               {e_fci_pyscf:.10f} Ha")
print(f"Difference:              {(e_fci_qiskit - e_fci_pyscf)*1000:+.4f} mHa")

# Use Qiskit FCI as the ground truth for VQE comparisons
e_fci_total = e_fci_qiskit
print(f"\nUsing Qiskit FCI as ground truth: {e_fci_total:.10f} Ha")


#---------------------------------------------------------------
# STEP 4: Define Ansatz
#---------------------------------------------------------------

if ANSATZ_LABEL == "UCCSD": 
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        tapered_mapper,
        initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, tapered_mapper)
    )
elif ANSATZ_LABEL == "UCCGSD":
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        tapered_mapper,
        initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, tapered_mapper),
        generalized=True,
    )
elif ANSATZ_LABEL == "PUCCD":
    from qiskit_nature.second_q.circuit.library import PUCCD
    ansatz = PUCCD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        tapered_mapper,
        initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, tapered_mapper)
    )
elif ANSATZ_LABEL == "SUCCD":
    from qiskit_nature.second_q.circuit.library import SUCCD
    ansatz = SUCCD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        tapered_mapper,
        initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, tapered_mapper)
    )
elif ANSATZ_LABEL == "UCCSD_reps2":
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        tapered_mapper,
        initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, tapered_mapper),
        reps=2
    )    
else:
    raise ValueError(f"Unknown ansatz: {ansatz_label}")

print(f"Ansatz done. Ansatz: {ANSATZ_LABEL}  Parameters: {ansatz.num_parameters}", flush=True)

#---------------------------------------------------------------
# STEP 5:  CCSD T1/T2 -> UCCSD initial point
# No frozen core. Run after cc_seed and ansatz are both defined.
#---------------------------------------------------------------

initial_point = ccsd_initial_point(
    t1, t2,
    mol_data["nocc"],
    mol_data["nvirt"],
    ansatz,
    problem.num_spatial_orbitals
)

#---------------------------------------------------------------
# STEP 6: Define Optimizer
#---------------------------------------------------------------

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA


from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_algorithms.optimizers import L_BFGS_B


MaxIt = 1000
Ftol = 1e-12
Gtol = 1e-6
eps = 1e-4
print(f"L_BFGS settings: maxiter={MaxIt}, ftol={Ftol}, gtol={Gtol}, eps={eps}")
optimizer = L_BFGS_B(maxiter=MaxIt, options={'ftol': Ftol, 'gtol': Gtol, 'eps': eps})


#---------------------------------------------------------------
# STEP 7: Define Estimator (reproducible)
#---------------------------------------------------------------

print("Building Estimator...", flush=True)

from qiskit.primitives.base import BaseEstimatorV1
from qiskit.primitives import EstimatorResult, PrimitiveJob
from qiskit.quantum_info import Statevector
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
import numpy as np

# ── Exact V1 Estimator — no transpilation, no shots, no randomness ────────────
class ExactEstimatorV1(BaseEstimatorV1):
    """
    Minimal exact statevector estimator.
    Bypasses all transpilation — computes <psi|H|psi> directly via numpy.
    Compatible with qiskit_algorithms 0.3.1 VQE (V1 API).
    """
    def _run(self, circuits, observables, parameter_values, **run_options):
        def compute():
            evs = []
            for circ, obs, params in zip(circuits, observables, parameter_values):
                # Bind parameters directly — no transpilation
                bound = circ.assign_parameters(
                    dict(zip(circ.parameters, params))
                )
                sv  = Statevector(bound)
                ev  = sv.expectation_value(obs).real
                evs.append(ev)
            return EstimatorResult(np.array(evs), [{}] * len(evs))

        job = PrimitiveJob(compute)
        job._submit()
        return job   
estimator = ExactEstimatorV1()
print("Estimator done.", flush=True)

## Note: ReverseEstimatorGradient is INCOMPATIBLE
## gradient needed for L_BFGS, not SPSA
gradient  = ParamShiftEstimatorGradient(estimator)

# Trim/pad
_n = ansatz.num_parameters
if len(initial_point) >= _n:
    initial_point = initial_point[:_n]
else:
    initial_point = np.concatenate([initial_point,
                                    np.zeros(_n - len(initial_point))])

# --- STEP 7.5 (optional): reset to zero to check if CCSD/Zero initialization for ansatz is better
if ANSATZ_LABEL == "XXXX": #replace to test for a specific ansatz
    initial_point = np.zeros(ansatz.num_parameters)
    print("Using zeros for initialization instead of CCSD T1, T2 amplitudes...\n")
    # Sanity check
    # If e_init equals HF energy, zeros is a valid physical starting point for SUCCD. Then use L-BFGS-B from there — it should find the true minimum without the spurious sector issue.
    job_init = estimator.run([ansatz], [qubit_op], [initial_point])
    e_init   = job_init.result().values[0] + problem.nuclear_repulsion_energy
    print(f"Energy at zeros: {e_init:.10f} Ha", flush=True)
    print(f"Should equal HF: {mf_vqe.e_tot:.10f} Ha", flush=True)

job_init = estimator.run([ansatz], [qubit_op], [initial_point])
e_init   = job_init.result().values[0] + problem.nuclear_repulsion_energy
print(f"Energy at CCSD initial point: {e_init:.10f} Ha", flush=True)
print(f"HF energy:                    {mf_vqe.e_tot:.10f} Ha", flush=True)
print(f"CCSD energy:                  {e_ccsd:.10f} Ha", flush=True)


#---------------------------------------------------------------
# STEP 8: VQE Execution
#---------------------------------------------------------------

print("\n" + "="*35)
print(f"VQE RESOURCE ESTIMATE ({ANSATZ_LABEL})")
print("="*35)
print(f"  Electrons:      {problem.num_particles}")
print(f"  Spatial Orbitals: {problem.num_spatial_orbitals}")
print(f"  Qubits (Tapered): {qubit_op.num_qubits}")
print(f"  Parameters:       {ansatz.num_parameters}")
print("="*35)


# 1. Callback
best_energy = np.inf
counts, values = [], []
def vqe_callback(eval_count, parameters, mean, metadata):
    global best_energy
    total_energy = mean + problem.nuclear_repulsion_energy
    counts.append(eval_count)
    values.append(total_energy)
    if total_energy < best_energy:
        best_energy = total_energy
    if eval_count % 1 == 0:
        print(f"  Eval {eval_count:3}: {total_energy:15.10f} Ha  "
              f"Best: {best_energy:15.10f} Ha", flush=True)

# 2. VQE
print("Building VQE...", flush=True)

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
    gradient=gradient,
    callback=vqe_callback,
    initial_point=initial_point,
)
print(type(optimizer).__name__)
print(optimizer)
print("VQE built. Running...", flush=True)
result = vqe.compute_minimum_eigenvalue(qubit_op)
print("VQE finished.", flush=True)


# ── TARGETED DIAGNOSTIC (optional) ───────────────────────────────────────────────────────
print(f"\nresult.eigenvalue (RAW, no nuc_rep added): {result.eigenvalue.real:.10f} Ha")
print(f"nuclear_repulsion_energy:                  {problem.nuclear_repulsion_energy:.10f} Ha")
print(f"result.eigenvalue + nuc_rep:               {result.eigenvalue.real + problem.nuclear_repulsion_energy:.10f} Ha")
print(f"Optimal parameters:                        {result.optimal_point}")

# Manually recompute energy at optimal point
job_opt = estimator.run([ansatz], [qubit_op], [result.optimal_point])
e_opt_manual = job_opt.result().values[0]
print(f"\nManual estimator at optimal point (RAW):   {e_opt_manual:.10f} Ha")
print(f"Manual + nuc_rep:                          {e_opt_manual + problem.nuclear_repulsion_energy:.10f} Ha")

# Eigenvalue bounds check
eigs = np.sort(np.linalg.eigvalsh(qubit_op.to_matrix()).real)
print(f"\nqubit_op eigenvalue range: [{eigs[0]:.10f}, {eigs[1]:.10f}] Ha")
print(f"Is result.eigenvalue inside range? {eigs[0] <= result.eigenvalue.real <= eigs[1]}")

# Check if qubit_op already contains nuclear repulsion
print(f"\nIf qubit_op includes nuc_rep, FCI would be: {eigs[0]:.10f} Ha  (no addition needed)")
print(f"If qubit_op excludes nuc_rep, FCI would be: {eigs[0] + problem.nuclear_repulsion_energy:.10f} Ha")
print(f"Known FCI:                                   {e_fci_pyscf:.10f} Ha")

# ----------  Diagnostic End ---------------------------------------------------

#---------------------------------------------------------------
# STEP 9: Report
#---------------------------------------------------------------

# Final energy using last_avg smoothed optimal point
vqe_final_total = result.eigenvalue.real + problem.nuclear_repulsion_energy

# Also evaluate at the best point seen during optimization
best_total = best_energy  # tracked in callback

print("\n" + "="*55)
print(f"VQE FINAL (last_avg smoothed): {vqe_final_total:15.10f} Ha")
print(f"VQE BEST  (callback minimum):  {best_total:15.10f} Ha")
print(f"HF REFERENCE:                  {mf_vqe.e_tot:15.10f} Ha")
print(f"CCSD REFERENCE:                {e_ccsd:15.10f} Ha")
print(f"FCI REFERENCE:                 {e_fci_total:15.10f} Ha")
print(f"Error (final vs CCSD):         {(vqe_final_total - e_ccsd)*1000:+12.5f} mHa")
print(f"Error (best  vs CCSD):         {(best_total - e_ccsd)*1000:+12.5f} mHa")
print(f"Error (final vs FCI):          {(vqe_final_total - e_fci_total)*1000:+12.5f} mHa")
print(f"Correlation recovered (final): {(vqe_final_total - mf_vqe.e_tot)/(e_ccsd - mf_vqe.e_tot)*100:6.2f} %")
print(f"Correlation recovered (best):  {(best_total - mf_vqe.e_tot)/(e_ccsd - mf_vqe.e_tot)*100:6.2f} %")
print("="*55)

#--------------------------------
# STEP 9: Save results
#--------------------------------

import json

# ── SET THIS before each run ─────────────────────────────────────────────────

os.makedirs("ansatz_results", exist_ok=True)
outfile = f"ansatz_results/{ANSATZ_LABEL}.json"


# Determine canonical energy — final (last_avg smoothed) is always the reported value
# best is logged but flagged if unphysical
best_is_physical = bool(best_energy >= e_fci_total - 1e-3)

if not best_is_physical:
    print(f"WARNING: best energy {best_energy:.10f} Ha is below FCI "
          f"{e_fci_total:.10f} Ha by "
          f"{(e_fci_total - best_energy)*1000:.3f} mHa — spurious SPSA eval",
          flush=True)
    print(f"  Using vqe_final_total as canonical result.", flush=True)

canonical_e   = vqe_final_total
canonical_err = (canonical_e - e_ccsd) * 1000
fci_err = (canonical_e - e_fci_total) * 1000

save_data = {
    "label":          ANSATZ_LABEL,
    "system":         MOL_NAME,
    "basis":          BASIS,
    "mapper":         "ParityMapper (tapered)",

    "n_electrons":    list(problem.num_particles),
    "n_spatial_orbs": problem.num_spatial_orbitals,
    "n_qubits":       int(qubit_op.num_qubits),
    "n_params":       int(ansatz.num_parameters),
    "circuit_depth":  int(ansatz.depth()),

    "e_hf_ha":        float(mf_vqe.e_tot),
    "e_ccsd_ha":      float(e_ccsd),
    "e_ccsd_init_ha": float(e_init),
    "e_fci_ha":       float(e_fci_total),
    "nuc_repulsion":  float(problem.nuclear_repulsion_energy),

    # Canonical result — always vqe_final_total
    "e_vqe_ha":       float(canonical_e),
    "error_mha":      float(canonical_err),
    "error_fci_mha":  float((canonical_e - e_fci_total) * 1000),
    "corr_recovered_pct": float(
        (canonical_e - mf_vqe.e_tot) / (e_ccsd - mf_vqe.e_tot) * 100
    ),

    # Informational only — not used in plots
    "e_vqe_best_ha":      float(best_energy),
    "error_best_mha":     float((best_energy - e_ccsd) * 1000),
    "best_is_physical":   best_is_physical,

    "eval_counts":    [int(c) for c in counts],
    "eval_energies":  [float(v) for v in values],
    "opt_params":     result.optimal_point.tolist(),

    "ccsd_init_nonzero": int(np.count_nonzero(initial_point)),
    "ccsd_init_max_amp": float(np.max(np.abs(initial_point))),

    "optimizer":      "L_BFGS",
}

with open(outfile, "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\n[Saved] {outfile}")
print(f"  Canonical energy:  {canonical_e:.10f} Ha")
print(f"  Error vs CCSD:     {canonical_err:+.3f} mHa")
print(f"  Error vs FCI:      {canonical_err:+.3f} mHa")
print(f"  Best is physical:  {best_is_physical}")
print(f"  Corr recovered:    {save_data['corr_recovered_pct']:.2f} %")

# Track time
end_wall = datetime.now()
end_perf = time.perf_counter()

# 3. Calculate duration
total_seconds = end_perf - start_perf
hours, rem = divmod(total_seconds, 3600)
minutes, seconds = divmod(rem, 60)

print("\n" + "-" * 50)
print(f"     JOB FINISHED : {end_wall.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"     TOTAL TIME   : {int(hours)}h {int(minutes)}m {seconds:.2f}s")
print("-" * 50)
