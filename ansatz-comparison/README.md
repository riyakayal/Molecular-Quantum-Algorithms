# VQE Ansatz Comparison
![alt text](https://github.com/riyakayal/Molecular-Quantum-Algorithms/blob/main/docs/Ansatz_comp.png?raw=true)


**Part of:** [Molecular-Quantum-Algorithms](https://github.com/riyakayal/Molecular-Quantum-Algorithms)

Systematic benchmarking of five unitary coupled-cluster ansätze for molecular ground-state energy via VQE on a statevector simulator. Any molecule from `molecule_setup.py` can be used; LiH / STO-3G is included as a worked example with complete output.

Each ansatz is run as an independent job, results are saved as JSON, and a comparison script generates publication-quality plots and a summary table against CCSD and FCI references.

---

## Ansätze compared

| Label | Ansatz | Notes |
|-------|--------|-------|
| `UCCSD` | Unitary CCSD | Standard chemically-motivated baseline |
| `UCCGSD` | Generalised UCCSD | Occupied→occupied and virtual→virtual rotations included |
| `PUCCD` | Paired UCCSD | Geminal doubles only — fewest parameters |
| `SUCCD` | Spin-adapted UCCSD | Singlet-adapted doubles — fewer params than UCCSD |
| `UCCSD_reps2` | UCCSD with 2 layers | Two-fold repetition of the excitation operator |

All are constructed from `qiskit_nature.second_q.circuit.library` with a `HartreeFock` initial state and `ParityMapper` with Z2 tapering.

---

## Key design features

**Custom exact estimator** — `vqe_runner.py` uses a hand-written `ExactEstimatorV1` that computes ⟨ψ|H|ψ⟩ directly via `Statevector` without any transpilation, shots, or randomness. This gives exact, reproducible expectation values with no ABI or seeding issues.

**CCSD T1/T2 warm-starting** — `ccsd_initial_point()` from `molecule_setup.py` maps PySCF CCSD T1/T2 amplitudes to the ansatz parameter ordering via `excitation_list`, using blocked spin-orbital indexing. This places the L-BFGS-B optimizer within ~5 mHa of the CCSD reference before the first gradient step.

**Physical correctness checks** — FCI is verified by cross-checking PySCF `fci.FCI(mf)`, Qiskit `NumPyMinimumEigensolver`, and exact diagonalisation of the qubit Hamiltonian matrix. VQE results below the FCI floor are flagged as unphysical.

**L-BFGS-B with parameter-shift gradients** — gradient-based optimisation via `ParamShiftEstimatorGradient` for all ansätze. An optional per-ansatz zero initialisation override is included for cases where CCSD warm-starting leads to a spurious minimum.

---

## Example output: LiH / STO-3G

System: 4 electrons, 6 spatial orbitals, 8 qubits (tapered). Three ansätze completed; UCCGSD and UCCSD\_reps2 not yet run.

### Results table

| Ansatz | Params | Depth | VQE energy (Ha) | Error vs CCSD (mHa) | Corr. recovered (%) |
|--------|--------|-------|-----------------|--------------------|--------------------|
| UCCSD  | 34 | 1 | −7.8823763388 | +0.000 | 100.00 |
| SUCCD  | 16 | 1 | −7.8799932430 | +2.383 | 88.31 |
| PUCCD  | 8  | 1 | −7.8779860740 | +4.390 | 78.46 |

CCSD reference: −7.8823764852 Ha | FCI reference: −7.8823869936 Ha | HF reference: −7.8619926887 Ha

**UCCSD reaches CCSD to within 0.15 μHa** — effectively exact convergence using L-BFGS-B with CCSD amplitude initialisation.

### CCSD initial point quality

All three ansätze achieved 100% amplitude coverage from the CCSD T1/T2 mapping (34/34, 16/16, 8/8 non-zero parameters). Initial energy before VQE: −7.8775 Ha (~4.9 mHa above CCSD), confirming the warm-start is working correctly.

---

## Repository structure

```
ansatz-comparison/
├── README.md
├── requirements.txt
├── molecule_setup.py             # RHF + CCSD + FCI; CCSD→ansatz amplitude mapping
├── vqe_runner.py                 # VQE pipeline: geometry → qubit op → ansatz → VQE → save
├── compare_uccsd_ansatzes.py     # Reads ansatz_results/*.json; generates all plots
├── submit.sh                     # SLURM submission for a single ansatz
├── workflow                      # Plain-text usage guide
├── ansatz_results/               # JSON output, one file per ansatz run
│   ├── UCCSD.json
│   ├── SUCCD.json
│   └── PUCCD.json
└── ansatz_comparison/            # Generated figures and summary
    ├── 1_convergence.png
    ├── 2_resources.png
    ├── 3_accuracy_vs_cost.png
    ├── 4_energy_levels.png
    ├── 5_ccsd_init_quality.png
    ├── 6_radar.png
    ├── results.json
    └── results_summary.txt
```

---

## Usage

### 1. Configure molecule and basis

In `vqe_runner.py`, set:

```python
MOL_NAME = "LiH"    # any key from molecule_setup.MOLECULES, or "custom"
BASIS    = "sto-3g"
```

Any molecule from `molecule_setup.MOLECULES` works: `H2`, `LiH`, `HF`, `BeH2`, `H2O`, `LiH_H2`, `H2_HF`. For a custom molecule:

```python
mol_data = setup_molecule("custom",
                          geometry="N 0 0 0; N 0 0 1.098",
                          basis="sto-3g")
```

### 2. Run each ansatz

**On HPC (SLURM):**
```bash
bash submit.sh UCCSD
bash submit.sh SUCCD
bash submit.sh PUCCD
bash submit.sh UCCGSD
bash submit.sh UCCSD_reps2
```

**Locally:**
```bash
python vqe_runner.py UCCSD
python vqe_runner.py SUCCD
# etc.
```

Jobs are fully independent — run in any order, in parallel, or incrementally. Results are saved to `ansatz_results/<ANSATZ>.json`.

### 3. Compare and plot

Run at any point (works with one or more completed ansätze):

```bash
python compare_uccsd_ansatzes.py
```

Output goes to `ansatz_comparison/`.

---

## Output plots

| File | Description |
|------|-------------|
| `1_convergence.png` | Running minimum energy vs evaluations for all ansätze |
| `2_resources.png` | Bar chart: parameters, circuit depth, error, wall time |
| `3_accuracy_vs_cost.png` | Accuracy vs parameter count and vs wall time (log scale) |
| `4_energy_levels.png` | Horizontal energy level diagram vs HF / CCSD / FCI |
| `5_ccsd_init_quality.png` | CCSD initial point energy and amplitude coverage per ansatz |
| `6_radar.png` | Multi-dimensional radar chart (accuracy, efficiency, compactness, speed) |

---

## Computational details

| Parameter | Value |
|-----------|-------|
| Qubit mapping | ParityMapper + Z2 tapering |
| Estimator | Custom `ExactEstimatorV1` (exact statevector, no transpilation) |
| Optimizer | L-BFGS-B (maxiter=1000, ftol=1e-12, gtol=1e-6) |
| Gradients | Parameter-shift via `ParamShiftEstimatorGradient` |
| Initial point | CCSD T1/T2 amplitudes (blocked spin-orbital mapping) |
| Reproducibility | `OMP_NUM_THREADS=1`, exact statevector (no sampling) |
| HPC | Forschungszentrum Jülich PGI cluster, ep29th partition |

---

## Part of the larger project

- `pes-vqe/` — general PES scanning pipeline (any molecule, any bond coordinate)
- `ansatz-comparison/` — this repository
  
