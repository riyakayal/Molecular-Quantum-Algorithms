# VQE Potential Energy Surface Pipeline
![alt text](https://github.com/riyakayal/Molecular-Quantum-Algorithms/blob/main/docs/pes.png?raw=True)
**Part of:** [Molecular-Quantum-Algorithms](https://github.com/riyakayal/Molecular-Quantum-Algorithms)

A general-purpose pipeline for computing molecular potential energy surfaces (PES) using Variational Quantum Eigensolver (VQE) on a statevector simulator. Any diatomic or small polyatomic molecule can be scanned by setting a molecule name, R range, and ansatz selection. Classical CCSD and FCI references are computed automatically for each geometry point.

The included example output covers LiH / STO-3G with PUCCD and SUCCD ansätze, demonstrating the pipeline end-to-end and providing a concrete benchmark against classical methods.

---

## What the pipeline does

For each (molecule, bond length R, ansatz) combination:

1. Runs PySCF RHF + CCSD + FCI at the given geometry
2. Maps CCSD T1/T2 amplitudes to the UCCSD parameter space as a warm-start initial point
3. Builds the qubit Hamiltonian via ParityMapper with Z2 tapering
4. Runs VQE with L-BFGS-B (small ansatze ≤60 params) or SPSA (large ansatze)
5. Saves a self-contained JSON with all energies, metadata, and convergence trace
6. Post-processing scripts generate PES plots and spectroscopic constants

---

## Supported molecules

Preset geometries are defined in `molecule_setup.py`:

| Key | Description |
|-----|-------------|
| `H2` | H₂ — minimal benchmark |
| `LiH` | LiH — standard VQE benchmark |
| `HF` | HF — strong correlation reference |
| `BeH2` | BeH₂ — linear triatomic |
| `H2O` | H₂O — strongly correlated benchmark |
| `LiH_H2` | LiH·H₂ — intermolecular complex |
| `H2_HF` | H₂·HF — hydrogen bond complex |
| `custom` | Any molecule via geometry string |

Adding a new molecule requires only a geometry string in `molecule_setup.MOLECULES` or passing it directly:

```python
mol_data = setup_molecule("custom",
                          geometry="N 0 0 0; N 0 0 1.098",
                          basis="sto-3g")
```

Bond-stretching templates for diatomics and symmetric triatomics are handled automatically by `pes_runner.build_geometry()`. Non-standard geometries can be passed as custom strings.

---

## Supported ansätze

| Ansatz | Class | Parameters (LiH/STO-3G) | Notes |
|--------|-------|------------------------|-------|
| UCCSD | `UCCSD` | 26 | Standard baseline |
| UCCGSD | `UCCSD(generalized=True)` | > 26 | Generalised excitations |
| PUCCD | `PUCCD` | 15 | Paired doubles only |
| SUCCD | `SUCCD` | 24 | Spin-adapted doubles |
| UCCSD_reps2 | `UCCSD(reps=2)` | ~52 | Two-layer ansatz |

Enable/disable per run in `pes_runner.PES_ANSATZE` or via the `ANSATZ` environment variable.

---

## Example output: LiH / STO-3G

Included in `pes_results/` and `pes_plots/`. 21 bond lengths from 1.2 to 3.0 Å, PUCCD and SUCCD ansätze.

### Spectroscopic constants

| Method | Re (Å) | De (eV) | ωe (cm⁻¹) | ωexe (cm⁻¹) | Be (cm⁻¹) |
|--------|--------|---------|-----------|------------|----------|
| FCI    | 1.5475 | 2.2836  | 1681.6    | 38.38      | 7.9979   |
| CCSD   | 1.5475 | 2.2855  | 1681.7    | 38.36      | 7.9982   |
| SUCCD  | 1.5411 | 2.4360  | 1712.8    | 37.33      | 8.0650   |
| PUCCD  | 1.5342 | 3.3722  | 1747.3    | 28.06      | 8.1375   |

SUCCD recovers Re and ωe to within 0.4% and 2% of FCI respectively. PUCCD overestimates De by ~48% due to incorrect dissociation behaviour — a known limitation of the paired-doubles truncation for heteronuclear diatomics.

### Error vs CCSD along the PES

| R (Å) | SUCCD (mHa) | PUCCD (mHa) |
|-------|-------------|-------------|
| 1.596 (Re) | +2.1 | +4.3 |
| 2.000 | +3.7 | +7.9 |
| 2.500 | +7.5 | +19.5 |
| 3.000 | +7.5 | +44.0 |

Both ansätze diverge at stretched geometries, motivating adaptive methods (ADAPT-VQE) for accurate dissociation curves.

---

## Computational details (LiH example)

| Parameter | Value |
|-----------|-------|
| Basis set | STO-3G |
| Active space | Full (4 electrons, 6 spatial orbitals) |
| Qubit mapping | ParityMapper + Z2 tapering |
| Qubits | 8 (SUCCD), 6 (PUCCD) |
| Optimizer | L-BFGS-B (maxiter=200) |
| Initial point | CCSD T1/T2 amplitudes |
| Simulator | Qiskit Aer statevector, seed=42 |
| Reproducibility | OMP_NUM_THREADS=1, seed_simulator=42 |
| HPC | Forschungszentrum Jülich PGI cluster, ep29th partition |

---

## Repository structure

```
pes-vqe/
├── README.md
├── requirements.txt
├── molecule_setup.py          # RHF + CCSD + FCI; CCSD→ansatz amplitude mapping
├── pes_runner.py              # Core VQE logic; geometry templates; ansatz builder
├── pes_point.py               # SLURM entry point (reads PES_R, ANSATZ, MOL_NAME)
├── estimator_class.py         # Reproducible AerSimulator (seed=42, OMP=1)
├── submit_pes.sh              # Submits one SLURM job per (R, ansatz); skips done
├── check_pes.sh               # Reports done / running / missing status
├── plot_pes.py                # PES plots: absolute, relative, error vs CCSD, correlation
├── spectr_consts.py           # Spectroscopic constants, Morse fit, derivative plots
├── pes_results/               # JSON output, one file per (molecule, ansatz, R)
└── pes_plots/                 # Generated figures and summary tables
```

---

## Running the pipeline

### 1. Environment

```bash
python3.9 -m venv venv_vqe
source venv_vqe/bin/activate
pip install -r requirements.txt
```

### 2. Single point (test)

```bash
export PES_R=1.5960
export MOL_NAME=LiH
export ANSATZ=SUCCD
export BASIS=sto-3g
python pes_point.py
```

### 3. Full PES scan on HPC

Edit `submit_pes.sh` — set `MOL_NAME`, `R_VALUES`, `ANSATZE`, partition, and working directory:

```bash
bash submit_pes.sh       # submits all jobs; skips already-completed points
bash check_pes.sh        # monitor done/running/missing
```

### 4. Generate plots and spectroscopic constants

```bash
python plot_pes.py           # PES curves (absolute, relative, error, correlation)
python spectr_consts.py      # spectroscopic constants + Morse fit + derivatives
```

---

## Reproducibility

All results are deterministic given:
- `seed_simulator=42` in `estimator_class.py`
- `OMP_NUM_THREADS=1` (set in both script and SLURM job)
- Fixed SLURM node (`--nodelist` in `submit_pes.sh`)
- CCSD T1/T2 initial point (deterministic given PySCF convergence tolerances)

Each JSON result file contains all parameters needed to reproduce or extend the calculation.

---

## Part of the larger project

This pipeline is one component of **quantum-molecule-algorithms**:

- `pes-vqe/` — this repository (general PES pipeline; LiH example)
- `ansatz-comparison/` — UCCSD / UCCGSD / PUCCD / SUCCD at equilibrium (LiH)
- `adapt-vqe/` — adaptive circuit construction
- `oo-vqe/` — orbital-optimised VQE for hydrogen-bonded complexes
