# Molecular-Quantum-Algorithms
![alt text](https://github.com/riyakayal/Molecular-Quantum-Algorithms/blob/main/docs/QA.png?raw=true)
Quantum algorithm implementations for molecular electronic structure: VQE with UCCSD, SUCCD, PUCCD, UCCGSD and UCCSD_reps2 ansätze on a statevector simulator. Includes a general PES pipeline, spectroscopic constant extraction, and systematic ansatz benchmarking. Classical CCSD/FCI references throughout.


## Sub-repositories

### [`pes-vqe/`](./pes-vqe/)
**General potential energy surface pipeline**

A molecule-agnostic pipeline for computing PES curves with VQE. Supports any diatomic or small polyatomic molecule via a geometry template or custom string. Ansatz selection (UCCSD, SUCCD, PUCCD, UCCGSD), CCSD T1/T2 warm-starting, and spectroscopic constant extraction (Re, De, ωe, ωexe, Be) are all built in. Each geometry point is an independent SLURM job. LiH / STO-3G with PUCCD and SUCCD is included as a worked example.

### [`ansatz-comparison/`](./ansatz-comparison/)
**Systematic ansatz benchmarking at equilibrium geometry**

Comparison of UCCSD, UCCGSD, PUCCD, SUCCD, and UCCSD(reps=2) for molecular systems at fixed geometry. Benchmarks parameter count, circuit depth, correlation recovery, and wall time. Demonstrates the accuracy-cost tradeoff and the importance of CCSD amplitude initialisation. LiH·H₂ / STO-3G is the primary example.

### [`adapt-vqe/`](./adapt-vqe/)
**Adaptive circuit construction**

ADAPT-VQE with a singles/doubles operator pool. Circuits are grown iteratively by selecting operators with the largest energy gradient. Demonstrates parameter economy versus fixed-structure ansätze. 

## Technical stack

| Package | Version |
|---------|---------|
| Python | 3.9 |
| qiskit | 1.2.4 |
| qiskit-nature | 0.7.2 |
| qiskit-aer | 0.15.1 |
| qiskit-algorithms | 0.3.1 |
| symengine | 0.11.0 |
| pyscf | 2.12.1 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |

Full installation instructions in each sub-repository's `requirements.txt`.

---

## Design principles

**Reproducibility** — All results use `seed_simulator=42`, `OMP_NUM_THREADS=1`, and a fixed SLURM node. Each result is saved as a self-contained JSON with all metadata including optimizer settings, circuit depth, wall time, and convergence trace.

**Physical correctness** — All VQE energies are verified against FCI (exact diagonalisation of the qubit Hamiltonian). Results below the FCI floor are flagged as unphysical.

**CCSD warm-starting** — CCSD T1/T2 amplitudes are mapped to ansatz parameters before VQE begins. This places the optimizer within ~0.5 mHa of the CCSD reference at iteration zero, dramatically reducing the number of circuit evaluations needed.

**Molecule-agnostic design** — `molecule_setup.py` provides a unified interface. Adding a new molecule requires only a geometry string. PES scanning via `pes_runner.py` handles any diatomic or linear triatomic automatically.

**HPC-native** — Each geometry point or single-point calculation is an independent SLURM job. Failed or missing points are detected and resubmitted automatically.

---

## Author

Riya Kayal
PhD, Theoretical Chemistry — Max-Planck-Institut für Kohlenforschung.
GitHub: [riyakayal](https://github.com/riyakayal)
