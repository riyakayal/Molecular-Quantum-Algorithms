# ADAPT-VQE for Molecules

**Part of:** [Molecular-Quantum-Algorithms](https://github.com/riyakayal/Molecular-Quantum-Algorithms)

A general implementation of the ADAPT-VQE algorithm for molecular ground-state energy, designed to handle systems up to ~20 qubits on a classical statevector simulator. Any molecule can be targeted by changing a single configuration variable. BeH₂ / STO-3G is included as a worked example demonstrating sub-μHa convergence to FCI in 8.2 minutes.

**Reference:** Grimsley et al., *Nat. Commun.* **10**, 3007 (2019).

---

## What ADAPT-VQE does

Unlike fixed-structure ansätze (UCCSD, SUCCD, etc.), ADAPT-VQE builds the ansatz circuit adaptively, one operator at a time:

1. Compute the energy gradient `|⟨ψ|[H, Aₖ]|ψ⟩|` for every operator Aₖ in the pool
2. Select the operator with the largest gradient
3. Append it to the ansatz with a new parameter initialised at zero
4. Re-optimise all parameters with L-BFGS-B
5. Repeat until the maximum gradient falls below a threshold

The result is a compact, problem-specific ansatz that selects only the operators that matter for the target molecule — in contrast to UCCSD which includes all singles and doubles regardless of their contribution.

---

## Key implementation features

**No Qiskit circuits at runtime** — the ADAPT loop runs entirely in NumPy/SciPy. Operators are stored as sparse matrices; state evolution uses `scipy.sparse.linalg.expm_multiply`. This avoids Qiskit transpilation overhead and makes each ADAPT iteration fast regardless of qubit count.

**Analytic forward-backward gradient** — energy and gradient are computed in a single forward-backward pass:
- Forward: `|f_k⟩ = U_{k-1}...U_0|HF⟩`
- Backward: `|b_k⟩ = U_k†...U_{n-1}† H|ψ⟩`
- Gradient: `dE/dθ_k = −2 Re⟨f_{k+1}|A_k|b_{k+1}⟩`

This is exact (no finite differences, no parameter-shift) and scales as O(n) in the number of selected operators.

**Commutator gradient for operator selection** — `|⟨ψ|[H,Aₖ]|ψ⟩|` is computed via sparse matrix-vector products without forming circuits or using parameter-shift. The pool matrices are pre-converted to sparse format once and reused at every ADAPT iteration.

**Sparse Hamiltonian throughout** — `H_sparse = qubit_op.to_matrix(sparse=True)` is built once. The dense Hamiltonian matrix is never formed, making the code memory-safe for systems up to ~20 qubits.

**Frozen core via `FreezeCoreTransformer`** — chemically inert core orbitals are frozen before qubit mapping, reducing qubit count significantly for second- and third-row atoms. The effective nuclear repulsion offset is derived from the PySCF HF energy to correctly absorb the frozen-core contribution.

**JordanWigner mapping, no tapering** — tapering is deliberately avoided to keep the qubit operator consistent with the sparse statevector arithmetic throughout the ADAPT loop.

**Complete fermionic pool** — αα, ββ, αβ singles and doubles, using JW spin-orbital indexing (alpha: 0..n−1, beta: n..2n−1). All operators are anti-Hermitian by construction.

---

## Supported molecules

Any molecule can be added via a geometry string in the configuration block. Built-in templates:

| Molecule | Geometry source |
|----------|----------------|
| `LiH` | Pre-set (B3LYP/cc-pVTZ optimised) |
| `LiF` | Pre-set |
| `BeH2` | B3LYP/cc-pVTZ optimisation at runtime |
| Custom | Any `atom_str` in PySCF format |

Change `MOLECULE = "BeH2"` at the top of `adapt_vqe.py` to switch systems. Geometry optimisation at B3LYP/cc-pVTZ runs automatically on the first step; the optimised geometry is saved to `optimized_<molecule>.xyz`.

---

## Example output: BeH₂ / STO-3G

System: Be frozen core (1 orbital), 4 active electrons, 6 active spatial orbitals, 12 qubits (JordanWigner). Operator pool: 92 operators.

### Summary

| Quantity | Value |
|----------|-------|
| HF energy | −15.5603349360 Ha |
| CCSD energy | −15.5944638855 Ha |
| FCI energy | −15.5948434748 Ha |
| ADAPT-VQE energy | −15.5948434685 Ha |
| Error vs FCI | **+0.000006 mHa** |
| Chemical accuracy (< 1 mHa) | ✓ |
| Operators selected | 35 / 92 |
| Wall time | 8.2 min |

ADAPT-VQE selects 35 of the 92 available operators and converges to within 6.4 nHa of FCI — well below chemical accuracy (1 mHa ≈ 0.627 kcal/mol). Only 38% of the pool was needed.

### Convergence trace

| Operators | Energy (Ha) | Error vs FCI (mHa) |
|-----------|-------------|-------------------|
| 1 | −15.5663392 | 28.504 |
| 5 | −15.5821747 | 12.669 |
| 10 | −15.5941853 | 0.658 ✓ |
| 15 | −15.5944530 | 0.391 ✓ |
| 20 | −15.5946476 | 0.196 ✓ |
| 25 | −15.5948288 | 0.015 ✓ |
| 30 | −15.5948429 | 0.001 ✓ |
| 35 | −15.5948435 | 0.000 ✓ |

(✓ = within chemical accuracy 1 mHa)

Chemical accuracy is reached at operator 10. The remaining 25 operators refine the energy further towards FCI.

### Dominant parameters at convergence

The 10 largest amplitudes at the optimum (|θ| > 0.2 rad):

```
θ_14 = +0.370  θ_12 = +0.370  θ_22 = −0.281  θ_23 = −0.281
θ_8  = −0.267  θ_9  = −0.267  θ_21 = +0.230  θ_18 = +0.223
```

The near-degeneracy of pairs (14,12), (22,23), (8,9) reflects BeH₂'s D∞h symmetry: equivalent α and β spin channels contribute symmetrically.

---

## Repository structure

```
adapt-vqe/
├── README.md
├── adapt_vqe.py              # Main script: geometry → pool → ADAPT loop → save
├── vqe_analysis/             # Output directory (example: BeH2)
│   ├── results.json
│   ├── results_summary.txt
│   ├── 3a_convergence.png
│   ├── 3b_parameter_distribution.png
│   ├── 3c_1d_landscape_top8.png
│   ├── 3d_2d_landscape.png
│   ├── 3e_barren_plateau.png
│   └── 4a_resource_scaling.png
└── optimized_beh2.xyz        # Optimised geometry from B3LYP/cc-pVTZ
```

---

## Usage

### 1. Configure

At the top of `adapt_vqe.py`:

```python
MOLECULE    = "BeH2"   # "LiH", "LiF", "BeH2", or add your own geometry
BASIS       = "sto-3g"
FREEZE_CORE = True     # recommended for second-row and heavier atoms
GRAD_THR    = 1e-6     # ADAPT convergence threshold (max gradient norm)
MAX_ITER    = 250      # maximum ADAPT iterations
```

### 2. Run

```bash
python adapt_vqe.py
```

Or on HPC:

```bash
sbatch submit.sh
```

### 3. Output

Results are saved to `vqe_analysis/results.json` and `vqe_analysis/results_summary.txt`. Analysis plots are generated automatically at the end of the run.

---

## Computational details

| Parameter | Value |
|-----------|-------|
| Geometry optimisation | B3LYP/cc-pVTZ (PySCF + pyberny) |
| Active space | FreezeCoreTransformer (1 core orbital for Be) |
| Qubit mapping | JordanWignerMapper (no tapering) |
| Operator pool | Complete αα/ββ/αβ singles + doubles |
| Gradient | Exact commutator `\|⟨ψ\|[H,A]\|ψ⟩\|` via sparse matvec |
| Optimiser | L-BFGS-B with analytic forward-backward gradient |
| State evolution | `scipy.sparse.linalg.expm_multiply` |
| FCI reference | ARPACK sparse eigensolver (`scipy.sparse.linalg.eigsh`) |
| HPC | Forschungszentrum Jülich PGI cluster, ep29th partition |

---

## Comparison with fixed-structure ansätze

For BeH₂ / STO-3G, ADAPT-VQE selects 35 operators from a pool of 92. The full UCCSD ansatz for the same system would include all 92 operators regardless of their gradient contribution. ADAPT-VQE therefore achieves FCI accuracy with ~38% of the full UCCSD parameter count — and the parameters it does select are those with the largest physical contribution, not an arbitrary ordering.

This parameter economy becomes more pronounced for larger systems where many UCCSD excitations contribute negligibly to the correlation energy.

---

## Part of the larger project

- `pes-vqe/` — general PES scanning pipeline
- `ansatz-comparison/` — fixed-structure ansatz benchmarking (UCCSD, SUCCD, PUCCD, UCCGSD, UCCSD_reps2)
- `adapt-vqe/` — this repository
