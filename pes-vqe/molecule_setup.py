

# Author: Riya Kayal
# Created: 25/10/2025


"""
molecule_setup.py
=================
Molecule configuration and classical reference calculations for VQE pipeline.

    from molecule_setup import setup_molecule, MOLECULES

Usage:
    mol_data = setup_molecule("LiH_H2")          # use preset
    mol_data = setup_molecule("custom",           # use custom geometry
                              geometry="H 0 0 0; F 0 0 0.917",
                              basis="sto-3g")
"""

import os
import numpy as np
from pyscf import gto, scf, cc, fci

# =============================================================================
# PRESET MOLECULES
# Geometry in Angstrom, semicolon-separated
# =============================================================================

MOLECULES = {

    "LiH_H2": {
        "geometry": (
            "H  0.000000  0.000000   -4.975260; "
            "H  0.000000  0.000000   -4.232370; "
            "Li 0.000000  0.000000    7.478690; "
            "H  0.000000  0.000000    9.068939"
        ),
        "basis":   "sto-3g",
        "charge":  0,
        "spin":    0,
        "description": "LiH·H2 complex — intermolecular interaction benchmark",
    },

    "H2_HF": {
        "geometry": (
            "H  0.000  0.000  0.000; "
            "H  0.000  0.000  0.740; "
            "H  0.000  0.000  2.740; "
            "F  0.000  0.000  3.657"
        ),
        "basis":   "sto-3g",
        "charge":  0,
        "spin":    0,
        "description": "H2·HF hydrogen bond complex",
    },

    "H2O": {
        "geometry": (
            "O  0.000000  0.000000  0.117176; "
            "H  0.000000  0.757306 -0.468706; "
            "H  0.000000 -0.757306 -0.468706"
        ),
        "basis":   "sto-3g",
        "charge":  0,
        "spin":    0,
        "description": "Water — strongly correlated benchmark",
    },

    "H2": {
        "geometry": (
            "H  0.000  0.000  0.000; "
            "H  0.000  0.000  0.735"
        ),
        "basis":   "sto-3g",
        "charge":  0,
        "spin":    0,
        "description": "H2 — minimal benchmark",
    },

    "LiH": {
        "geometry": (
            "Li 0.000  0.000  0.000; "
            "H  0.000  0.000  1.596"
        ),
        "basis":   "sto-3g",
        "charge":  0,
        "spin":    0,
        "description": "LiH — standard VQE benchmark",
    },

    "BeH2": {
        "geometry": (
            "Be 0.000  0.000  0.000; "
            "H  0.000  0.000  1.334; "
            "H  0.000  0.000 -1.334"
        ),
        "basis":   "sto-3g",
        "charge":  0,
        "spin":    0,
        "description": "BeH2 — linear triatomic benchmark",
    },

    "HF": {
        "geometry": "H  0.000  0.000  0.000; F  0.000  0.000  0.917",
        "basis":   "sto-3g",
        "charge":  0,
        "spin":    0,
        "description": "HF — strong correlation reference",
    },
}


# =============================================================================
# MAIN SETUP FUNCTION
# =============================================================================

def setup_molecule(
    name,
    geometry=None,
    basis=None,
    charge=None,
    spin=None,
    run_fci=True,
    conv_tol_rhf=1e-12,
    conv_tol_grad_rhf=1e-10,
    conv_tol_ccsd=1e-10,
    conv_tol_normt_ccsd=1e-8,
    output_dir=".",
    verbose=True,
):
    """
    Run RHF + CCSD + FCI for a molecule and return all data needed for VQE.

    Parameters
    ----------
    name : str
        Key from MOLECULES dict, or "custom" when providing geometry directly.
    geometry : str, optional
        Semicolon-separated XYZ string (Angstrom). Required if name="custom".
    basis : str, optional
        Basis set. Overrides preset if provided.
    charge : int, optional
        Molecular charge. Overrides preset if provided.
    spin : int, optional
        Spin multiplicity (2S). Overrides preset if provided.
    run_fci : bool
        Whether to run FCI (exact diag). Can be slow for large systems.
    conv_tol_rhf : float
        RHF energy convergence threshold.
    conv_tol_grad_rhf : float
        RHF gradient convergence threshold.
    conv_tol_ccsd : float
        CCSD energy convergence threshold.
    conv_tol_normt_ccsd : float
        CCSD amplitude convergence threshold.
    output_dir : str
        Directory to save XYZ file.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        name, geometry_str, basis, charge, spin,
        clean_atom_list,
        mol, mf, cc_solver,
        e_hf, e_ccsd, e_fci (None if run_fci=False),
        t1, t2, nocc, nvirt,
        description
    """

    def _print(msg):
        if verbose:
            print(msg, flush=True)

    # ── Resolve config ────────────────────────────────────────────────────────
    if name == "custom":
        if geometry is None:
            raise ValueError("geometry must be provided when name='custom'")
        cfg = {
            "geometry":    geometry,
            "basis":       basis or "sto-3g",
            "charge":      charge if charge is not None else 0,
            "spin":        spin   if spin   is not None else 0,
            "description": "Custom molecule",
        }
    elif name in MOLECULES:
        cfg = dict(MOLECULES[name])
        if basis  is not None: cfg["basis"]  = basis
        if charge is not None: cfg["charge"] = charge
        if spin   is not None: cfg["spin"]   = spin
    else:
        raise ValueError(
            f"Unknown molecule '{name}'. "
            f"Available: {list(MOLECULES.keys())} or 'custom'."
        )

    geom_str = cfg["geometry"]
    basis_   = cfg["basis"]
    charge_  = cfg["charge"]
    spin_    = cfg["spin"]

    _print("\n" + "="*60)
    _print(f"MOLECULE SETUP: {name}")
    _print(f"  Basis:  {basis_}  Charge: {charge_}  Spin: {spin_}")
    _print(f"  {cfg['description']}")
    _print("="*60)

    # ── Build PySCF molecule ──────────────────────────────────────────────────
    mol = gto.M(
        atom=geom_str,
        basis=basis_,
        charge=charge_,
        spin=spin_,
        unit='Angstrom',
        verbose=0,
    )

    # Clean atom list for downstream use (qiskit driver etc.)
    atoms  = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords = mol.atom_coords() * 0.529177210903   # Bohr -> Angstrom
    clean_atom_list = [[atoms[i], tuple(coords[i])] for i in range(len(atoms))]

    # Save XYZ
    os.makedirs(output_dir, exist_ok=True)
    xyz_path = os.path.join(output_dir, f"{name}.xyz")
    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n{name} / {basis_}\n")
        for atom, pos in clean_atom_list:
            f.write(f"{atom:2} {pos[0]:15.10f} {pos[1]:15.10f} {pos[2]:15.10f}\n")
    _print(f"  Geometry saved: {xyz_path}")

    # ── RHF ───────────────────────────────────────────────────────────────────
    _print("\n[RHF]")
    mf = scf.RHF(mol)
    mf.conv_tol      = conv_tol_rhf
    mf.conv_tol_grad = conv_tol_grad_rhf
    mf.verbose       = 0
    mf.run()
    e_hf = mf.e_tot
    _print(f"  E(HF)   = {e_hf:.10f} Ha")

    # ── CCSD ──────────────────────────────────────────────────────────────────
    _print("[CCSD]")
    cc_solver = cc.CCSD(mf)
    cc_solver.conv_tol       = conv_tol_ccsd
    cc_solver.conv_tol_normt = conv_tol_normt_ccsd
    cc_solver.verbose        = 0
    cc_solver.run()
    e_ccsd = cc_solver.e_tot
    t1     = cc_solver.t1
    t2     = cc_solver.t2
    nocc   = cc_solver.nocc
    nvirt  = cc_solver.nmo - cc_solver.nocc
    _print(f"  E(CCSD) = {e_ccsd:.10f} Ha")
    _print(f"  E_corr  = {cc_solver.e_corr:.10f} Ha")
    _print(f"  nocc={nocc}  nvirt={nvirt}  "
           f"max|T1|={np.max(np.abs(t1)):.6f}  "
           f"max|T2|={np.max(np.abs(t2)):.6f}")

    # ── FCI ───────────────────────────────────────────────────────────────────
    if run_fci:
        _print("[FCI]")
        try:
            from pyscf import fci
            cisolver = fci.FCI(mf)   # pass mf, not mol — handles integrals correctly
            cisolver.verbose = 0
            e_fci, _ = cisolver.kernel()  # returns total energy (electronic + nuclear)
            _print(f"  E(FCI)  = {e_fci:.10f} Ha")
        except Exception as ex:
            _print(f"  FCI failed: {ex} — skipping")
            e_fci = None

    # ── Summary ───────────────────────────────────────────────────────────────
    _print("\n[Summary]")
    _print(f"  E(HF)   = {e_hf:.10f} Ha")
    _print(f"  E(CCSD) = {e_ccsd:.10f} Ha")
    if e_fci is not None:
        _print(f"  E(FCI)  = {e_fci:.10f} Ha")
        _print(f"  FCI-CCSD gap = {(e_fci - e_ccsd)*1000:.4f} mHa")
    _print(f"  Correlation energy (CCSD): {cc_solver.e_corr*1000:.4f} mHa")

    return {
        "name":            name,
        "geometry_str":    geom_str,
        "basis":           basis_,
        "charge":          charge_,
        "spin":            spin_,
        "description":     cfg["description"],
        "clean_atom_list": clean_atom_list,
        "mol":             mol,
        "mf":              mf,
        "cc_solver":       cc_solver,
        "e_hf":            e_hf,
        "e_ccsd":          e_ccsd,
        "e_fci":           e_fci,
        "t1":              t1,
        "t2":              t2,
        "nocc":            nocc,
        "nvirt":           nvirt,
    }


# =============================================================================
# CCSD AMPLITUDE -> UCCSD INITIAL POINT MAPPING
# =============================================================================

def ccsd_initial_point(t1, t2, nocc, nvirt, ansatz, n_spatial_orbs):
    """
    Map CCSD T1/T2 amplitudes to UCCSD parameter ordering.

    Parameters
    ----------
    t1 : np.ndarray, shape (nocc, nvirt)
    t2 : np.ndarray, shape (nocc, nocc, nvirt, nvirt)
    nocc : int
    nvirt : int
    ansatz : UCCSD (pre-transpilation)
    n_spatial_orbs : int

    Returns
    -------
    np.ndarray of shape (ansatz.num_parameters,)
    """
    N_SP            = n_spatial_orbs
    excitation_list = ansatz.excitation_list
    initial_point   = np.zeros(len(excitation_list))

    for idx, (occ_so, virt_so) in enumerate(excitation_list):

        if len(occ_so) == 1 and len(virt_so) == 1:
            i_so = occ_so[0];  a_so = virt_so[0]
            i_sp   = i_so if i_so < N_SP else i_so - N_SP
            i_spin = 0    if i_so < N_SP else 1
            a_sp   = a_so if a_so < N_SP else a_so - N_SP
            a_spin = 0    if a_so < N_SP else 1
            a_v    = a_sp - nocc
            if i_spin != a_spin: continue
            if not (0 <= i_sp < nocc and 0 <= a_v < nvirt): continue
            initial_point[idx] = t1[i_sp, a_v]

        elif len(occ_so) == 2 and len(virt_so) == 2:
            i_so = occ_so[0];  j_so = occ_so[1]
            a_so = virt_so[0]; b_so = virt_so[1]
            i_sp   = i_so if i_so < N_SP else i_so - N_SP
            i_spin = 0    if i_so < N_SP else 1
            j_sp   = j_so if j_so < N_SP else j_so - N_SP
            j_spin = 0    if j_so < N_SP else 1
            a_sp   = a_so if a_so < N_SP else a_so - N_SP
            a_spin = 0    if a_so < N_SP else 1
            b_sp   = b_so if b_so < N_SP else b_so - N_SP
            b_spin = 0    if b_so < N_SP else 1
            a_v    = a_sp - nocc
            b_v    = b_sp - nocc
            if i_spin != a_spin or j_spin != b_spin: continue
            if not (0 <= i_sp < nocc and 0 <= j_sp < nocc and
                    0 <= a_v  < nvirt and 0 <= b_v  < nvirt): continue
            initial_point[idx] = t2[i_sp, j_sp, a_v, b_v]

    return initial_point


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing molecule_setup.py...\n")
    for mol_name in ["H2", "LiH", "LiH_H2"]:
        d = setup_molecule(mol_name, run_fci=True)
        fci_val = f"{d['e_fci']:.6f}" if d['e_fci'] is not None else "N/A"
        print(f"\n{mol_name}: HF={d['e_hf']:.6f}  ",\
              f"CCSD={d['e_ccsd']:.6f}  ",\
              f"FCI={fci_val}\n")
