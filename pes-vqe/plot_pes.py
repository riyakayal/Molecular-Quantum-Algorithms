
# Author: Riya Kayal
# Created: 01/12/2025


"""
plot_pes.py
===========
Read all pes_results/*.json and plot PES curves.
Run standalone after all SLURM jobs finish:

    python plot_pes.py
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "pes_results"
OUTPUT_DIR  = "pes_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOAD ALL RESULTS
# =============================================================================

all_files = sorted(glob.glob(f"{RESULTS_DIR}/*.json"))

if not all_files:
    print(f"No results found in {RESULTS_DIR}/")
    exit(1)

# Group by (mol_name, ansatz)
from collections import defaultdict
by_mol_ansatz = defaultdict(list)
by_mol        = defaultdict(list)   # for CCSD/FCI/HF (same for all ansatze)

for fpath in all_files:
    with open(fpath) as f:
        d = json.load(f)
    key = (d["mol_name"], d["ansatz"])
    by_mol_ansatz[key].append(d)
    by_mol[d["mol_name"]].append(d)

# Sort each curve by R
for key in by_mol_ansatz:
    by_mol_ansatz[key].sort(key=lambda x: x["R_angstrom"])

mol_names = sorted(set(k[0] for k in by_mol_ansatz))
ansatze   = sorted(set(k[1] for k in by_mol_ansatz))

print(f"Molecules: {mol_names}")
print(f"Ansatze:   {ansatze}")

COLORS = {
    "UCCSD":       "steelblue",
    "UCCGSD":      "darkorange",
    "PUCCD":       "seagreen",
    "SUCCD":       "mediumpurple",
    "UCCSD_reps2": "firebrick",
    "EfficientSU2":"dimgray",
}
MARKERS = {
    "UCCSD":       "o",
    "UCCGSD":      "s",
    "PUCCD":       "^",
    "SUCCD":       "D",
    "UCCSD_reps2": "P",
    "EfficientSU2":"X",
}

# =============================================================================
# PLOT FOR EACH MOLECULE
# =============================================================================

for mol in mol_names:
    print(f"\nPlotting {mol}...", flush=True)

    # Collect reference curves from first ansatz (same for all)
    ref_points = sorted(
        [d for d in by_mol[mol]],
        key=lambda x: x["R_angstrom"]
    )
    # Deduplicate by R
    seen_R = set()
    refs   = []
    for d in ref_points:
        if d["R_angstrom"] not in seen_R:
            refs.append(d)
            seen_R.add(d["R_angstrom"])
    refs.sort(key=lambda x: x["R_angstrom"])

    R_ref   = np.array([d["R_angstrom"] for d in refs])
    E_hf    = np.array([d["e_hf"]       for d in refs])
    E_ccsd  = np.array([d["e_ccsd"]     for d in refs])
    E_fci   = np.array([d["e_fci"] if d["e_fci"] else np.nan for d in refs])

    # ── Plot 1: Absolute PES ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Reference curves
    axes[0].plot(R_ref, E_hf,   'k:',  lw=1.5, label='HF')
    axes[0].plot(R_ref, E_ccsd, 'r--', lw=2.0, label='CCSD')
    if not np.all(np.isnan(E_fci)):
        axes[0].plot(R_ref, E_fci, 'k-', lw=1.5, label='FCI')

    # VQE curves
    mol_ansatze = sorted(set(k[1] for k in by_mol_ansatz if k[0] == mol))
    for ans in mol_ansatze:
        pts = by_mol_ansatz[(mol, ans)]
        R_v = np.array([d["R_angstrom"] for d in pts])
        E_v = np.array([d["e_vqe"]      for d in pts])
        c   = COLORS.get(ans, "black")
        m   = MARKERS.get(ans, "o")
        axes[0].plot(R_v, E_v, color=c, marker=m, ms=6, lw=1.8,
                     label=f"VQE-{ans}")

    axes[0].set_ylabel('Total Energy (Ha)')
    axes[0].set_title(f'Potential Energy Surface — {mol} / STO-3G')
    axes[0].legend(fontsize=9, ncol=2)

    # ── Plot 2: Error vs FCI ────────────────────────────────────────────────
    axes[1].axhline(0,   color='crimson', ls='--', lw=1.5, label='FCI')
    axes[1].axhline(1.0, color='gray',   ls=':',  lw=1.0, label='1 mHa')
    axes[1].axhline(-1.0,color='gray',   ls=':',  lw=1.0)

    for ans in mol_ansatze:
        pts   = by_mol_ansatz[(mol, ans)]
        R_v   = np.array([d["R_angstrom"]     for d in pts])
        err_v = np.array([d["error_fci_mha"] for d in pts])
        c     = COLORS.get(ans, "black")
        m     = MARKERS.get(ans, "o")
        axes[1].plot(R_v, err_v, color=c, marker=m, ms=6, lw=1.8,
                     label=f"VQE-{ans}")

    axes[1].set_xlabel('Bond length R (Å)')
    axes[1].set_ylabel('Error vs FCI (mHa)')
    axes[1].legend(fontsize=9, ncol=2)

    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{mol}_pes.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")

    # ── Plot 3: Correlation recovery vs R ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(100, color='crimson', ls='--', lw=1.5, label='CCSD (100%)')
    ax.axhline(90,  color='gray',   ls=':',  lw=1.0, label='90%')

    for ans in mol_ansatze:
        pts    = by_mol_ansatz[(mol, ans)]
        R_v    = np.array([d["R_angstrom"] for d in pts])
        corr_v = np.array([d["corr_pct"]   for d in pts])
        c      = COLORS.get(ans, "black")
        m      = MARKERS.get(ans, "o")
        ax.plot(R_v, corr_v, color=c, marker=m, ms=6, lw=1.8, label=f"VQE-{ans}")

    ax.set_xlabel('Bond length R (Å)')
    ax.set_ylabel('Correlation recovered (%)')
    ax.set_title(f'Correlation Recovery vs Bond Length — {mol} / STO-3G')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{mol}_correlation.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")

    # ── Plot 4: Relative PES (subtract minimum) ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(R_ref, (E_ccsd - E_ccsd.min())*1000, 'r--', lw=2.0, label='CCSD')
    if not np.all(np.isnan(E_fci)):
        ax.plot(R_ref, (E_fci - np.nanmin(E_fci))*1000, 'k-', lw=1.5,
                label='FCI')

    for ans in mol_ansatze:
        pts = by_mol_ansatz[(mol, ans)]
        R_v = np.array([d["R_angstrom"] for d in pts])
        E_v = np.array([d["e_vqe"]      for d in pts])
        c   = COLORS.get(ans, "black")
        m   = MARKERS.get(ans, "o")
        ax.plot(R_v, (E_v - E_v.min())*1000, color=c, marker=m,
                ms=6, lw=1.8, label=f"VQE-{ans}")

    ax.set_xlabel('Bond length R (Å)')
    ax.set_ylabel('Relative Energy (mHa)')
    ax.set_title(f'Relative PES — {mol} / STO-3G (each shifted to own minimum)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{mol}_pes_relative.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")

    # ── Results table ─────────────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/{mol}_summary.txt", "w") as f:
        f.write(f"PES SUMMARY — {mol} / STO-3G\n")
        f.write("="*80 + "\n\n")
        header = (f"{'R(Å)':>6} {'E_HF':>14} {'E_CCSD':>14} "
                  f"{'E_FCI':>14} ")
        for ans in mol_ansatze:
            header += f"{'E_'+ans:>14} {'err(mHa)':>10} "
        f.write(header + "\n")
        f.write("-"*80 + "\n")

        for r in sorted(seen_R):
            ref = next(d for d in refs if d["R_angstrom"] == r)
            row = (f"{r:>6.3f} {ref['e_hf']:>14.8f} "
                   f"{ref['e_ccsd']:>14.8f} "
                   f"{ref['e_fci']:>14.8f} "
                   if ref['e_fci'] else
                   f"{r:>6.3f} {ref['e_hf']:>14.8f} "
                   f"{ref['e_ccsd']:>14.8f} {'N/A':>14} ")
            for ans in mol_ansatze:
                pts = by_mol_ansatz.get((mol, ans), [])
                pt  = next((d for d in pts if d["R_angstrom"] == r), None)
                if pt:
                    row += f"{pt['e_vqe']:>14.8f} {pt['error_ccsd_mha']:>+10.3f} "
                else:
                    row += f"{'N/A':>14} {'N/A':>10} "
            f.write(row + "\n")

    print(f"  -> {OUTPUT_DIR}/{mol}_summary.txt")

print("\nDone. Output:", OUTPUT_DIR)
