

# Author: Riya Kayal
# Created: 16/12/2025


"""
spectroscopic_constants.py
==========================
Extract spectroscopic constants and generate extended PES plots.

Computes for each method (VQE ansatz, CCSD, FCI):
  - Re    : equilibrium bond length (Å)
  - De    : dissociation energy (eV)
  - we    : harmonic vibrational frequency (cm⁻¹)
  - wexe  : anharmonicity constant (cm⁻¹)
  - Be    : rotational constant (cm⁻¹)
  - ae    : vibration-rotation coupling (cm⁻¹)

Also generates:
  Plot 5: Spectroscopic constants bar chart
  Plot 6: Polynomial fit quality
  Plot 7: First and second derivatives of PES
  Plot 8: Morse potential fit comparison
  Plot 9: Dissociation energy comparison

Run after plot_pes.py:
    python spectroscopic_constants.py
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import CubicSpline
from collections import defaultdict

# =============================================================================
# CONSTANTS
# =============================================================================
HARTREE_TO_EV     = 27.211396
HARTREE_TO_JOULE  = 4.3597447e-18
BOHR_TO_METER     = 5.2917721e-11
AMU_TO_KG         = 1.66053906e-27
SPEED_OF_LIGHT    = 2.99792458e10    # cm/s
HBAR              = 1.054571817e-34  # J·s
CM_TO_HARTREE     = 4.556335e-6

RESULTS_DIR = "pes_results"
OUTPUT_DIR  = "pes_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "UCCSD":       "steelblue",
    "UCCGSD":      "darkorange",
    "PUCCD":       "seagreen",
    "SUCCD":       "mediumpurple",
    "UCCSD_reps2": "firebrick",
    "CCSD":        "crimson",
    "FCI":         "black",
    "HF":          "gray",
}
MARKERS = {
    "UCCSD":       "o",
    "UCCGSD":      "s",
    "PUCCD":       "^",
    "SUCCD":       "D",
    "UCCSD_reps2": "P",
    "CCSD":        "x",
    "FCI":         "*",
    "HF":          "+",
}

# =============================================================================
# LOAD DATA
# =============================================================================

all_files = sorted(glob.glob(f"{RESULTS_DIR}/*.json"))
if not all_files:
    print(f"No results in {RESULTS_DIR}/")
    exit(1)

by_mol_ansatz = defaultdict(list)
for fpath in all_files:
    with open(fpath) as f:
        d = json.load(f)
    by_mol_ansatz[(d["mol_name"], d["ansatz"])].append(d)

for key in by_mol_ansatz:
    by_mol_ansatz[key].sort(key=lambda x: x["R_angstrom"])

mol_names = sorted(set(k[0] for k in by_mol_ansatz))

# =============================================================================
# SPECTROSCOPIC CONSTANTS FUNCTIONS
# =============================================================================

def reduced_mass_amu(mol_name):
    """Reduced mass in AMU for diatomics."""
    masses = {
        "H2":   {"H": 1.00794},
        "LiH":  {"Li": 6.941,  "H": 1.00794},
        "HF":   {"H": 1.00794, "F": 18.9984},
        "BeH2": {"Be": 9.0122, "H": 1.00794},
        "LiH_H2": None,  # not a simple diatomic
    }
    m = masses.get(mol_name)
    if m is None:
        return None
    atoms = list(m.values())
    if len(atoms) == 2:
        return atoms[0] * atoms[1] / (atoms[0] + atoms[1])
    return None


def morse_potential(R, De, Re, a, E_inf):
    """Morse potential: E(R) = De*(1 - exp(-a*(R-Re)))^2 + E_inf"""
    return De * (1 - np.exp(-a * (R - Re)))**2 + E_inf


def fit_polynomial(R, E, order=10):
    """Fit polynomial of given order, return coefficients."""
    # Centre around R midpoint for numerical stability
    R_mid  = (R.max() + R.min()) / 2
    R_norm = R - R_mid
    coeffs = np.polyfit(R_norm, E, order)
    return coeffs, R_mid


def polynomial_energy(R, coeffs, R_mid):
    return np.polyval(coeffs, R - R_mid)


def polynomial_deriv(R, coeffs, R_mid, order=1):
    """Analytical derivative of fitted polynomial."""
    d_coeffs = np.polyder(coeffs, m=order)
    return np.polyval(d_coeffs, R - R_mid)


def find_minimum(R, E, poly_order=10):
    """Find Re and E(Re) from polynomial fit."""
    coeffs, R_mid = fit_polynomial(R, E, poly_order)
    result = minimize_scalar(
        lambda r: polynomial_energy(r, coeffs, R_mid),
        bounds=(R.min(), R.max()),
        method='bounded'
    )
    Re   = result.x
    E_Re = result.fun
    return Re, E_Re, coeffs, R_mid


def compute_spectroscopic_constants(R_ang, E_ha, mol_name, poly_order=10):
    """
    Compute spectroscopic constants from PES data.

    Parameters
    ----------
    R_ang : np.ndarray  — bond lengths in Angstrom
    E_ha  : np.ndarray  — energies in Hartree
    mol_name : str
    poly_order : int    — polynomial order for fitting

    Returns
    -------
    dict of spectroscopic constants
    """
    mu_amu = reduced_mass_amu(mol_name)
    if mu_amu is None:
        return None

    mu_kg = mu_amu * AMU_TO_KG
    R_m   = R_ang * 1e-10   # Å -> m
    E_J   = E_ha * HARTREE_TO_JOULE

    # ── Polynomial fit ────────────────────────────────────────────────────────
    coeffs, R_mid = fit_polynomial(R_ang, E_ha, poly_order)

    # ── Re: equilibrium bond length ───────────────────────────────────────────
    result = minimize_scalar(
        lambda r: polynomial_energy(r, coeffs, R_mid),
        bounds=(R_ang.min(), R_ang.max()),
        method='bounded'
    )
    Re_ang = result.x
    E_Re   = result.fun

    # ── De: dissociation energy ───────────────────────────────────────────────
    # De = E(R->inf) - E(Re)
    # Use largest R as approximation for E(inf)
    E_inf  = E_ha[-1]                      # largest R point
    De_ha  = E_inf - E_Re
    De_eV  = De_ha * HARTREE_TO_EV

    # ── Force constant k = d²E/dR² at Re (Hartree/Å²) ───────────────────────
    d2_coeffs = np.polyder(coeffs, m=2)
    k_ha_A2   = np.polyval(d2_coeffs, Re_ang - R_mid)  # Ha/Å²

    # Convert: Ha/Å² -> J/m²
    k_Jm2 = k_ha_A2 * HARTREE_TO_JOULE / (1e-10)**2

    # ── ωe: harmonic frequency ────────────────────────────────────────────────
    if k_Jm2 > 0:
        omega_rad = np.sqrt(k_Jm2 / mu_kg)         # rad/s
        we_cm     = omega_rad / (2 * np.pi * SPEED_OF_LIGHT)  # cm⁻¹
    else:
        we_cm = np.nan

    # ── ωexe: anharmonicity from 3rd/4th derivative (perturbation theory) ────
    d3_coeffs = np.polyder(coeffs, m=3)
    d4_coeffs = np.polyder(coeffs, m=4)
    f3_ha_A3  = np.polyval(d3_coeffs, Re_ang - R_mid)
    f4_ha_A4  = np.polyval(d4_coeffs, Re_ang - R_mid)

    # Convert derivatives to SI
    f3_Jm3 = f3_ha_A3 * HARTREE_TO_JOULE / (1e-10)**3
    f4_Jm4 = f4_ha_A4 * HARTREE_TO_JOULE / (1e-10)**4

    if k_Jm2 > 0 and not np.isnan(we_cm):
        hbar_omega = HBAR * omega_rad
        # Dunham perturbation theory formula
        wexe_cm = we_cm * (
            (5/3) * (f3_Jm3**2 / (k_Jm2**3)) * (HBAR / (2 * mu_kg))**0 -
            (7/12) * (f4_Jm4 / k_Jm2**2) * (HBAR / (2 * mu_kg))**0
        ) if (k_Jm2**2 > 0) else np.nan
        # Simplified: wexe ≈ we²/(4De) for Morse
        wexe_cm = we_cm**2 / (4 * De_eV * 8065.54)  # De in cm⁻¹
    else:
        wexe_cm = np.nan

    # ── Be: rotational constant ───────────────────────────────────────────────
    Re_m = Re_ang * 1e-10
    Be_cm = HBAR / (4 * np.pi * SPEED_OF_LIGHT * mu_kg * Re_m**2)

    # ── αe: vibration-rotation coupling ──────────────────────────────────────
    # αe ≈ -6Be²/ωe * (1 + f3*Re/(3*k))  [approximate]
    if we_cm > 0 and k_ha_A2 > 0:
        ae_cm = (6 * Be_cm**2 / we_cm) * (
            (f3_ha_A3 * Re_ang / (3 * k_ha_A2)) - 1
        )
    else:
        ae_cm = np.nan

    return {
        "Re_ang":   round(Re_ang, 6),
        "De_eV":    round(De_eV, 6),
        "De_ha":    round(De_ha, 8),
        "we_cm":    round(we_cm, 4) if not np.isnan(we_cm) else None,
        "wexe_cm":  round(wexe_cm, 4) if not np.isnan(wexe_cm) else None,
        "Be_cm":    round(Be_cm, 6),
        "ae_cm":    round(ae_cm, 6) if not np.isnan(ae_cm) else None,
        "k_ha_A2":  round(k_ha_A2, 6),
        "poly_order": poly_order,
    }


def fit_morse(R_ang, E_ha):
    """Fit Morse potential, return parameters and fitted curve."""
    De_guess  = abs(E_ha[-1] - E_ha.min())
    Re_guess  = R_ang[np.argmin(E_ha)]
    a_guess   = 2.0
    inf_guess = E_ha[-1]

    try:
        popt, _ = curve_fit(
            morse_potential, R_ang, E_ha,
            p0=[De_guess, Re_guess, a_guess, inf_guess],
            bounds=([0, 0.3, 0.1, -np.inf], [10, 5.0, 10, np.inf]),
            maxfev=10000
        )
        return popt
    except Exception:
        return None


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

for mol in mol_names:
    print(f"\n{'='*60}")
    print(f"SPECTROSCOPIC CONSTANTS — {mol}")
    print(f"{'='*60}")

    mol_ansatze = sorted(set(k[1] for k in by_mol_ansatz if k[0] == mol))

    # Build reference arrays from first ansatz
    first_key = (mol, mol_ansatze[0])
    refs       = by_mol_ansatz[first_key]
    R_ref      = np.array([d["R_angstrom"] for d in refs])
    E_ccsd     = np.array([d["e_ccsd"]     for d in refs])
    E_fci      = np.array([d["e_fci"] if d["e_fci"] else np.nan for d in refs])
    E_hf       = np.array([d["e_hf"]       for d in refs])

    # Collect all curves
    all_curves = {
        "HF":   (R_ref, E_hf),
        "CCSD": (R_ref, E_ccsd),
    }
    if not np.all(np.isnan(E_fci)):
        all_curves["FCI"] = (R_ref, E_fci)

    for ans in mol_ansatze:
        pts = by_mol_ansatz[(mol, ans)]
        R_v = np.array([d["R_angstrom"] for d in pts])
        E_v = np.array([d["e_vqe"]      for d in pts])
        all_curves[ans] = (R_v, E_v)

    # ── Compute constants ─────────────────────────────────────────────────────
    all_constants = {}
    for label, (R_v, E_v) in all_curves.items():
        if len(R_v) < 6:
            print(f"  {label}: insufficient points ({len(R_v)}) — skip")
            continue
        if label == "HF":
            continue  # HF constants not interesting
        consts = compute_spectroscopic_constants(R_v, E_v, mol, poly_order=10)
        if consts:
            all_constants[label] = consts
            print(f"\n  {label}:")
            print(f"    Re    = {consts['Re_ang']:.4f} Å")
            print(f"    De    = {consts['De_eV']:.4f} eV  ({consts['De_ha']*1000:.2f} mHa)")
            print(f"    ωe    = {consts['we_cm']:.1f} cm⁻¹" if consts['we_cm'] else "    ωe    = N/A")
            print(f"    ωexe  = {consts['wexe_cm']:.2f} cm⁻¹" if consts['wexe_cm'] else "    ωexe  = N/A")
            print(f"    Be    = {consts['Be_cm']:.4f} cm⁻¹")

    if not all_constants:
        print(f"  No constants computed for {mol}")
        continue

    # ── Plot 5: Spectroscopic constants bar chart ─────────────────────────────
    const_labels = [l for l in all_constants if l != "HF"]
    fig, axes    = plt.subplots(1, 4, figsize=(16, 5))
    bar_colors   = [COLORS.get(l, "black") for l in const_labels]

    def bar_const(ax, vals, title, ylabel, ref_label=None):
        bars = ax.bar(const_labels, vals, color=bar_colors,
                      alpha=0.85, edgecolor='white')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(range(len(const_labels)))
        ax.set_xticklabels(const_labels, rotation=30, ha='right', fontsize=8)
        if ref_label and ref_label in all_constants:
            ref_val = [all_constants[ref_label][
                "Re_ang" if "Re" in title else
                "De_eV"  if "De" in title else
                "we_cm"  if "ωe" in title else
                "Be_cm"
            ]]
            ax.axhline(ref_val[0], color='crimson', ls='--', lw=1.5,
                       label=ref_label)
            ax.legend(fontsize=8)
        for b, v in zip(bars, vals):
            if v is not None:
                ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                        f'{v:.3f}', ha='center', fontsize=7)

    re_vals   = [all_constants[l]["Re_ang"]  for l in const_labels]
    de_vals   = [all_constants[l]["De_eV"]   for l in const_labels]
    we_vals   = [all_constants[l]["we_cm"] or 0 for l in const_labels]
    be_vals   = [all_constants[l]["Be_cm"]   for l in const_labels]

    bar_const(axes[0], re_vals,  'Equilibrium Re',   'Re (Å)',    'FCI')
    bar_const(axes[1], de_vals,  'Dissociation De',  'De (eV)',   'FCI')
    bar_const(axes[2], we_vals,  'Harmonic freq ωe', 'ωe (cm⁻¹)','FCI')
    bar_const(axes[3], be_vals,  'Rotational Be',    'Be (cm⁻¹)','FCI')

    plt.suptitle(f'Spectroscopic Constants — {mol} / STO-3G', fontsize=12)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{mol}_spectroscopic_constants.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"\n  -> {fname}")

    # ── Plot 6: Polynomial fit quality ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label in ["CCSD", "FCI"] + [a for a in mol_ansatze]:
        if label not in all_curves:
            continue
        R_v, E_v = all_curves[label]
        if len(R_v) < 6:
            continue
        c   = COLORS.get(label, "black")
        m   = MARKERS.get(label, "o")

        # Data points
        axes[0].scatter(R_v, E_v, color=c, marker=m, s=50, zorder=3)

        # Polynomial fit
        coeffs, R_mid = fit_polynomial(R_v, E_v, 10)
        R_fine = np.linspace(R_v.min(), R_v.max(), 200)
        E_fit  = polynomial_energy(R_fine, coeffs, R_mid)
        axes[0].plot(R_fine, E_fit, color=c, lw=1.5, label=label)

        # Residuals
        E_at_data = polynomial_energy(R_v, coeffs, R_mid)
        residuals = (E_v - E_at_data) * 1000   # mHa
        axes[1].plot(R_v, residuals, color=c, marker=m,
                     ms=5, lw=1.2, label=label)

    axes[0].set_xlabel('R (Å)')
    axes[0].set_ylabel('Energy (Ha)')
    axes[0].set_title('PES with 10th order polynomial fit')
    axes[0].legend(fontsize=8)

    axes[1].axhline(0, color='black', lw=0.8, ls='--')
    axes[1].set_xlabel('R (Å)')
    axes[1].set_ylabel('Residual (mHa)')
    axes[1].set_title('Polynomial fit residuals')
    axes[1].legend(fontsize=8)

    plt.suptitle(f'Polynomial Fit Quality — {mol} / STO-3G (order=10)', fontsize=11)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{mol}_poly_fit.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")

    # ── Plot 7: First and second derivatives ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label in ["CCSD", "FCI"] + [a for a in mol_ansatze]:
        if label not in all_curves:
            continue
        R_v, E_v = all_curves[label]
        if len(R_v) < 6:
            continue
        c = COLORS.get(label, "black")

        coeffs, R_mid = fit_polynomial(R_v, E_v, 10)
        R_fine        = np.linspace(R_v.min() + 0.05, R_v.max() - 0.05, 200)
        dE_dR         = polynomial_deriv(R_fine, coeffs, R_mid, order=1)
        d2E_dR2       = polynomial_deriv(R_fine, coeffs, R_mid, order=2)

        # Convert to Ha/Å and Ha/Å²
        axes[0].plot(R_fine, dE_dR,   color=c, lw=1.8, label=label)
        axes[1].plot(R_fine, d2E_dR2, color=c, lw=1.8, label=label)

    axes[0].axhline(0, color='black', lw=0.8, ls='--')
    axes[0].set_xlabel('R (Å)')
    axes[0].set_ylabel('dE/dR (Ha/Å)')
    axes[0].set_title('First derivative of PES (force)')
    axes[0].legend(fontsize=8)

    axes[1].axhline(0, color='black', lw=0.8, ls='--')
    axes[1].set_xlabel('R (Å)')
    axes[1].set_ylabel('d²E/dR² (Ha/Å²)')
    axes[1].set_title('Second derivative (force constant)')
    axes[1].legend(fontsize=8)

    plt.suptitle(f'PES Derivatives — {mol} / STO-3G', fontsize=11)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{mol}_derivatives.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")

    # ── Plot 8: Morse potential fit ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    for label in ["CCSD", "FCI"] + [a for a in mol_ansatze]:
        if label not in all_curves:
            continue
        R_v, E_v = all_curves[label]
        if len(R_v) < 6:
            continue
        c    = COLORS.get(label, "black")
        m    = MARKERS.get(label, "o")
        popt = fit_morse(R_v, E_v)

        ax.scatter(R_v, E_v, color=c, marker=m, s=40, zorder=3)
        if popt is not None:
            R_fine = np.linspace(R_v.min(), R_v.max(), 300)
            E_morse = morse_potential(R_fine, *popt)
            ax.plot(R_fine, E_morse, color=c, lw=1.5,
                    label=f"{label} (Morse: Re={popt[1]:.3f}Å)")

    ax.set_xlabel('R (Å)')
    ax.set_ylabel('Energy (Ha)')
    ax.set_title(f'Morse Potential Fit — {mol} / STO-3G')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{mol}_morse_fit.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")

    # ── Plot 9: De comparison ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    de_labels = [l for l in all_constants if l not in ("HF",)]
    de_eV     = [all_constants[l]["De_eV"]  for l in de_labels]
    de_mHa    = [all_constants[l]["De_ha"] * 1000 for l in de_labels]
    de_colors = [COLORS.get(l, "black") for l in de_labels]

    bars0 = axes[0].bar(de_labels, de_eV,  color=de_colors, alpha=0.85,
                        edgecolor='white')
    bars1 = axes[1].bar(de_labels, de_mHa, color=de_colors, alpha=0.85,
                        edgecolor='white')

    for ax, bars, vals, ylabel in [
        (axes[0], bars0, de_eV,  'De (eV)'),
        (axes[1], bars1, de_mHa, 'De (mHa)'),
    ]:
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(de_labels)))
        ax.set_xticklabels(de_labels, rotation=30, ha='right', fontsize=8)
        if "FCI" in all_constants:
            fci_val = all_constants["FCI"]["De_eV" if ylabel == "De (eV)" else "De_ha"]
            if ylabel == "De (mHa)":
                fci_val *= 1000
            ax.axhline(fci_val, color='black', ls='--', lw=1.5, label='FCI')
            ax.legend(fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                    f'{v:.3f}', ha='center', fontsize=7)

    axes[0].set_title('Dissociation energy (eV)')
    axes[1].set_title('Dissociation energy (mHa)')
    plt.suptitle(f'Dissociation Energy Comparison — {mol} / STO-3G', fontsize=12)
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/{mol}_dissociation_energy.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")

    # ── Write constants table ─────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/{mol}_spectroscopic_constants.txt", "w") as f:
        f.write(f"SPECTROSCOPIC CONSTANTS — {mol} / STO-3G\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Method':<15} {'Re(Å)':>8} {'De(eV)':>8} "
                f"{'ωe(cm⁻¹)':>10} {'ωexe':>7} {'Be(cm⁻¹)':>10}\n")
        f.write("-"*70 + "\n")
        for label, c in all_constants.items():
            we   = f"{c['we_cm']:.1f}"   if c['we_cm']   else "N/A"
            wexe = f"{c['wexe_cm']:.2f}" if c['wexe_cm'] else "N/A"
            f.write(f"{label:<15} {c['Re_ang']:>8.4f} {c['De_eV']:>8.4f} "
                    f"{we:>10} {wexe:>7} {c['Be_cm']:>10.4f}\n")

    print(f"  -> {OUTPUT_DIR}/{mol}_spectroscopic_constants.txt")

print(f"\nAll done. Output: {OUTPUT_DIR}/")
