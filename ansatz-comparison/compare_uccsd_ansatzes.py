#!/usr/bin/env python3
# =============================================================================
# ANSATZ COMPARISON — reads all ansatz_results/*.json and makes plots
# Run this standalone after all VQE jobs have finished:
#   python compare_ansatzes.py
# =============================================================================

# Author: Riya Kayal
# Created: 30/12/2025

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = "ansatz_results"
OUTPUT_DIR  = "ansatz_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Expected ansatze — script reports which are missing
EXPECTED = ["UCCSD", "UCCGSD", "PUCCD", "SUCCD", "UCCSD_reps2", "EfficientSU2"]

# Color and marker map — extended for 6 ansatze
COLORS = {
    "UCCSD":        "steelblue",
    "UCCGSD":       "darkorange",
    "PUCCD":        "seagreen",
    "SUCCD":        "mediumpurple",
    "UCCSD_reps2":  "firebrick",
    "EfficientSU2": "dimgray",
}
MARKERS = {
    "UCCSD":        "o",
    "UCCGSD":       "s",
    "PUCCD":        "^",
    "SUCCD":        "D",
    "UCCSD_reps2":  "P",
    "EfficientSU2": "X",
}

# =============================================================================
# LOAD DATA
# =============================================================================

all_files = sorted(glob.glob(f"{RESULTS_DIR}/*.json"))
data      = {}

print("="*60)
print("ANSATZ COMPARISON")
print("="*60)

for fpath in all_files:
    with open(fpath) as f:
        d = json.load(f)
    label = d["label"]
    data[label] = d
    print(f"  Loaded: {label:15s}  "
          f"params={d['n_params']:4d}  "
          f"error={d['error_mha']:+8.3f} mHa  "
          f"corr={d['corr_recovered_pct']:6.2f}%")

missing = [e for e in EXPECTED if e not in data]
if missing:
    print(f"\n  Missing: {missing}")
    print("  Plots will be made with available results.\n")

if not data:
    print("No results found. Run claude01.py for each ansatz first.")
    exit(1)

labels   = list(data.keys())
results  = list(data.values())
n        = len(results)
colors   = [COLORS.get(l, "black") for l in labels]
markers  = [MARKERS.get(l, "o")    for l in labels]

# Reference values from first result (same for all)
E_CCSD = results[0]["e_ccsd_ha"]
#E_FCI  = results[0]["e_fci_ha"]
E_FCI = -9.0197159570 
E_HF   = results[0]["e_hf_ha"]

# =============================================================================
# PLOT 1: CONVERGENCE — running minimum energy vs evaluations
# =============================================================================
print("\n[Plot 1] Convergence...", flush=True)

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)

for res in results:
    label = res["label"]
    c     = COLORS.get(label, "black")
    iters = np.array(res["eval_counts"])
    vals  = np.array(res["eval_energies"])
    rmin  = np.minimum.accumulate(vals)

    axes[0].plot(iters, rmin, color=c, lw=2.0, label=label)
    axes[1].plot(iters, (rmin - E_CCSD)*1000, color=c, lw=2.0, label=label)

axes[0].axhline(E_CCSD, color='crimson', ls='--', lw=1.5, label='CCSD')
axes[0].axhline(E_FCI,  color='black',   ls=':',  lw=1.0, label='FCI')
axes[0].axhline(E_HF,   color='gray',    ls=':',  lw=1.0, label='HF')
axes[0].set_ylabel('Total Energy (Ha)')
axes[0].set_title('VQE Convergence — LiH·H₂ / STO-3G (running minimum)')
axes[0].legend(fontsize=9, ncol=2)

axes[1].axhline(0,   color='crimson', ls='--', lw=1.5, label='CCSD')
axes[1].axhline(1.0, color='gray',    ls=':',  lw=1.0, label='1 mHa')
axes[1].set_ylabel('Error vs CCSD (mHa)')
axes[1].set_xlabel('Evaluations')
axes[1].legend(fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_convergence.png", dpi=150)
plt.close()
print(f"  -> {OUTPUT_DIR}/1_convergence.png")

# =============================================================================
# PLOT 2: RESOURCE BARS — params, depth, error, corr recovered
# =============================================================================
print("[Plot 2] Resource bars...", flush=True)

n_params  = [r["n_params"]           for r in results]
depths    = [r["circuit_depth"]      for r in results]
errors = [abs(r["error_mha"]) for r in results]
corr_pct  = [r["corr_recovered_pct"] for r in results]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

def bar(ax, vals, title, ylabel, log=False):
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    if log:
        ax.set_yscale('log')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() * (1.05 if not log else 1.3),
                f'{v:.1f}', ha='center', fontsize=7)

bar(axes[0], n_params, 'Parameter count',       'Parameters')
bar(axes[1], depths,   'Transpiled circuit depth','Gates')
bar(axes[2], errors,   'Error vs CCSD',          '|ΔE| (mHa)', log=True)
bar(axes[3], corr_pct, 'Correlation recovered',  '%')

plt.suptitle('Resource Comparison — LiH·H₂ / STO-3G', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_resources.png", dpi=150)
plt.close()
print(f"  -> {OUTPUT_DIR}/2_resources.png")

# =============================================================================
# PLOT 3: ACCURACY vs COST (scatter)
# =============================================================================
print("[Plot 3] Accuracy vs cost...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for res, c, m in zip(results, colors, markers):
    e = abs(res["error_mha"])
    axes[0].scatter(res["n_params"],      e, color=c, marker=m, s=120,
                    zorder=3, label=res["label"])
    axes[1].scatter(res["circuit_depth"], e, color=c, marker=m, s=120, zorder=3)
    axes[2].scatter(res["corr_recovered_pct"], e, color=c, marker=m,
                    s=120, zorder=3)

for ax in axes[:2]:
    ax.axhline(1.0, color='gray', ls=':', lw=1.0, label='1 mHa')
    ax.set_ylabel('|Error vs CCSD| (mHa)')
    ax.set_yscale('log')

axes[0].set_xlabel('Parameter count')
axes[0].set_title('Accuracy vs Parameters')
axes[0].legend(fontsize=8, ncol=1)

axes[1].set_xlabel('Circuit depth')
axes[1].set_title('Accuracy vs Circuit depth')

axes[2].set_xlabel('Correlation recovered (%)')
axes[2].set_ylabel('|Error vs CCSD| (mHa)')
axes[2].set_title('Accuracy vs Correlation recovery')
axes[2].set_yscale('log')
axes[2].axhline(1.0, color='gray', ls=':', lw=1.0)

plt.suptitle('Accuracy–Cost Tradeoff — LiH·H₂ / STO-3G', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_accuracy_vs_cost.png", dpi=150)
plt.close()
print(f"  -> {OUTPUT_DIR}/3_accuracy_vs_cost.png")

# =============================================================================
# PLOT 4: ENERGY LEVEL DIAGRAM
# =============================================================================
print("[Plot 4] Energy level diagram...", flush=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Reference levels
for e_val, lbl, col, ls in [
    (E_HF,   'HF',   'gray',   '--'),
    (E_CCSD, 'CCSD', 'crimson','--'),
    (E_FCI,  'FCI',  'black',  ':'),
]:
    ax.axhline(e_val, color=col, ls=ls, lw=1.5, label=lbl)

# VQE results as horizontal ticks
x_positions = np.linspace(0.1, 0.9, n)
for i, (res, c, m) in enumerate(zip(results, colors, markers)):
    e_f = res["e_vqe_ha"]
    e_b = res["e_vqe_best_ha"]
    ax.scatter(x_positions[i], e_f, color=c, marker=m, s=150,
               zorder=4, label=res["label"])
    ax.plot([x_positions[i]-0.03, x_positions[i]+0.03], [e_b, e_b],
            color=c, lw=2.5, zorder=3)
    ax.annotate(res["label"], (x_positions[i], e_f),
                textcoords="offset points", xytext=(0, 8),
                ha='center', fontsize=7, color=c)

ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_ylabel('Total Energy (Ha)')
ax.set_title('Energy Level Diagram — LiH·H₂ / STO-3G\n'
             '(dot=final, bar=best seen)')
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_energy_levels.png", dpi=150)
plt.close()
print(f"  -> {OUTPUT_DIR}/4_energy_levels.png")

# =============================================================================
# PLOT 5: CCSD INITIAL POINT QUALITY
# =============================================================================
print("[Plot 5] CCSD initial point quality...", flush=True)

e_inits   = [r["e_ccsd_init_ha"]    for r in results]
nonzeros  = [r["ccsd_init_nonzero"] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

bars0 = axes[0].bar(labels, [(e - E_HF)*1000 for e in e_inits],
                    color=colors, alpha=0.85, edgecolor='white')
axes[0].axhline((E_CCSD - E_HF)*1000, color='crimson', ls='--',
                lw=1.5, label='CCSD')
axes[0].set_ylabel('Energy below HF (mHa)')
axes[0].set_title('Initial point energy\n(how close to CCSD before VQE starts)')
axes[0].set_xticks(range(n))
axes[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
axes[0].legend(fontsize=9)
for b, v in zip(bars0, e_inits):
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height()*1.02,
                 f'{(v-E_HF)*1000:.1f}', ha='center', fontsize=7)

bars1 = axes[1].bar(labels,
                    [nz/r["n_params"]*100 for nz, r in zip(nonzeros, results)],
                    color=colors, alpha=0.85, edgecolor='white')
axes[1].set_ylabel('Non-zero parameters (%)')
axes[1].set_title('CCSD amplitude coverage\n(% of UCCSD params with non-zero CCSD guess)')
axes[1].set_xticks(range(n))
axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
for b, nz, r in zip(bars1, nonzeros, results):
    pct = nz/r["n_params"]*100
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height()*1.02,
                 f'{pct:.0f}%', ha='center', fontsize=7)

plt.suptitle('CCSD Initial Point Quality', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/5_ccsd_init_quality.png", dpi=150)
plt.close()
print(f"  -> {OUTPUT_DIR}/5_ccsd_init_quality.png")

# =============================================================================
# PLOT 6: RADAR CHART (if >= 3 results)
# =============================================================================
if n >= 3:
    print("[Plot 6] Radar chart...", flush=True)

    categories = ['Accuracy\n(inv. error)', 'Parameter\nefficiency',
                  'Circuit\ncompactness', 'Correlation\nrecovery',
                  'Init point\nquality']

    def norm_inv(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [1.0] * len(vals)
        return [1.0 - (v - mn)/(mx - mn) for v in vals]

    def norm(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [1.0] * len(vals)
        return [(v - mn)/(mx - mn) for v in vals]

    scores = list(zip(
        norm_inv([abs(r["error_mha"])    for r in results]),
        norm_inv([r["n_params"]                for r in results]),
        norm_inv([r["circuit_depth"]           for r in results]),
        norm([r["corr_recovered_pct"]          for r in results]),
        norm([(r["e_ccsd_init_ha"] - E_HF)     for r in results]),
    ))

    N_cat = len(categories)
    theta = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
    theta += theta[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for res, score, c, m in zip(results, scores, colors, markers):
        vals = list(score) + [score[0]]
        ax.plot(theta, vals, color=c, lw=2.0, marker=m, ms=7,
                label=res["label"])
        ax.fill(theta, vals, color=c, alpha=0.07)

    ax.set_thetagrids(np.degrees(theta[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25','0.50','0.75','1.00'], fontsize=7)
    ax.set_title('Multi-dimensional Ansatz Comparison\n(higher = better)',
                 pad=20, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.2), fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_radar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR}/6_radar.png")

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
print("\n[Writing summary...]", flush=True)

with open(f"{OUTPUT_DIR}/results_summary.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("VQE ANSATZ COMPARISON — LiH·H₂ / STO-3G\n")
    f.write("="*80 + "\n\n")
    f.write(f"HF   reference: {E_HF:.10f} Ha\n")
    f.write(f"CCSD reference: {E_CCSD:.10f} Ha\n")
    f.write(f"FCI  reference: {E_FCI:.10f} Ha\n\n")

    # Main table
    f.write(f"{'Ansatz':<15} {'Params':>7} {'Depth':>7} "
            f"{'VQE Final (Ha)':>16} {'Error (mHa)':>12} "
            f"{'Best (mHa)':>11} {'Corr (%)':>9}\n")
    f.write("-"*80 + "\n")
    for res in results:
        f.write(f"{res['label']:<15} {res['n_params']:>7} "
                f"{res['circuit_depth']:>7} "
                f"{res['e_vqe_ha']:>16.10f} "
                f"{res['error_mha']:>+12.3f} "
                f"{'OK' if res['best_is_physical'] else 'UNPHYSICAL':>11} "
                f"{res['corr_recovered_pct']:>9.2f}\n")

    # CCSD initial point table
    f.write("\nCCSD Initial Point Quality:\n")
    f.write(f"{'Ansatz':<15} {'Non-zero':>9} {'Coverage%':>10} "
            f"{'E_init (Ha)':>14} {'ΔE_init (mHa)':>14}\n")
    f.write("-"*65 + "\n")
    for res in results:
        cov = res['ccsd_init_nonzero'] / res['n_params'] * 100
        de  = (res['e_ccsd_init_ha'] - E_CCSD) * 1000
        f.write(f"{res['label']:<15} {res['ccsd_init_nonzero']:>9} "
                f"{cov:>10.1f} {res['e_ccsd_init_ha']:>14.10f} "
                f"{de:>+14.3f}\n")

    # Missing
    if missing:
        f.write(f"\nMissing ansatze (not yet run): {missing}\n")

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump({
        "references": {"e_hf": E_HF, "e_ccsd": E_CCSD, "e_fci": E_FCI},
        "results": results
    }, f, indent=2)

print(f"  -> {OUTPUT_DIR}/results_summary.txt")
print(f"  -> {OUTPUT_DIR}/results.json")

print("\n" + "="*60)
print(f"COMPARISON COMPLETE — {n}/{len(EXPECTED)} ansatze")
if missing:
    print(f"Still missing: {missing}")
print(f"Output: {OUTPUT_DIR}/")
print("="*60)
