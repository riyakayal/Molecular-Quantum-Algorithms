#!/bin/bash
# =============================================================================
# submit_pes.sh — submits individual SLURM jobs for each (R, ansatz) point
# No job arrays needed.
#
# Usage:
#   bash submit_pes.sh
# =============================================================================

# Author: Riya Kayal
# Created: 15/12/2025


# ── Configure here ────────────────────────────────────────────────────────────
MOL_NAME="LiH"
BASIS="sto-3g"
ANSATZE=("UCCSD" "SUCCD" "PUCCD")


# R values in Angstrom
#LiH
R_VALUES=(1.2 1.25 1.3 1.35 1.4 0.145 1.5 0.155 1.596 1.65 1.7 1.75 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0)
WORKDIR=$(pwd)
# Change these following 3 lines based on your system
VENV="/path_to_your_virtual_env_python3.9"
PARTITION="partitionId"
NODE="node"
MEM="8G"

mkdir -p ${WORKDIR}/pes_results
mkdir -p ${WORKDIR}/pes_logs

# ── Submit one job per (R, ansatz) ────────────────────────────────────────────
N_SUBMITTED=0
N_SKIPPED=0

for ANSATZ in "${ANSATZE[@]}"; do
    for R in "${R_VALUES[@]}"; do

        # Check if result already exists — skip if so
        OUTFILE="${WORKDIR}/pes_results/${MOL_NAME}_${ANSATZ}_R${R}.json"
        # Normalize R to 4 decimal places for filename matching
        R_FMT=$(python3 -c "print(f'{float(\"${R}\"):.4f}')")
        OUTFILE="${WORKDIR}/pes_results/${MOL_NAME}_${ANSATZ}_R${R_FMT}.json"

        if [ -f "${OUTFILE}" ]; then
            echo "SKIP: ${MOL_NAME} R=${R} ${ANSATZ} (already done)"
            N_SKIPPED=$((N_SKIPPED + 1))
            continue
        fi

        JOBNAME="pes_${MOL_NAME}_${ANSATZ}_R${R}"
        LOGFILE="${WORKDIR}/pes_logs/${JOBNAME}.out"
        ERRFILE="${WORKDIR}/pes_logs/$R.err"

            
        sbatch \
            --job-name="${JOBNAME}" \
            --partition="${PARTITION}" \
            --nodelist="${NODE}" 
            --cpus-per-task=1 \
            --mem="${MEM}" \
            --output="${LOGFILE}" \
            --error="${ERRFILE}" \
            --export=ALL,PES_R="${R}",MOL_NAME="${MOL_NAME}",ANSATZ="${ANSATZ}",BASIS="${BASIS}",PES_OUTPUT_DIR="${WORKDIR}/pes_results" \
            --wrap="
                export OMP_NUM_THREADS=1
                export GOMP_NUM_THREADS=1
                export MKL_NUM_THREADS=1
                export OPENBLAS_NUM_THREADS=1
                cd ${WORKDIR}
                source /hsnfs/users/kayal/venv_vqe/bin/activate
                ${VENV} pes_point.py
            "

        echo "SUBMITTED: ${MOL_NAME} R=${R} ${ANSATZ} -> ${LOGFILE}"
        N_SUBMITTED=$((N_SUBMITTED + 1))

        # Small delay to avoid overwhelming the scheduler
        sleep 0.2

    done
done

echo ""
echo "========================================"
echo "Submitted: ${N_SUBMITTED} jobs"
echo "Skipped:   ${N_SKIPPED} (already done)"
echo "Total:     $((N_SUBMITTED + N_SKIPPED)) points"
echo "========================================"
echo ""
echo "Monitor with:  squeue -u kayal"
echo "Check status:  bash check_pes.sh"
