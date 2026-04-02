#!/bin/bash
# =============================================================================
# check_pes.sh — check which PES points are done, running, or missing
# =============================================================================


# Author: Riya Kayal
# Created: 30/03/2026


MOL_NAME="LiH"
ANSATZE=("UCCSD" "SUCCD" "PUCCD")
R_VALUES=(1.2 1.25 1.3 1.35 1.4 0.145 1.5 0.155 1.596 1.65 1.7 1.75 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.8 3.0)
WORKDIR=$(pwd)

DONE=0
MISSING=0
RUNNING=0

echo "PES STATUS — ${MOL_NAME}"
echo "========================================"
printf "%-12s %-10s %-10s\n" "Ansatz" "R (Å)" "Status"
echo "----------------------------------------"

for ANSATZ in "${ANSATZE[@]}"; do
    for R in "${R_VALUES[@]}"; do
        R_FMT=$(python3 -c "print(f'{float(\"${R}\"):.4f}')")
        OUTFILE="${WORKDIR}/pes_results/${MOL_NAME}_${ANSATZ}_R${R_FMT}.json"
        JOBNAME="pes_${MOL_NAME}_${ANSATZ}_R${R}"

        if [ -f "${OUTFILE}" ]; then
            printf "%-12s %-10s %-10s\n" "${ANSATZ}" "${R}" "DONE"
            DONE=$((DONE + 1))
        elif squeue -u kayal -h -o "%j" 2>/dev/null | grep -q "${JOBNAME}"; then
            printf "%-12s %-10s %-10s\n" "${ANSATZ}" "${R}" "RUNNING"
            RUNNING=$((RUNNING + 1))
        else
            printf "%-12s %-10s %-10s\n" "${ANSATZ}" "${R}" "MISSING"
            MISSING=$((MISSING + 1))
        fi
    done
done

TOTAL=$(( ${#ANSATZE[@]} * ${#R_VALUES[@]} ))
echo "========================================"
echo "Done:    ${DONE} / ${TOTAL}"
echo "Running: ${RUNNING}"
echo "Missing: ${MISSING}"
echo ""
if [ ${MISSING} -gt 0 ]; then
    echo "Resubmit missing jobs with: bash submit_pes.sh"
fi
