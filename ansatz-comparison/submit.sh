# Author: Riya Kayal
# Created: 30/11/2025

#!/bin/bash

# USAGE: ./submit.sh ANSATZ_NAME
# e.g. ./submit.sh PUCCD

# Check if a label was provided; if not, default to 'succd'
LABEL=${1:-"UCCSD"}
# Convert to lowercase just to be safe
label_=$(echo "$LABEL" | tr '[:upper:]' '[:lower:]')

#SUBMIT_SCRIPT="job_${LABEL}.sh"
SUBMIT_SCRIPT="current"

cat <<EOF > "$SUBMIT_SCRIPT"
#!/bin/bash
#SBATCH --job-name=vqe_${LABEL}
#SBATCH --partition=ep30th
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=${label_}_%j.out
#SBATCH --error=%j.err

mkdir -p ansatz_results
export OMP_NUM_THREADS=1
export GOMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

# Get the absolute path of where YOU ARE RIGHT NOW
CURRENT_DIR=$(pwd)
# Use the auto-detected directory instead of a hardcoded path
cd "$CURRENT_DIR"

# Pass the label as an argument to the python script here
/hsnfs/users/kayal/venv_vqe/bin/python3.9 vqe_runner.py "$LABEL"
EOF

sbatch "$SUBMIT_SCRIPT"
echo "Submitted job for ansatz: $LABEL"
