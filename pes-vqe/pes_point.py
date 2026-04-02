
# Author: Riya Kayal
# Created: 30/11/2025


"""
pes_point.py
============
Single PES point — called by run_pes.sh for each array task.
Reads R, MOL_NAME, ANSATZ, BASIS from environment variables.
"""

import os
import sys

# Reproducibility — must be first
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GOMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, os.getcwd())

from pes_runner import run_pes_point

# Read from environment
R          = float(os.environ["PES_R"])
MOL_NAME   = os.environ["MOL_NAME"]
ANSATZ     = os.environ["ANSATZ"]
BASIS      = os.environ.get("BASIS", "sto-3g")
OUTPUT_DIR = os.environ.get("PES_OUTPUT_DIR", "pes_results")

# SPSA settings
SPSA_MAXITER = int(os.environ.get("SPSA_MAXITER", "300"))
SPSA_LR      = float(os.environ.get("SPSA_LR",    "0.005"))
SPSA_PERT    = float(os.environ.get("SPSA_PERT",  "0.005"))

# L-BFGS-B for small ansatze (PUCCD, SUCCD)
USE_LBFGS    = os.environ.get("USE_LBFGS", "1") == "1"

run_pes_point(
    mol_name      = MOL_NAME,
    R             = R,
    ansatz_label  = ANSATZ,
    basis         = BASIS,
    output_dir    = OUTPUT_DIR,
    seed_simulator= 42,
    spsa_maxiter  = SPSA_MAXITER,
    spsa_lr       = SPSA_LR,
    spsa_pert     = SPSA_PERT,
    use_lbfgs     = USE_LBFGS,
    lbfgs_maxiter = 200,
)
