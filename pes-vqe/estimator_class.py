

# Author: Riya Kayal
# Created: 05/02/2026


#  Estimator
# ── Reproducible estimator ────────────────────────────────────────────────────
print("Building Estimator...", flush=True)

from qiskit.primitives.base import BaseEstimatorV1
from qiskit.primitives import EstimatorResult, PrimitiveJob
from qiskit.quantum_info import Statevector
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
import numpy as np

# ── Exact V1 Estimator — no transpilation, no shots, no randomness ────────────
class ExactEstimatorV1(BaseEstimatorV1):
    """
    Minimal exact statevector estimator.
    Bypasses all transpilation — computes <psi|H|psi> directly via numpy.
    Compatible with qiskit_algorithms 0.3.1 VQE (V1 API).
    """
    def _run(self, circuits, observables, parameter_values, **run_options):
        def compute():
            evs = []
            for circ, obs, params in zip(circuits, observables, parameter_values):
                # Bind parameters directly — no transpilation
                bound = circ.assign_parameters(
                    dict(zip(circ.parameters, params))
                )
                sv  = Statevector(bound)
                ev  = sv.expectation_value(obs).real
                evs.append(ev)
            return EstimatorResult(np.array(evs), [{}] * len(evs))

        job = PrimitiveJob(compute)
        job._submit()
        return job  

def get_estimator():    
    estimator = ExactEstimatorV1()
    return estimator
    print("Estimator done.", flush=True)
