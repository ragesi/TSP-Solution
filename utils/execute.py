# -*- coding: UTF-8 -*-
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Sampler, Session


def sampler_run(backend, resilience_level, optimization_level, qc, shots):
    service = QiskitRuntimeService()
    backend = backend

    options = Options()
    options.resilience_level = resilience_level
    options.optimization_level = optimization_level

    sampler = Sampler(backend, options=options)
    job = sampler.run(circuits=qc, shots=shots)

    return job.result().quasi_dists[0]
