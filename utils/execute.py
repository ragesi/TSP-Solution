# -*- coding: UTF-8 -*-
from qiskit import QuantumCircuit, Aer, execute
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Sampler, Session


def local_simulator(qc, shots):
    backend = Aer.backends(name='qasm_simulator')[0]
    job = execute(qc, backend, shots=shots)
    return job.result().get_counts()


def sampler_run(backend, qc, shots, resilience_level=1, optimization_level=3):
    service = QiskitRuntimeService()
    backend = backend

    options = Options()
    options.resilience_level = resilience_level
    options.optimization_level = optimization_level

    sampler = Sampler(backend, options=options)
    job = sampler.run(circuits=qc, shots=shots)

    return job.result().quasi_dists[0]
