# -*- coding: UTF-8 -*-
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Sampler, Session
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import Fake27QPulseV1, Fake127QPulseV1, GenericBackendV2


def exec_qcircuit(qc, shots, env, noisy, backend, print_detail=True):
    if print_detail:
        print("The circuit depth before transpile", qc.depth())

    if env == 'sim':
        if noisy:
            device_backend = GenericBackendV2(qc.num_qubits)
            simulator = AerSimulator.from_backend(device_backend)
        else:
            simulator = AerSimulator()
            # device_backend = GenericBackendV2(qc.num_qubits)
            # simulator = AerSimulator.from_backend(device_backend)
        trans_qc = transpile(qc, simulator)
        if print_detail:
            print("The circuit depth after transpile", trans_qc.depth())
        job = simulator.run(trans_qc, shots=shots)
    else:
        # real quantum computer
        service = QiskitRuntimeService()
        device_backend = service.backend(backend)
        sampler = Sampler(backend=device_backend)
        trans_qc = transpile(qc, device_backend)
        if print_detail:
            print("The circuit depth after transpile", trans_qc.depth())
        job = sampler.run(circuits=trans_qc, shots=shots)
    return job


def get_output(job, env):
    if env == 'sim':
        output = job.result().get_counts()
    else:
        output = job.result().quasi_dists[0]
    return output
