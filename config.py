from qiskit_ibm_runtime import QiskitRuntimeService

ibm_api_token = 'your api token'
# test.delete_account()
QiskitRuntimeService.save_account(
    channel='ibm_quantum',
    token=ibm_api_token,
    instance='ibm-q/open/main',
    overwrite=True)

output = QiskitRuntimeService().active_account()
print(output)
