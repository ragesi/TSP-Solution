import numpy as np
import matplotlib.pyplot as plt

from normalized_cut import QAOACut
from utils.read_dataset import read_dataset
from utils import execute
from dataset import test


def estimation(points, theta, lamda, max_sum):
    cut = QAOACut(points, theta, lamda, max_sum)
    qc = cut.qaoa()
    job = execute.exec_qcircuit(qc, 20000, 'sim', False, None, False)
    output = execute.get_output(job, 'sim')
    return output


if __name__ == '__main__':
    points = test.point_test_for_7
    theta = [2.2650002790427624, 16.366021135696226, 1.6702216317942067, 10.15676114582114]
    output = estimation(points, theta, 6, 0.05)
    print(len(output))

    max_num = 0
    for item in output.items():
        if item[1] > max_num:
            max_num = item[1]

    bitstring_list = list()
    for item in output.items():
        if item[1] == max_num:
            bitstring_list.append(item[0])

    print("bitstring_list: ", bitstring_list)
    print("max_num: ", max_num)

    weights = list()
    for bitstring in bitstring_list:
        cur_weight = 0.
        for i in range(len(bitstring)):
            for j in range(i + 1, len(bitstring)):
                if bitstring[i] != bitstring[j]:
                    cur_weight += np.linalg.norm(np.array(points[i]) - np.array(points[j]))
        weights.append(cur_weight)
    print("weights: ", weights)
