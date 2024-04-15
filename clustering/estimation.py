import numpy as np
import matplotlib.pyplot as plt

from normalized_cut import QAOACut
from utils.read_dataset import read_dataset
from utils import execute
from dataset import test


def get_results(points, theta, lamda, max_sum):
    cut = QAOACut(points, theta, lamda, max_sum)
    qc = cut.qaoa()
    job = execute.exec_qcircuit(qc, 20000, 'sim', False, None, False)
    output = execute.get_output(job, 'sim')
    return output


if __name__ == '__main__':
    points = test.point_test_for_7
    theta = [14.263161169678494, 7.667464665208423, 0.8960059305707773, -0.908906538222956]
    output = get_results(points, theta, 6, 0.05)
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
