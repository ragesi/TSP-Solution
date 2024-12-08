import numpy as np
import matplotlib.pyplot as plt
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from clustering.qncut import QAOACut
from utils.read_dataset import read_dataset
from utils import execute
from clustering import estimation_util
from dataset import test


def get_results(points, theta, lamda, max_sum):
    cut = QAOACut(points, theta, lamda, max_sum)
    qc = cut.qaoa()
    print("qubits cost: ", qc.num_qubits)
    job = execute.exec_qcircuit(qc, 20000, 'sim', False, None, True)
    output = execute.get_output(job, 'sim')
    return output


if __name__ == '__main__':
    lines = read_dataset("att48.tsp", 48)
    points = []
    for line in lines:
        point = line.strip().split(' ')[1:]
        points.append([float(point[i]) for i in np.arange(len(point))])

    # '1011010001000000', '1110011110110010', '1010010000111001'
    # points = [[39.57, 26.15], [33.48, 10.54], [38.42, 13.11], [37.52, 20.44], [41.23, 9.1], [36.08, -5.21], [37.51, 15.17], [39.36, 19.56]]
    theta = [18.65311940314177, -7.721655528055066, 13.043217198650726, 2.6626378626526988]
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

    bitstring = bitstring_list[0]
    clusters = [[], []]
    for i in range(len(bitstring)):
        if bitstring[i] == '0':
            clusters[0].append(points[i])
        else:
            clusters[1].append(points[i])

    weights, cut_weights = estimation_util.estimation_with_weight(clusters)
    print("The weights in all subgraphs for result is: ", weights)
    print("The cut-weights for result is: ", cut_weights)
