# -*- coding: UTF-8 -*-
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute
import numpy as np
import math as m
import scipy as sci
import random
import time

from clustering import clustering_unitary_func as uf
from utils import display_result


class QMeans:
    def __init__(self, points, cluster_num, x_range, y_range):
        """
        :param points: node list
        :param cluster_num: the number of clusters that need to divide
        :param x_range: the x_bound for the point list, which is [x_low, x_high]
        :param y_range: the y_bound for the point list, which is [y_low, y_high]
        """
        self.points = points
        self.cluster_num = cluster_num
        self.x_range = x_range
        self.y_range = y_range
        self.centroid = []

        self.init_centroid()
        print(self.centroid)

    def init_centroid(self):
        """
        initial the centroids for every cluster
        """
        # TODO: 此处可以使用论文中的方法去设置中心点
        if self.cluster_num > 6:
            print("too many clusters!")
            exit()
        delta_x = 1.0 * (self.x_range[1] - self.x_range[0]) / min(self.cluster_num + 1, 4)
        delta_y = 1.0 * (self.y_range[1] - self.y_range[0]) / min(self.cluster_num + 1, 4)
        proportion_x = []
        proportion_y = []
        if self.cluster_num < 4:
            # linear distribution of center points
            proportion_x = list(range(1, self.cluster_num + 1))
            proportion_y = list(range(1, self.cluster_num + 1))
        else:
            # the four corners firstly
            proportion_x = [1, 1, 3, 3]
            proportion_y = [1, 3, 1, 3]
            if self.cluster_num == 5:
                proportion_x.append(2)
                proportion_y.append(2)
            elif self.cluster_num == 6:
                if (self.x_range[1] - self.x_range[0]) <= (self.y_range[1] - self.y_range[0]):
                    proportion_x.extend([1, 3])
                    proportion_y.extend([2, 2])
                else:
                    proportion_x.extend([2, 2])
                    proportion_y.extend([1, 3])

        for i in np.arange(len(proportion_x)):
            self.centroid.append(
                [self.x_range[0] + proportion_x[i] * delta_x, self.y_range[0] + proportion_y[i] * delta_y])


if __name__ == '__main__':
    # cluster_num = 6
    # x_range = [0, 8]
    # y_range = [0, 16]
    # test = QMeans([1, 1], cluster_num, x_range, y_range)
    control = QuantumRegister(6)
    target = QuantumRegister(1)
    c = ClassicalRegister(6)
    qc = QuantumCircuit(control, target, c)
    point1 = [0.4, 0.29]
    point2 = [0.17, 0.25]
    qc.append(uf.cal_distance(point1, point2, 6), [*control, *target])
    qc.measure(control, c)
    display_result.Measurement(qc, return_M=False, print_M=True, shots=1000)
