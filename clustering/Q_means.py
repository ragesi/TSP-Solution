# -*- coding: UTF-8 -*-
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from utils import display_result as disp, util
from cluster import SingleCluster
import random


class QMeans:
    def __init__(self, points, cluster_num):
        """
        :param points: node list
        :param cluster_num: the number of clusters that need to divide
        """
        self.points = points
        self.cluster_num = cluster_num
        self.iter_num = 15
        self.centroids = []
        self.clusters = [[] for _ in np.arange(self.cluster_num)]
        self.range = None
        self.x_range = None
        self.y_range = None
        # self.boundary = set()

        self.init_range()
        self.init_centroid()
        self.init_clusters()

    def init_centroid(self):
        """
        initial the centroids for every cluster
        """
        centroid_set = set()
        point_num = len(self.points)
        while len(centroid_set) < self.cluster_num:
            tmp_index = random.randint(0, point_num - 1)
            if tmp_index not in centroid_set:
                centroid_set.add(tmp_index)
                self.centroids.append(self.points[tmp_index])

    def init_clusters(self):
        for point in self.points:
            # self.clusters[self.find_optimal_cluster(point)].append(point)
            self.clusters[self.classical_find_optimal_cluster(point)].append(point)

    def init_range(self):
        tmp_x = sorted([point[0] for point in self.points])
        tmp_y = sorted([point[1] for point in self.points])
        self.range = max(tmp_x[-1] - tmp_x[0], tmp_y[-1] - tmp_y[0])
        self.x_range = [tmp_x[0], tmp_x[-1]]
        self.y_range = [tmp_y[0], tmp_y[-1]]

    def to_bloch_state(self, point):
        """
        transforming bases, which is converting Cartesian coordinates to Bloch coordinates
        :param point: the point waiting to be transformed
        :return: QuantumCircuit
        """
        delta_x = (point[0] - self.x_range[0]) / (1.0 * self.range)
        delta_y = (point[1] - self.y_range[0]) / (1.0 * self.range)
        theta = np.pi / 2 * (delta_x + delta_y)
        phi = np.pi / 2 * (delta_x - delta_y + 1)

        return theta, phi

    def classical_find_optimal_cluster(self, point):
        min_dist = 1000
        min_id = -1
        for i in range(len(self.centroids)):
            cur_dist = abs(point[0] - self.centroids[i][0]) + abs(point[1] - self.centroids[i][1])
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_id = i
        return min_id

    def find_optimal_cluster(self, point):
        q = QuantumRegister(self.cluster_num * 3)
        cl = ClassicalRegister(self.cluster_num)
        qc = QuantumCircuit(q, cl)

        point_theta, point_phi = self.to_bloch_state(point)
        for i in range(self.cluster_num):
            qc.u(point_theta, point_phi, 0, q[i * 3 + 1])
            cent_theta, cent_phi = self.to_bloch_state(self.centroids[i])
            qc.u(cent_theta, cent_phi, 0, q[i * 3 + 2])

            qc.h(q[i * 3])
            qc.cswap(q[i * 3], q[i * 3 + 1], q[i * 3 + 2])
            qc.h(q[i * 3])

        for i in np.arange(self.cluster_num):
            qc.measure(q[i * 3], cl[i])

        output = disp.Measurement(qc, return_M=True, print_M=False, shots=1000)
        dists = [0 for _ in np.arange(self.cluster_num)]
        for item in output.items():
            for i in np.arange(self.cluster_num):
                dists[i] += item[1] if item[0][i] == '0' else 0
        return dists.index(max(dists))
        # dists.sort(reverse=True)
        # if dists[0] - dists[1] < 10:
        #     self.boundary.add(tuple(point))
        # return res_index


    def update_clusters(self):
        new_clusters = [[] for _ in np.arange(self.cluster_num)]
        is_terminate = True
        for i in np.arange(self.cluster_num):
            for point in self.clusters[i]:
                new_cluster_id = self.classical_find_optimal_cluster(point)
                # new_cluster_id = self.find_optimal_cluster(point)
                new_clusters[new_cluster_id].append(point)
                if new_cluster_id != i:
                    is_terminate = False

        self.clusters = new_clusters
        return is_terminate

    def update_centroids(self):
        for i in np.arange(self.cluster_num):
            tmp_x = 0
            tmp_y = 0
            for point in self.clusters[i]:
                tmp_x += point[0]
                tmp_y += point[1]
            self.centroids[i] = [1.0 * tmp_x / len(self.clusters[i]), 1.0 * tmp_y / len(self.clusters[i])]

    def main(self):
        for _ in np.arange(self.iter_num):
            self.update_centroids()
            # self.boundary = set()
            if self.update_clusters():
                break
            # print(_ + 1)
            # colors = plt.cm.rainbow(np.linspace(0, 1, 22))
            # for i in np.arange(22):
            #     plt.scatter(self.centroids[i][0], self.centroids[i][1], color=colors[i], s=30, marker='x')
            #     for point in self.clusters[i]:
            #         plt.scatter(point[0], point[1], color=colors[i], s=5)
            # plt.show()

        return [SingleCluster(self.centroids[i], self.clusters[i]) for i in range(self.cluster_num)]


if __name__ == '__main__':
    with open('../dataset/xqf131.tsp', 'r') as file:
        lines = file.readlines()

    lines = lines[8: -1]
    points = []
    for line in lines:
        point = line.strip().split(' ')[1:]
        points.append([float(point[i]) for i in np.arange(len(point))])

    test = QMeans(points, 22)

    print('0')
    colors = plt.cm.rainbow(np.linspace(0, 1, 22))
    for i in np.arange(22):
        plt.scatter(test.centroids[i][0], test.centroids[i][1], color=colors[i], s=30, marker='x')
        for point in test.clusters[i]:
            plt.scatter(point[0], point[1], color=colors[i], s=5)
    plt.show()

    test.main()

    colors = plt.cm.rainbow(np.linspace(0, 1, 22))
    for i in np.arange(22):
        plt.scatter(test.centroids[i][0], test.centroids[i][1], color=colors[i], s=30, marker='x')
        for point in test.clusters[i]:
            plt.scatter(point[0], point[1], color=colors[i], s=5)
    plt.show()
