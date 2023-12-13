# -*- coding: UTF-8 -*-
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from utils import display_result as disp
from cluster import Clusters


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
        self.x_range = None
        self.y_range = None

        self.init_range()
        self.init_centroid()
        self.init_clusters()

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
            self.centroids.append(
                [self.x_range[0] + proportion_x[i] * delta_x, self.y_range[0] + proportion_y[i] * delta_y])

    def init_clusters(self):
        for point in self.points:
            self.clusters[self.find_optimal_cluster(point)].append(point)

    def init_range(self):
        tmp_x = sorted([point[0] for point in self.points])
        tmp_y = sorted([point[1] for point in self.points])
        self.x_range = [tmp_x[0], tmp_x[-1]]
        self.y_range = [tmp_y[0], tmp_y[-1]]

    def to_bloch_state(self, point):
        """
        transforming bases, which is converting Cartesian coordinates to Bloch coordinates
        :param point: the point waiting to be transformed
        :return: QuantumCircuit
        """
        delta_x = (point[0] - self.x_range[0]) / (1.0 * self.x_range[1] - self.x_range[0])
        delta_y = (point[1] - self.y_range[0]) / (1.0 * self.y_range[1] - self.y_range[0])
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

    def update_clusters(self):
        new_clusters = [[] for _ in np.arange(self.cluster_num)]
        is_terminate = True
        for i in np.arange(self.cluster_num):
            for point in self.clusters[i]:
                # new_cluster_id = self.classical_find_optimal_cluster(point)
                new_cluster_id = self.find_optimal_cluster(point)
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
            if self.update_clusters():
                break
            print(_ + 1)
            color = ['red', 'green', 'orange', 'blue', 'black']
            for i in np.arange(2):
                plt.scatter(self.centroids[i][0], self.centroids[i][1], color=color[i], s=30, marker='x')
                for point in self.clusters[i]:
                    plt.scatter(point[0], point[1], color=color[i], s=5)
            plt.show()

        return Clusters(self.centroids, self.clusters)


if __name__ == '__main__':
    with open('../dataset/xqf7.tsp', 'r') as file:
        lines = file.readlines()

    lines = lines[8: -1]
    points = []
    for line in lines:
        point = line.strip().split(' ')[1:]
        points.append([float(point[i]) for i in np.arange(len(point))])

    test = QMeans(points, 2)
    # test.main()

    print('0')
    color = ['red', 'green', 'orange', 'blue', 'black']
    for i in np.arange(2):
        plt.scatter(test.centroids[i][0], test.centroids[i][1], color=color[i], s=30, marker='x')
        for point in test.clusters[i]:
            plt.scatter(point[0], point[1], color=color[i], s=5)
    plt.show()

    test.main()
