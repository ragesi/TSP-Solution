# -*- coding: UTF-8 -*-
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from utils import unitary_function as uf
from utils import display_result as disp, util


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

        x_range, y_range = QMeans.get_range(self.points)
        delta_x = 1.0 * (x_range[1] - x_range[0]) / min(self.cluster_num + 1, 4)
        delta_y = 1.0 * (y_range[1] - y_range[0]) / min(self.cluster_num + 1, 4)
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
                if (x_range[1] - x_range[0]) <= (y_range[1] - y_range[0]):
                    proportion_x.extend([1, 3])
                    proportion_y.extend([2, 2])
                else:
                    proportion_x.extend([2, 2])
                    proportion_y.extend([1, 3])

        for i in np.arange(len(proportion_x)):
            self.centroids.append(
                [x_range[0] + proportion_x[i] * delta_x, y_range[0] + proportion_y[i] * delta_y])

    def init_clusters(self):
        for point in self.points:
            self.clusters[QMeans.find_optimal_cluster(point, self.centroids)].append(point)

    @staticmethod
    def get_range(points):
        tmp_x = sorted([point[0] for point in points])
        tmp_y = sorted([point[1] for point in points])
        return [tmp_x[0], tmp_x[-1]], [tmp_y[0], tmp_y[-1]]

    @staticmethod
    def to_bloch_state(point, x_range, y_range):
        """
        transforming bases, which is converting Cartesian coordinates to Bloch coordinates
        :param point: the point waiting to be transformed
        :param x_range: the x bound of Cartesian coordinate
        :param y_range: the y bound of Cartesian coordinate
        :return: QuantumCircuit
        """
        q = QuantumRegister(1)
        qc = QuantumCircuit(q)

        delta_x = (point[0] - x_range[0]) / (1.0 * x_range[1] - x_range[0])
        delta_y = (point[1] - y_range[0]) / (1.0 * y_range[1] - y_range[0])
        theta = np.pi / 2 * (delta_x + delta_y)
        phi = np.pi / 2 * (delta_x - delta_y + 1)

        qc.u(theta, phi, 0, q[0])
        return qc

    @staticmethod
    def find_optimal_cluster(source, targets):
        dist_num = len(targets)
        # control = QuantumRegister(dist_num)
        # data = QuantumRegister(dist_num + 1)
        # cl = ClassicalRegister(dist_num)
        # qc = QuantumCircuit(control, data, cl)
        q = QuantumRegister(dist_num * 3)
        cl = ClassicalRegister(dist_num)
        qc = QuantumCircuit(q, cl)

        # find the x and y's bound
        x_range, y_range = QMeans.get_range([source, *targets])

        for i in np.arange(dist_num):
            qc.append(QMeans.to_bloch_state(source, x_range, y_range), [q[i * 3 + 1]])
            qc.append(QMeans.to_bloch_state(targets[i], x_range, y_range), [q[i * 3 + 2]])

            qc.append(uf.inner_product(), q[i * 3: (i + 1) * 3])

        for i in np.arange(dist_num):
            qc.measure(q[i * 3], cl[i])

        # transform to bloch coordinate
        # qc.append(QMeans.to_bloch_state(source, x_range, y_range), [data[0]])
        # for i in np.arange(dist_num):
        #     qc.append(QMeans.to_bloch_state(targets[i], x_range, y_range), [data[i + 1]])
        #
        # # calculate similarity through inner product
        # for i in np.arange(dist_num):
        #     qc.append(uf.inner_product(), [control[i], data[i], data[i + 1]])
        # qc.measure(control, cl)
        output = disp.Measurement(qc, return_M=True, print_M=False, shots=100)
        dists = [0 for _ in np.arange(dist_num)]
        for item in output.items():
            for i in np.arange(dist_num):
                dists[i] += item[1] if item[0][i] == '0' else 0
        return dists.index(max(dists))

    def update_clusters(self):
        new_clusters = [[] for _ in np.arange(self.cluster_num)]
        is_terminate = True
        for i in np.arange(self.cluster_num):
            for point in self.clusters[i]:
                new_cluster_id = QMeans.find_optimal_cluster(point, self.centroids)
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

        return self.centroids, self.clusters


if __name__ == '__main__':
    with open('../dataset/xqf131.tsp', 'r') as file:
        lines = file.readlines()

    lines = lines[8: -1]
    points = []
    for line in lines:
        point = line.strip().split(' ')[1:]
        points.append([float(point[i]) for i in np.arange(len(point))])

    test = QMeans(points, 5)

    color = ['red', 'green', 'orange', 'blue', 'black']
    for i in np.arange(5):
        plt.scatter(test.centroids[i][0], test.centroids[i][1], color=color[i], s=30, marker='x')
        for point in test.clusters[i]:
            plt.scatter(point[0], point[1], color=color[i], s=5)
    x_major_locator = MultipleLocator(10)
    y_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    plt.show()
