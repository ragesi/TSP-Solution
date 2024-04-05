# -*- coding: UTF-8 -*-
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import Fake27QPulseV1, Fake127QPulseV1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from utils import execute, inner_product
from cluster import SingleCluster
from utils.read_dataset import read_dataset
import random
import math as m


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
        min_x, max_x = min([point[0] for point in self.points]), max([point[0] for point in self.points])
        min_y, max_y = min([point[1] for point in self.points]), max([point[1] for point in self.points])
        self.range = max(max_x - min_x, max_y - min_y)
        self.x_range = [min_x, max_x]
        self.y_range = [min_y, max_y]

        # tmp_x = sorted([point[0] for point in self.points])
        # tmp_y = sorted([point[1] for point in self.points])
        # self.range = max(tmp_x[-1] - tmp_x[0], tmp_y[-1] - tmp_y[0])
        # self.x_range = [tmp_x[0], tmp_x[-1]]
        # self.y_range = [tmp_y[0], tmp_y[-1]]

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

    def normalization(self, point):
        return [(point[0] - self.x_range[0]) / self.range, (point[1] - self.y_range[0]) / self.range]

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
        # normalization
        base_vec = self.normalization(point)
        cur_vec_list = list()
        for i in range(self.cluster_num):
            cur_vec_list.append(self.normalization(self.centroids[i]))

        return inner_product.get_max_inner_product(base_vec, cur_vec_list)

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

    def q_means(self):
        for _ in range(self.iter_num):
            self.update_centroids()
            if self.update_clusters():
                break

        return [SingleCluster(self.centroids[i], self.clusters[i]) for i in range(self.cluster_num)]


if __name__ == '__main__':
    lines = read_dataset('ulysses22.tsp', 22)
    points = []
    for line in lines:
        point = line.strip().split(' ')[1:]
        points.append([float(point[i]) for i in np.arange(len(point))])

    cluster_num = m.ceil(len(points) / 6)
    clusters = QMeans(points, cluster_num).q_means()
    i = 0
    while i < len(clusters):
        if clusters[i].element_num <= 6:
            i += 1
        else:
            tmp_clusters = QMeans(clusters[i].elements, m.ceil(clusters[i].element_num / 6)).q_means()
            del clusters[i]
            clusters[i: i] = tmp_clusters

    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster.centroid[0], cluster.centroid[1], color=colors[i], s=30, marker='x')
        for point in cluster.elements:
            plt.scatter(point[0], point[1], color=colors[i], s=5)
    plt.show()
    print("The number of clusters: ", len(clusters))
    print("The maximum number of points for all clusters: ", max([cluster.element_num for cluster in clusters]))
    print("The minimum number of points for all clusters: ", min([cluster.element_num for cluster in clusters]))
    print("The number of clusters which has only one point: ",
          sum(1 for cluster in clusters if cluster.element_num == 1))
