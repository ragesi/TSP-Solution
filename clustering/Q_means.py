# -*- coding: UTF-8 -*-
import argparse
import sys

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import Fake27QPulseV1, Fake127QPulseV1
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

sys.path.append("..")
from utils import execute, inner_product
from cluster import SingleCluster
from utils.read_dataset import read_dataset
import random
import math as m
import copy


class Job:
    def __init__(self, job, range_2):
        self.job = job
        self.range_2 = range_2


class QMeans:
    def __init__(self, points, cluster_num, env, backend, max_qubit_num):
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
        print("---------------initialization: ----------------------")
        print("centroids: ", self.centroids)
        print("clusters: ", self.clusters)

        self.env = env
        if self.env == 'sim':
            self.backend = AerSimulator()
        else:
            self.backend = backend
        self.max_qubit_num = max_qubit_num

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

    def find_optimal_cluster(self, points):
        task_num_per_circuit = self.max_qubit_num // 3

        vec_list_2 = [self.normalization(centroid) for centroid in self.centroids]
        jobs = list()
        output = list()
        vec_list_1 = list()
        start_idx = 0
        cur_cal_num = task_num_per_circuit
        range_2 = list()
        for point in points:
            norm_point = self.normalization(point)
            for i in range(self.cluster_num):
                if cur_cal_num > 0:
                    cur_cal_num -= 1
                else:
                    cur_cal_num = task_num_per_circuit - 1
                    range_2.append([start_idx, i])
                    start_idx = i
                    vec_list_1.append(norm_point)

                    jobs.append(inner_product.cal_inner_product(vec_list_1, vec_list_2, range_2, task_num_per_circuit,
                                                                self.env, self.backend))

                    vec_list_1.clear()
                    range_2.clear()

                    # when the number of jobs that is in queue reaches 3, waiting for these jobs to finish
                    if len(jobs) >= 3:
                        for job in jobs:
                            output += inner_product.get_inner_product_result(job, task_num_per_circuit, self.env)

                        jobs.clear()

            vec_list_1.append(norm_point)
            range_2.append([start_idx, self.cluster_num])
            start_idx = 0

        jobs.append(inner_product.cal_inner_product(vec_list_1, vec_list_2, range_2, task_num_per_circuit, self.env,
                                                    self.backend))
        for job in jobs:
            output += inner_product.get_inner_product_result(job, task_num_per_circuit, self.env)

        print("output: ", output)
        new_cluster_id_list = list()
        for i in range(len(points)):
            new_cluster_id_list.append(output[i * self.cluster_num: (i + 1) * self.cluster_num].index(
                max(output[i * self.cluster_num: (i + 1) * self.cluster_num])))
        print("new_cluster_id_list: ", new_cluster_id_list)

        return new_cluster_id_list

    def update_clusters(self):
        tmp_points = [point for cluster in self.clusters for point in cluster]
        new_cluster_id_list = self.find_optimal_cluster(tmp_points)

        index = 0
        new_clusters = [[] for _ in np.arange(self.cluster_num)]
        is_terminal = True
        for i in range(self.cluster_num):
            for point in self.clusters[i]:
                new_clusters[new_cluster_id_list[index]].append(point)
                if new_cluster_id_list[index] != i:
                    is_terminal = False
                index += 1

        self.clusters = new_clusters
        return is_terminal

    def update_centroids(self):
        for i in np.arange(self.cluster_num):
            tmp_x = 0
            tmp_y = 0
            for point in self.clusters[i]:
                tmp_x += point[0]
                tmp_y += point[1]
            self.centroids[i] = [1.0 * tmp_x / len(self.clusters[i]), 1.0 * tmp_y / len(self.clusters[i])]

    def q_means(self):
        for i in range(self.iter_num):
            print('-------------------', i, '-th Q-means: ------------------------')
            self.update_centroids()
            if self.update_clusters():
                print("centroids: ", self.centroids)
                print("clusters: ", self.clusters)
                break
            print("centroids: ", self.centroids)
            print("clusters: ", self.clusters)

        return [SingleCluster(self.centroids[i], self.clusters[i]) for i in range(self.cluster_num)]


def divide_clusters(points, cluster_num, env, backend, max_qubit_num):
    clusters = QMeans(points, cluster_num, env, backend, max_qubit_num).q_means()
    i = 0
    while i < len(clusters):
        if clusters[i].element_num <= 6:
            i += 1
        else:
            print("The following cluster need to be split again: ", clusters[i].elements)
            clusters[i: i + 1] = QMeans(clusters[i].elements, m.ceil(clusters[i].element_num / 6), args.env,
                                        args.backend, args.max_qubit_num).q_means()

    return clusters


def estimation(clusters):
    weights = 0.
    for cluster in clusters:
        for i in range(cluster.element_num):
            for j in range(i + 1, cluster.element_num):
                weights += np.linalg.norm(np.array(cluster.elements[i]) - np.array(cluster.elements[j]))
    return weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-means')
    parser.add_argument('--file_name', '-f', type=str, default='ulysses16.tsp', help='Dataset of TSP')
    parser.add_argument('--scale', '-s', type=int, default=16, help='The scale of dataset')
    parser.add_argument('--env', '-e', type=str, default='sim',
                        help='The environment to run program, parameter: "sim"; "remote_sim"; "real"')
    parser.add_argument('--backend', '-b', type=str, default=None, help='The backend to run program')
    parser.add_argument('--max_qubit_num', '-m', type=int, default=15,
                        help='The maximum number of qubits in the backend')

    args = parser.parse_args()

    lines = read_dataset(args.file_name, args.scale)
    points = []
    for line in lines:
        point = line.strip().split(' ')[1:]
        points.append([float(point[i]) for i in np.arange(len(point))])

    cluster_num = m.ceil(len(points) / 6)
    clusters = divide_clusters(points, cluster_num, args.env, args.backend, args.max_qubit_num)

    print("The number of clusters: ", len(clusters))
    print("The maximum number of points for all clusters: ", max([cluster.element_num for cluster in clusters]))
    print("The minimum number of points for all clusters: ", min([cluster.element_num for cluster in clusters]))
    print("The final result of Q-means: ")
    for i in range(len(clusters)):
        print(i, '-th final cluster: ', "centroid: ", clusters[i].centroid, " cluster: ", clusters[i].elements)
    print("The weight for result is: ", estimation(clusters))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster.centroid[0], cluster.centroid[1], color=colors[i], s=30, marker='x')
        for point in cluster.elements:
            plt.scatter(point[0], point[1], color=colors[i], s=5)
    plt.show()
