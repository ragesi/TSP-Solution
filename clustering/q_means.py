# -*- coding: UTF-8 -*-
import argparse
import sys
import os
import random
import math as m
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from utils import inner_product
from entity.single_cluster import SingleCluster
from utils.read_dataset import read_dataset
import estimation_util


class Job:
    def __init__(self, job, range_2):
        self.job = job
        self.range_2 = range_2


class QMeans:
    def __init__(self, points, cluster_num, env, backend, max_qubit_num, print_detail=False):
        """
        :param points: list, coordinates of all cities
        :param cluster_num: int, the number of clusters that need to divide
        :param env: string, the environment type for implementing the circuit
        :param backend: string, the backend name when running the circuit on real quantum devices
        :param max_qubit_num: int, maximum number of available qubits
        :param print_detail: boolean, whether to print the execution detail
        """
        self.points = points
        self.cluster_num = cluster_num
        self.iter_num = 15
        self.centroids = []
        self.clusters = [[] for _ in np.arange(self.cluster_num)]
        self.range = None
        self.x_range = None
        self.y_range = None
        self.print_detail = print_detail

        self.env = env
        self.backend = backend
        # if self.env == 'sim':
        #     self.backend = AerSimulator()
        # else:
        #     self.backend = backend
        self.max_qubit_num = max_qubit_num

        self.initialize_all()

    def initialize_all(self):
        """
        executing all initialization steps
        """
        self.init_range()
        self.init_centroid()
        self.init_clusters()
        if self.print_detail:
            print("---------------initialization: ----------------------")
            print("centroids: ", self.centroids)
            print("clusters: ", self.clusters)

    def init_centroid(self):
        """
        initial the centroids for each cluster
        """
        centroid_set = set()
        point_num = len(self.points)
        while len(centroid_set) < self.cluster_num:
            tmp_index = random.randint(0, point_num - 1)
            if tmp_index not in centroid_set:
                centroid_set.add(tmp_index)
                self.centroids.append(self.points[tmp_index])

    def init_clusters(self):
        # the classical method
        for point in self.points:
            self.clusters[self.classical_find_optimal_cluster(point)].append(point)

        # the quantum method
        # tmp_points = [point for point in self.points]
        # new_cluster_id_list = self.find_optimal_cluster(tmp_points)
        #
        # for id, i in enumerate(new_cluster_id_list):
        #     self.clusters[id].append(self.points[i])

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

    def to_bloch_state(self, point) -> tuple[float, float]:
        """
        transforming bases, which is converting Cartesian coordinates to Bloch coordinates
        :param point: the point waiting to be transformed
        """
        delta_x = (point[0] - self.x_range[0]) / (1.0 * self.range)
        delta_y = (point[1] - self.y_range[0]) / (1.0 * self.range)
        theta = np.pi / 2 * (delta_x + delta_y)
        phi = np.pi / 2 * (delta_x - delta_y + 1)

        return theta, phi

    # def normalization(self, point) -> list:
    #     """
    #     executing the normalization of points
    #     :param point: list
    #     """
    #     return [(point[0] - self.x_range[0]) / self.range, (point[1] - self.y_range[0]) / self.range]

    def classical_find_optimal_cluster(self, point) -> int:
        """
        calculating the optimal cluster which the point belongs to
        :param point: list
        """
        min_dist = 1000
        min_id = -1
        for i in range(len(self.centroids)):
            cur_dist = abs(point[0] - self.centroids[i][0]) + abs(point[1] - self.centroids[i][1])
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_id = i
        return min_id

    def find_optimal_cluster(self, points) -> list:
        """
        using quantum method to find the optimal cluster which points belong to
        :param points: list
        """
        task_num_per_circuit = self.max_qubit_num // 3

        norm_cents = [inner_product.normalization(centroid, self.x_range[0], self.y_range[0], self.range) for centroid
                      in self.centroids]
        jobs = list()
        output = list()
        vec_list_1 = list()
        vec_list_2 = list()
        cur_cal_num = task_num_per_circuit

        for point in points:
            norm_point = inner_product.normalization(point, self.x_range[0], self.y_range[0], self.range)
            for i in range(self.cluster_num):
                vec_list_1.append(norm_point)
                vec_list_2.append(norm_cents[i])
                cur_cal_num -= 1

                if cur_cal_num <= 0:
                    cur_cal_num = task_num_per_circuit
                    jobs.append(inner_product.cal_inner_product(vec_list_1, vec_list_2, task_num_per_circuit, self.env,
                                                                self.backend, self.print_detail))

                    vec_list_1.clear()
                    vec_list_2.clear()

                    if len(jobs) >= 3:
                        for job in jobs:
                            output += inner_product.get_inner_product_result(job, task_num_per_circuit, self.env)

                        jobs.clear()

        if vec_list_1:
            jobs.append(inner_product.cal_inner_product(vec_list_1, vec_list_2, task_num_per_circuit, self.env,
                                                        self.backend, self.print_detail))
        for job in jobs:
            output += inner_product.get_inner_product_result(job, task_num_per_circuit, self.env)

        if self.print_detail:
            print("output: ", output)
        new_cluster_id_list = list()
        for i in range(len(points)):
            new_cluster_id_list.append(output[i * self.cluster_num: (i + 1) * self.cluster_num].index(
                max(output[i * self.cluster_num: (i + 1) * self.cluster_num])))
        if self.print_detail:
            print("new_cluster_id_list: ", new_cluster_id_list)

        return new_cluster_id_list

    def update_clusters(self) -> bool:
        """
        updating the clusters based on new centroids
        :return: if there is no change between new and original clusters, return True
        """
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
        """
        updating the sites of centroids
        """
        for i in np.arange(self.cluster_num):
            tmp_x = 0
            tmp_y = 0
            for point in self.clusters[i]:
                tmp_x += point[0]
                tmp_y += point[1]
            self.centroids[i] = [1.0 * tmp_x / len(self.clusters[i]), 1.0 * tmp_y / len(self.clusters[i])]
            # try:
            #     self.centroids[i] = [1.0 * tmp_x / len(self.clusters[i]), 1.0 * tmp_y / len(self.clusters[i])]
            # except ZeroDivisionError:
            #     print("You are not lucky. Please try again.")
            #     sys.exit()

    def q_means(self) -> list:
        """
        the controller that handles the entire process of QMeans
        :return: the final result of QMeans
        """
        for i in range(self.iter_num):
            if self.print_detail:
                print('-------------------', i, '-th Q-means: ------------------------')
            self.update_centroids()
            if self.update_clusters():
                if self.print_detail:
                    print("centroids: ", self.centroids)
                    print("clusters: ", self.clusters)
                break
            if self.print_detail:
                print("centroids: ", self.centroids)
                print("clusters: ", self.clusters)

        return [SingleCluster(self.centroids[i], self.clusters[i], self.env, self.backend, self.print_detail) for i in
                range(self.cluster_num)]


def divide_clusters(points, cluster_max_size, env, backend, max_qubit_num, print_detail=False) -> list:
    """
    dividing points into clusters and ensuring that the size of each cluster is not more than cluster_max_size
    :param points: list, all cities waiting to be clustered
    :param cluster_max_size: int, max size of each cluster
    :param env: string, the environment type for implementing the circuit
    :param backend: string, the backend name when running the circuit on real quantum devices
    :param max_qubit_num: int, maximum number of available qubits
    :param print_detail: boolean, whether to print the execution detail
    :return: the final result of graph partition module
    """
    clusters = QMeans(points, m.ceil(len(points) / cluster_max_size), env, backend, max_qubit_num,
                      print_detail).q_means()
    i = 0
    while i < len(clusters):
        if clusters[i].element_num <= cluster_max_size:
            i += 1
        else:
            if print_detail:
                print("The following cluster need to be split again: ", clusters[i].elements)
            clusters[i: i + 1] = QMeans(clusters[i].elements, m.ceil(clusters[i].element_num / cluster_max_size), env,
                                        backend, max_qubit_num, print_detail).q_means()

    if print_detail:
        print("The final result of Q-means: ")
        for i in range(len(clusters)):
            print(i, '-th final cluster: ', "centroid: ", clusters[i].centroid, " cluster: ", clusters[i].elements)
        print("The number of clusters: ", len(clusters))

    # recalculating the centroids of clusters
    for cluster in clusters:
        cluster.calculate_centroid()

    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-means')
    parser.add_argument('--file_name', '-f', type=str, default='ulysses16.tsp', help='Dataset of TSP')
    parser.add_argument('--scale', '-s', type=int, default=16, help='The scale of dataset')
    parser.add_argument('--cluster_max_size', '-cms', type=int, default=6, help='The max size of clusters')
    parser.add_argument('--env', '-e', type=str, default='sim',
                        help='The environment to run program, parameter: "sim"; "remote_sim"; "real"')
    parser.add_argument('--backend', '-b', type=str, default=None, help='The backend to run program')
    parser.add_argument('--max_qubit_num', '-m', type=int, default=15,
                        help='The maximum number of qubits in the backend')
    parser.add_argument('--print_detail', '-p', type=bool, default=False)

    args = parser.parse_args()

    lines = read_dataset(args.file_name, args.scale)
    points = []
    for line in lines:
        point = line.strip().split(' ')[1:]
        points.append([float(point[i]) for i in np.arange(len(point))])

    clusters = divide_clusters(points, args.cluster_max_size, args.env, args.backend, args.max_qubit_num,
                               args.print_detail)

    print("The maximum number of points for all clusters: ", max([cluster.element_num for cluster in clusters]))
    print("The minimum number of points for all clusters: ", min([cluster.element_num for cluster in clusters]))
    weights, cut_weights = estimation_util.estimation_with_weight(clusters)
    print("The weights in all subgraphs for result is: ", weights)
    print("The cut-weights for result is: ", cut_weights)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster.centroid[0], cluster.centroid[1], color=colors[i], s=30, marker='x')
        for point in cluster.elements:
            plt.scatter(point[0], point[1], color=colors[i], s=5)
    plt.show()
