import argparse
import sys
import os
import numpy as np
from typing import Optional

from sklearn.neighbors import NearestNeighbors

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from entity.single_cluster import SingleCluster
from entity.multi_cluster import MultiCluster
from utils import util
from utils.read_dataset import read_dataset
from SQUARE import square_util
from clustering import q_means, qncut
from QAHCA.qahca_main import HierarchicalTree
from QCHSA.qchsa_main import ConvexHull


class TSPSolution:
    def __init__(self, file_name: str, point_num: int, partition_method: str, cluster_max_size: int, env: str,
                 backend: Optional[str], max_qubit_num: int, print_detail: bool):
        """
        :param file_name: string, the file path of test case
        :param point_num: int, the number of point
        :param partition_method: string, the partition method: QMeans or QNCut
        :param cluster_max_size: int, the maximum number of clusters
        :param env: string, the environment type for implementing the circuit
        :param backend: string, the backend name when running the circuit on real quantum devices
        :param max_qubit_num: int, maximum number of available qubits
        :param print_detail: boolean, whether to print the execution detail
        """
        self.points = []
        self.point_num = point_num
        self.point_map = dict()
        self.path = []

        self.partition_method = partition_method
        self.cluster_max_size = cluster_max_size
        self.max_qubit_num = max_qubit_num

        self.x_bounds = [10000, 0]
        self.y_bounds = [10000, 0]

        self.env = env
        self.backend = backend
        self.print_detail = print_detail

        self.file_name = file_name
        self.get_data()

    def get_data(self):
        """
        reading the input dataset and determine the bounds of coordinates
        """
        lines = read_dataset(self.file_name, self.point_num)
        for line in lines:
            point = line.strip().split(' ')
            point = [int(point[0]), float(point[1]), float(point[2])]
            self.points.append((point[1], point[2]))
            self.point_map[(point[1], point[2])] = point[0]

            if point[1] < self.x_bounds[0]:
                self.x_bounds[0] = point[1]
            if point[1] > self.x_bounds[1]:
                self.x_bounds[1] = point[1]
            if point[2] < self.y_bounds[0]:
                self.y_bounds[0] = point[2]
            if point[2] > self.y_bounds[1]:
                self.y_bounds[1] = point[2]
        print(f"x_bound: {self.x_bounds}, y_bound: {self.y_bounds}")

    def graph_partition_q_means(self):
        """
        using QMeans to graph partition
        """
        split_cluster_num = self.point_num // self.cluster_max_size
        self.path = q_means.divide_clusters(self.points, split_cluster_num, self.env, self.backend, self.max_qubit_num)
        if self.print_detail:
            print(len(self.path))

    def graph_partition_qncut(self):
        """
        using QNCut to graph partition
        """
        self.path = qncut.divide_clusters(self.points, self.env, self.backend, self.print_detail, self.cluster_max_size)
        if self.print_detail:
            print(len(self.path))

    def remove_outliers(self):
        """
        removing the outliers in graph (deprecated)
        """
        outliers = []
        nn = NearestNeighbors(n_neighbors=4)
        nearest_neighbors_matrix = nn.fit(self.points).kneighbors_graph(mode='distance')
        outlier_threshold = nearest_neighbors_matrix.sum() / self.point_num / 4.0 * 2.0
        print(outlier_threshold)
        for i in range(self.point_num - 1, -1, -1):
            cur_dist = nearest_neighbors_matrix[i].sum() / 4.0
            if cur_dist > outlier_threshold:
                outliers.append(self.points.pop(i))
        print(f"len(self.points):{len(self.points)}")
        return outliers

    def add_outliers(self, outliers):
        """
        adding the outliers to final result (deprecated)
        """
        for i in range(len(outliers)):
            min_dist = 10000
            target_index = -1
            for j in range(len(self.path)):
                tmp_dist = (util.cal_similarity(self.path[j - 1], outliers[i]) +
                            util.cal_similarity(outliers[i], self.path[j]) - util.cal_similarity(self.path[j - 1],
                                                                                                 self.path[j]))
                if tmp_dist < min_dist:
                    target_index = j
                    min_dist = tmp_dist
            self.path.insert(target_index, outliers[i])

    def main(self):
        """
        the controller handling the entire process of SQUARE
        """
        # if QUOTA can handle the problem independently
        if len(self.points) < self.cluster_max_size:
            cur_cluster = SingleCluster(None, self.points, self.env, self.backend, self.print_detail)
            cur_cluster.find_optimal_circle(self.max_qubit_num)
            cur_order = cur_cluster.get_nodes_in_path()
            for point in cur_order:
                self.path.append(self.point_map.get(point))
            return

        # graph partition module
        if self.partition_method == 'QMeans':
            self.graph_partition_q_means()
        else:
            self.graph_partition_qncut()

        # subgraph problem planning module
        h_tree = HierarchicalTree(self.path, len(self.path), self.cluster_max_size - 1, self.x_bounds,
                                  self.y_bounds, self.max_qubit_num, self.env, self.backend,
                                  self.print_detail)
        h_tree.build_tree()
        # h_tree.classical_build_tree()

        # finding the optimal Hamiltonian cycle
        self.path = [MultiCluster(None, self.path, self.env, self.backend, self.print_detail)]
        self.path[0].find_optimal_circle(self.max_qubit_num)
        self.path = self.path[0].elements

        h_tree.decompose_tree()

        # setting the start and end points for each underlying cluster and rearrange the vertices order
        for i in range(len(self.path)):
            self.path[i - 1].tail, self.path[i].head = square_util.find_diff_clusters_connector(self.path[i - 1],
                                                                                                self.path[i])
            if i > 0:
                self.path[i - 1].determine_head_and_tail()
        self.path[len(self.path) - 1].determine_head_and_tail()

        # subgraph solving, QUOTA
        for i in range(len(self.path)):
            self.path[i].find_optimal_path(self.max_qubit_num)

        # restoring to a single vertices state
        for i in range(len(self.path) - 1, -1, -1):
            self.path[i: i + 1] = self.path[i].elements

        if self.print_detail:
            square_util.draw_result(self.point_num, self.path)

    def cal_total_cost(self):
        dist = 0.
        for i in range(self.point_num):
            dist += util.cal_similarity(self.path[i - 1], self.path[i])
        return dist

    def get_accuracy(self):
        dist = self.cal_total_cost()
        print("distance: ", dist)

        opt_cost = float(read_dataset('opt_cost', self.point_num)[0])
        print("optimal cost: ", opt_cost)

        print("accuracy: ", dist / opt_cost)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SQUARE')
    parser.add_argument('--file_name', '-f', type=str, default='ulysses16.tsp', help='Dataset of TSP')
    parser.add_argument('--scale', '-s', type=int, default=16, help='The scale of dataset')
    parser.add_argument('--partition_method', '-p', type=str, default='QMeans',
                        help='The partition method used in graph partition module: QMeans or QNCut')
    parser.add_argument('--cluster_max_size', '-c', type=int, default=6, help='The maximum size of all clusters')
    parser.add_argument('--env', '-e', type=str, default='sim',
                        help='The environment to run program, parameter: "sim"; "remote_sim"; "real"')
    parser.add_argument('--backend', '-b', type=str, default=None, help='The backend to run program')
    parser.add_argument('--max_qubit_num', '-m', type=int, default=15,
                        help='The maximum number of qubits in the backend')
    parser.add_argument('--print_detail', '-pd', type=bool, default=False, help='Print detailed information')

    args = parser.parse_args()

    test = TSPSolution(args.file_name, args.scale, args.partition_method, args.cluster_max_size, args.env, args.backend,
                       args.max_qubit_num, args.print_detail)
    test.main()
    print(test.path)
    test.get_accuracy()
