import os
import sys
from typing import List, Union

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from entity.single_cluster import SingleCluster
from entity.multi_cluster import MultiCluster
from utils import inner_product, util


class HierarchicalTree:
    def __init__(self, clusters: List[Union[SingleCluster, MultiCluster]], cluster_num: int, stop_threshold: int,
                 x_range: List[float], y_range: List[float], max_qubit_num: int, env: str, backend: str,
                 print_detail: bool = False):
        """
        :param clusters: list, the input of QAHCA
        :param cluster_num: int, the number of clusters
        :param stop_threshold: int, the final number of clusters to stop QAHCA
        :param x_range: list, the range of x-axis values of all centroids
        :param y_range: list, the range of y-axis values of all centroids
        :param max_qubit_num: int, the maximum number of available qubits
        :param env: string, the environment type for implementing the circuit
        :param backend: string, the backend name when running the circuit on real quantum devices
        :param print_detail: boolean, whether to print the execution detail
        """
        self.clusters = clusters
        self.cluster_num = cluster_num
        self.norm_cents = []
        self.stop_threshold = stop_threshold
        self.x_range = x_range
        self.y_range = y_range
        self.range = max(self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0])
        self.cost_adj = [[0 for _ in range(self.cluster_num)] for _ in range(self.cluster_num)]
        self.max_qubit_num = max_qubit_num

        self.env = env
        self.backend = backend
        self.print_detail = print_detail

        self.initialization()

    def initialization(self):
        self.normalization()

        self.calculate_cost()

    def normalization(self):
        self.norm_cents = [[] for _ in range(self.cluster_num)]
        for i in range(self.cluster_num):
            self.norm_cents[i] = inner_product.normalization(self.clusters[i].centroid, self.x_range[0],
                                                             self.y_range[0], self.range)

    def calculate_cost(self):
        task_num_per_circuit = self.max_qubit_num // 3

        vec_list_1 = []
        vec_list_2 = []
        jobs = []
        cur_cal_num = task_num_per_circuit
        outputs = []
        for cent_1, i in enumerate(self.norm_cents):
            for cent_2, j in enumerate(self.norm_cents):
                if i >= j:
                    continue

                vec_list_1.append(cent_1)
                vec_list_2.append(cent_2)
                cur_cal_num -= 1

                if cur_cal_num <= 0:
                    cur_cal_num = task_num_per_circuit
                    jobs.append(inner_product.cal_inner_product(vec_list_1, vec_list_2, task_num_per_circuit, self.env,
                                                                self.backend, self.print_detail))

                    vec_list_1.clear()
                    vec_list_2.clear()

                    if len(jobs) >= 3:
                        for job in jobs:
                            outputs.append(inner_product.get_inner_product_result(job, task_num_per_circuit,
                                                                                  self.env))

                        jobs.clear()

        if vec_list_1:
            jobs.append(inner_product.cal_inner_product(vec_list_1, vec_list_2, task_num_per_circuit,
                                                        self.env, self.backend, self.print_detail))
        for job in jobs:
            outputs.append(inner_product.get_inner_product_result(job, task_num_per_circuit, self.env))

        index = 0
        for i in range(self.cluster_num):
            for j in range(i + 1, self.cluster_num):
                self.cost_adj[i][j] = outputs[index]
                index += 1

    def find_maximum(self):
        max_i, max_j, max_val = 0, 0, 0
        for i in range(self.cluster_num):
            for j in range(i + 1, self.cluster_num):
                if self.cost_adj[i][j] > max_val:
                    max_val = self.cost_adj[i][j]
                    max_i = i
                    max_j = j

        return max_i, max_j

    def update_info(self, max_i, max_j):
        # updating the size of cost matrix
        new_cost_adj = [[0 for _ in range(self.cluster_num - 1)] for _ in range(self.cluster_num - 1)]
        for i in range(max_j):
            for j in range(i + 1, self.cluster_num):
                if j == max_j:
                    continue
                elif j < max_j:
                    new_cost_adj[i][j] = self.cost_adj[i][j]
                else:
                    new_cost_adj[i][j - 1] = self.cost_adj[i][j]

        for i in range(max_j + 1, self.cluster_num):
            for j in range(i + 1, self.cluster_num):
                new_cost_adj[i - 1][j - 1] = self.cost_adj[i][j]

        self.cost_adj = new_cost_adj
        self.cluster_num -= 1

        # updating the cost between max_i-th cluster and other clusters
        self.norm_cents[max_i] = inner_product.normalization(self.clusters[max_i].centroid, self.x_range[0],
                                                             self.y_range[0], self.range)
        self.norm_cents.pop(max_j)

        vec_list_1 = []
        vec_list_2 = []
        task_num_per_circuit = self.max_qubit_num // 3
        cur_cal_num = task_num_per_circuit
        jobs = []
        outputs = []
        for i in range(self.cluster_num):
            if i == max_i:
                continue

            vec_list_1.append(self.clusters[max_i])
            vec_list_2.append(self.clusters[i])
            cur_cal_num -= 1

            if cur_cal_num <= 0:
                cur_cal_num = task_num_per_circuit
                jobs.append(inner_product.cal_inner_product(vec_list_1, vec_list_2, task_num_per_circuit, self.env,
                                                            self.backend, self.print_detail))
                vec_list_1.clear()
                vec_list_2.clear()

                if len(jobs) >= 3:
                    for job in jobs:
                        outputs.append(inner_product.get_inner_product_result(job, task_num_per_circuit, self.env))

                    jobs.clear()

        if vec_list_1:
            jobs.append(inner_product.cal_inner_product(vec_list_1, vec_list_2, task_num_per_circuit, self.env,
                                                        self.backend, self.print_detail))
        for job in jobs:
            outputs.append(inner_product.get_inner_product_result(job, task_num_per_circuit, self.env))

        index = 0
        for i in range(max_i):
            self.cost_adj[i][max_i] = outputs[index]
            index += 1
        for i in range(max_i + 1, self.cluster_num):
            self.cost_adj[max_i][i] = outputs[index]
            index += 1

    def build_tree(self):
        while self.cluster_num > self.stop_threshold:
            max_i, max_j = self.find_maximum()

            self.clusters[max_i] = MultiCluster(None, [self.clusters[max_i], self.clusters[max_j]], self.env,
                                                self.backend, self.print_detail)
            self.clusters[max_i].cal_centroid()
            self.clusters.pop(max_j)

            self.update_info(max_i, max_j)

    def decompose_tree(self):
        is_terminal = False
        while not is_terminal:
            is_terminal = True
            for i in range(len(self.clusters) - 1, -1, -1):
                if self.clusters[i].class_type == 'Multi':
                    is_terminal = False
                    # decomposing this MultiCluster
                    tmp_multi_cluster = MultiCluster(None, [self.clusters[i - 1], *self.clusters[i].elements,
                                                            self.clusters[(i + 1) % len(self.clusters)]], self.env,
                                                     self.backend, self.print_detail)
                    tmp_multi_cluster.find_optimal_path(self.max_qubit_num)
                    self.clusters[i: i + 1] = tmp_multi_cluster.elements[1: -1]

    def classical_build_tree(self):
        # merging nearest neighbor clusters
        while len(self.clusters) > self.stop_threshold:
            neighbors = [0 for _ in range(len(self.clusters))]
            for i in range(len(self.clusters)):
                min_len = float('inf')
                min_neighbor = -1
                for j in range(len(self.clusters)):
                    if i == j:
                        continue
                    cur_len = util.cal_similarity(self.clusters[i].centroid, self.clusters[j].centroid)
                    if cur_len < min_len:
                        min_len = cur_len
                        min_neighbor = j
                neighbors[i] = min_neighbor
            # 判断是否有可以合并的聚类
            delete_index = []
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    if neighbors[i] == j and neighbors[j] == i:
                        self.clusters[i] = MultiCluster(None, [self.clusters[i], self.clusters[j]], self.env,
                                                        self.backend, self.print_detail)
                        self.clusters[i].cal_centroid()
                        delete_index.append(j)
            delete_index.sort(reverse=True)
            for i in range(len(delete_index)):
                self.clusters.pop(delete_index[i])
