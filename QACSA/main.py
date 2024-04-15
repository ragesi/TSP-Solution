import argparse
import sys

import math as m
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull, QhullError

sys.path.append("..")
from clustering import Q_means
from cluster import SingleCluster, MultiCluster, cal_similarity
from utils.read_dataset import read_dataset


def spectral_clustering(points, cluster_num):
    k_neighbors = int(len(points) / cluster_num)
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nearest_neighbors_matrix = nn.fit(points).kneighbors_graph(mode='distance')

    sc = SpectralClustering(n_clusters=cluster_num, affinity='precomputed')
    y_pred = sc.fit_predict(nearest_neighbors_matrix)

    clusters = [[] for _ in range(cluster_num)]
    for i in range(len(y_pred)):
        clusters[y_pred[i]].append(tuple(points[i]))
    return clusters


def cal_dist_point_to_line(point, line_start, line_end):
    vec_of_point = point - line_start
    vec_of_line = line_end - line_start

    dist_vec_of_point = np.linalg.norm(vec_of_point)
    dist_vec_of_line = np.linalg.norm(vec_of_line)

    # 计算点到直线的垂线距离
    return abs(
        dist_vec_of_point * np.sin(np.arccos(np.dot(vec_of_point, vec_of_line) / dist_vec_of_line / dist_vec_of_point)))


class TSPSolution:
    def __init__(self, file_name, point_num, env, backend, max_qubit_num):
        self.points = []
        self.point_num = point_num
        self.point_map = dict()
        self.path = []

        self.sub_issue_max_size = 6
        self.max_qubit_num = max_qubit_num

        self.x_bounds = [10000, 0]
        self.y_bounds = [10000, 0]

        self.env = env
        self.backend = backend

        self.file_name = file_name
        self.get_data()

    def get_data(self):
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

    @staticmethod
    def find_diff_clusters_connector(cluster_1, cluster_2):
        points_1 = cluster_1.get_convex_hull()
        points_2 = cluster_2.get_convex_hull()
        conn_min_dist = 10000
        conn_begin = 0
        conn_end = 0
        for i in range(len(points_1)):
            for j in range(len(points_2)):
                cur_dist = cal_similarity(points_1[i], points_2[j])
                if cur_dist < conn_min_dist:
                    conn_min_dist = cur_dist
                    conn_begin = i
                    conn_end = j
        return points_1[conn_begin], points_2[conn_end]

    def divide_sub_issue(self):
        if len(self.points) < self.sub_issue_max_size:
            cur_cluster = SingleCluster(None, self.points)
            cur_cluster.find_optimal_circle(self.max_qubit_num)
            cur_order = cur_cluster.get_nodes_in_path()
            for point in cur_order:
                self.path.append(self.point_map.get(point))
            return

        # 删除掉离群点
        # outliers = []
        # nn = NearestNeighbors(n_neighbors=4)
        # nearest_neighbors_matrix = nn.fit(self.points).kneighbors_graph(mode='distance')
        # outlier_threshold = nearest_neighbors_matrix.sum() / self.point_num / 4.0 * 2.0
        # print(outlier_threshold)
        # for i in range(self.point_num - 1, -1, -1):
        #     cur_dist = nearest_neighbors_matrix[i].sum() / 4.0
        #     if cur_dist > outlier_threshold:
        #         outliers.append(self.points.pop(i))
        # print(f"len(self.points):{len(self.points)}")

        # 第一次聚类，之后处理的最小粒度从数据点变成聚类中心点
        # 改成谱聚类
        # split_cluster_num = m.ceil(1.0 * len(self.points) / 6)
        # self.path = spectral_clustering(np.array(self.points), split_cluster_num)
        # i = 0
        # while i < len(self.path):
        #     if len(self.path[i]) > 6:
        #         # 需要再次划分
        #         split_cluster_num = m.ceil(1.0 * len(self.path[i]) / 6)
        #         self.path[i: i + 1] = spectral_clustering(np.array(self.path[i]), split_cluster_num)
        #     else:
        #         # 计算聚类中心点
        #         tmp_centroid = [1.0 * sum(point[0] for point in self.path[i]) / len(self.path[i]),
        #                         1.0 * sum(point[1] for point in self.path[i]) / len(self.path[i])]
        #         self.path[i] = SingleCluster(tmp_centroid, self.path[i])
        #         i += 1
        # print(len(self.path))

        # k-means聚类
        split_cluster_num = self.point_num // self.sub_issue_max_size
        self.path = Q_means.divide_clusters(self.points, split_cluster_num, self.env, self.backend, self.max_qubit_num)
        print(len(self.path))

        # 绘制
        # cluster_num = len(self.path)
        # colors = plt.cm.rainbow(np.linspace(0, 1, cluster_num))
        # for i in range(cluster_num):
        #     cur_cluster = self.path[i]
        #     plt.scatter(cur_cluster.centroid[0], cur_cluster.centroid[1], color=colors[i], s=30, marker='x')
        #     for point in cur_cluster.elements:
        #         plt.scatter(point[0], point[1], color=colors[i], s=5)
        # plt.show()

        # 多次合并互为最近邻的聚类
        while len(self.path) > self.sub_issue_max_size:
            neighbors = [0 for _ in range(len(self.path))]
            for i in range(len(self.path)):
                min_len = 1000000
                min_neighbor = -1
                for j in range(len(self.path)):
                    if i == j:
                        continue
                    cur_len = cal_similarity(self.path[i].centroid, self.path[j].centroid)
                    if cur_len < min_len:
                        min_len = cur_len
                        min_neighbor = j
                neighbors[i] = min_neighbor
            # 判断是否有可以合并的聚类
            delete_index = []
            for i in range(len(self.path)):
                for j in range(i + 1, len(self.path)):
                    if neighbors[i] == j and neighbors[j] == i:
                        self.path[i] = MultiCluster(None, [self.path[i], self.path[j]])
                        self.path[i].cal_centroid()
                        delete_index.append(j)
            delete_index.sort(reverse=True)
            for i in range(len(delete_index)):
                self.path.pop(delete_index[i])

        # 寻找最短回路
        self.path = [MultiCluster(None, self.path)]
        self.path[0].find_optimal_circle(self.max_qubit_num)
        self.path = self.path[0].elements

        # 绘制
        # cluster_num = len(self.path)
        # colors = plt.cm.rainbow(np.linspace(0, 1, cluster_num))
        # for i in np.arange(cluster_num):
        #     cur_multi_cluster = self.path[i]
        #     plt.scatter(cur_multi_cluster.centroid[0], cur_multi_cluster.centroid[1], color=colors[i], s=5)
        # x_values = [*[cur_cluster.centroid[0] for cur_cluster in self.path], self.path[0].centroid[0]]
        # y_values = [*[cur_cluster.centroid[1] for cur_cluster in self.path], self.path[0].centroid[1]]
        # plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
        # plt.show()

        # 将multi-cluster一层一层返回到single cluster
        is_terminal = False
        while not is_terminal:
            is_terminal = True
            for i in range(len(self.path) - 1, -1, -1):
                if self.path[i].class_type == 'Multi':
                    # 是multi-cluster
                    is_terminal = False
                    # 将此multi-cluster解构
                    tmp_multi_cluster = MultiCluster(None, [self.path[i - 1], *self.path[i].elements,
                                                            self.path[(i + 1) % len(self.path)]])
                    tmp_multi_cluster.find_optimal_path(self.max_qubit_num)
                    self.path[i: i + 1] = tmp_multi_cluster.elements[1: -1]

        # for i in range(len(self.path)):
        #     print(i, "-th centroid: ", self.path[i].centroid)
        #     print(i, "-th points: ", self.path[i].points)

        # 为每一个底层聚类设置起点和终点，重新排列节点顺序
        for i in range(len(self.path)):
            self.path[i - 1].tail, self.path[i].head = TSPSolution.find_diff_clusters_connector(self.path[i - 1],
                                                                                                self.path[i])
            if i > 0:
                self.path[i - 1].determine_head_and_tail()
        self.path[len(self.path) - 1].determine_head_and_tail()
        for i in range(len(self.path)):
            self.path[i].find_optimal_path(self.max_qubit_num)

        # 还原到单个节点状态
        for i in range(len(self.path) - 1, -1, -1):
            self.path[i: i + 1] = self.path[i].elements

        # 绘制
        for i in np.arange(self.point_num):
            plt.plot([self.path[i - 1][0], self.path[i][0]], [self.path[i - 1][1], self.path[i][1]], linewidth=1,
                     marker='o', markersize=2)
        plt.show()

        # 将离群点加入到最终结果中
        # for i in range(len(outliers)):
        #     min_dist = 10000
        #     target_index = -1
        #     for j in range(len(self.path)):
        #         tmp_dist = cal_similarity(self.path[j - 1], outliers[i]) + cal_similarity(outliers[i], self.path[
        #             j]) - cal_similarity(self.path[j - 1], self.path[j])
        #         if tmp_dist < min_dist:
        #             target_index = j
        #             min_dist = tmp_dist
        #     self.path.insert(target_index, outliers[i])

        # 将节点映射成索引
        # for i in range(len(self.path)):
        #     self.path[i] = self.point_map[self.path[i]]

    def cal_total_cost(self):
        dist = 0.
        for i in range(self.point_num):
            dist += cal_similarity(self.path[i - 1], self.path[i])
        return dist

    def get_accuracy(self):
        dist = self.cal_total_cost()
        print("distance: ", dist)

        opt_cost = float(read_dataset('opt_cost', self.point_num)[0])
        print("optimal cost: ", opt_cost)

        print("accuracy: ", dist/opt_cost)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QACSA')
    parser.add_argument('--file_name', '-f', type=str, default='ulysses16.tsp', help='Dataset of TSP')
    parser.add_argument('--scale', '-s', type=int, default=16, help='The scale of dataset')
    parser.add_argument('--env', '-e', type=str, default='sim',
                        help='The environment to run program, parameter: "sim"; "remote_sim"; "real"')
    parser.add_argument('--backend', '-b', type=str, default=None, help='The backend to run program')
    parser.add_argument('--max_qubit_num', '-m', type=int, default=15,
                        help='The maximum number of qubits in the backend')

    args = parser.parse_args()

    test = TSPSolution(args.file_name, args.scale, args.env, args.backend, args.max_qubit_num)
    test.divide_sub_issue()
    print(test.path)
    test.get_accuracy()
