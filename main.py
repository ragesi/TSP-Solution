import math as m
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors

from clustering.Q_means import QMeans
from cluster import SingleCluster, MultiCluster, cal_similarity


class TSPSolution:
    def __init__(self):
        self.points = []
        self.point_map = dict()
        self.path = []

        self.sub_issue_max_size = 6
        self.max_qubit_num = 27

    def get_data(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        lines = lines[8: -1]
        for line in lines:
            tmp_point = line.strip().split(' ')
            tmp_point = [int(x) for x in tmp_point]
            tmp_tuple = (tmp_point[1], tmp_point[2])
            self.points.append(tmp_tuple)
            self.point_map[tmp_tuple] = tmp_point[0]

    @staticmethod
    def find_diff_clusters_connector(cluster_1, cluster_2):
        points_1 = cluster_1.get_nodes_in_path()
        points_2 = cluster_2.get_nodes_in_path()
        conn_begin = 1 if (cluster_1.head == 0 and cluster_1.point_num > 1) else 0
        conn_end = 1 if (cluster_2.tail == 0 and cluster_2.point_num > 1) else 0
        conn_min_dist = cal_similarity(points_1[conn_begin], points_2[conn_end])

        for i in range(len(points_1)):
            if i == cluster_1.head:
                continue
            for j in range(len(points_2)):
                if j == cluster_2.tail:
                    continue
                cur_dist = cal_similarity(points_1[i], points_2[j])
                if cur_dist < conn_min_dist:
                    conn_min_dist = cur_dist
                    conn_begin = i
                    conn_end = j

        return conn_begin, conn_end

    def divide_sub_issue(self):
        if len(self.points) < self.sub_issue_max_size:
            cur_cluster = SingleCluster(None, self.points)
            cur_cluster.find_optimal_circle(self.max_qubit_num)
            cur_order = cur_cluster.get_nodes_in_path()
            for point in cur_order:
                self.path.append(self.point_map.get(point))
            return

        # 第一次聚类，之后处理的最小粒度从数据点变成聚类中心点
        # 改成谱聚类
        points = np.array(self.points)
        k_neighbors = 5
        nn = NearestNeighbors(n_neighbors=k_neighbors)
        nearest_neighbors_matrix = nn.fit(points).kneighbors_graph(mode='distance')

        split_cluster_num = m.ceil(1.0 * len(self.points) / 6)
        sc = SpectralClustering(n_clusters=split_cluster_num, affinity='precomputed')
        y_pred = sc.fit_predict(nearest_neighbors_matrix)
        self.path = [[] for _ in range(split_cluster_num)]
        for i in range(len(y_pred)):
            self.path[y_pred[i]].append(self.points[i])
        for i in range(split_cluster_num):
            # 计算聚类中心点
            tmp_centroid = [1.0 * sum(point[0] for point in self.path[i]) / len(self.path[i]),
                            1.0 * sum(point[1] for point in self.path[i]) / len(self.path[i])]
            self.path[i] = SingleCluster(tmp_centroid, self.path[i])

        # split_cluster_num = m.ceil(1.0 * len(self.points) / 6)
        # self.path = QMeans(self.points, split_cluster_num).main()
        # print(self.path)

        # 绘制
        cluster_num = len(self.path)
        colors = plt.cm.rainbow(np.linspace(0, 1, cluster_num))
        for i in range(cluster_num):
            cur_cluster = self.path[i]
            plt.scatter(cur_cluster.centroid[0], cur_cluster.centroid[1], color=colors[i], s=30, marker='x')
            for point in cur_cluster.points:
                plt.scatter(point[0], point[1], color=colors[i], s=5)
        plt.show()

        # 多次合并互为最近邻的聚类
        is_terminal = False
        while (not is_terminal) and len(self.path) > 10:
            is_terminal = True

            neighbors = [0 for _ in range(len(self.path))]
            for i in range(len(self.path)):
                min_len = 100000
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
                        is_terminal = False
                        tmp_centroid = [1.0 * (self.path[i].centroid[0] + self.path[j].centroid[0]) / 2,
                                        1.0 * (self.path[i].centroid[1] + self.path[j].centroid[1]) / 2]
                        self.path[i] = MultiCluster(tmp_centroid, [self.path[i], self.path[j]])
                        delete_index.append(j)
            delete_index.sort(reverse=True)
            for i in range(len(delete_index)):
                self.path.pop(delete_index[i])

        # 寻找最短回路
        self.path = [MultiCluster(None, self.path)]
        self.path[0].find_optimal_circle(27)
        self.path = self.path[0].points

        # 绘制
        cluster_num = len(self.path)
        colors = plt.cm.rainbow(np.linspace(0, 1, cluster_num))
        for i in np.arange(cluster_num):
            cur_multi_cluster = self.path[i]
            plt.scatter(cur_multi_cluster.centroid[0], cur_multi_cluster.centroid[1], color=colors[i], s=5)
        x_values = [*[cur_cluster.centroid[0] for cur_cluster in self.path], self.path[0].centroid[0]]
        y_values = [*[cur_cluster.centroid[1] for cur_cluster in self.path], self.path[0].centroid[1]]
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
        plt.show()

        # 将multi-cluster一层一层返回到single cluster
        is_terminal = False
        while not is_terminal:
            is_terminal = True
            for i in range(len(self.path) - 1, -1, -1):
                if self.path[i].class_type == 'Multi':
                    # 是multi-cluster
                    is_terminal = False
                    # 将此multi-cluster解构
                    tmp_multi_cluster = MultiCluster(None, [self.path[i - 1], *self.path[i].points,
                                                            self.path[(i + 1) % len(self.path)]])
                    tmp_multi_cluster.find_optimal_path(self.max_qubit_num)
                    self.path[i: i + 1] = tmp_multi_cluster.points[1: -1]

        # 绘制
        cluster_num = len(self.path)
        colors = plt.cm.rainbow(np.linspace(0, 1, cluster_num))
        for i in np.arange(cluster_num):
            cur_multi_cluster = self.path[i]
            plt.scatter(cur_multi_cluster.centroid[0], cur_multi_cluster.centroid[1], color=colors[i], s=5)
        x_values = [*[cur_cluster.centroid[0] for cur_cluster in self.path], self.path[0].centroid[0]]
        y_values = [*[cur_cluster.centroid[1] for cur_cluster in self.path], self.path[0].centroid[1]]
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
        plt.show()

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
            self.path[i: i + 1] = self.path[i].points
        for i in range(len(self.path)):
            self.path[i] = self.point_map[self.path[i]]

        # 下面是错误的，不能再次聚类了
        # 再次聚类，将single cluster 聚类成multi-cluster
        # tmp_points = [cluster.centroid for cluster in self.path]
        # split_cluster_num = m.ceil(1.0 * len(tmp_points) / 6)
        # split_cluster = QMeans(tmp_points, split_cluster_num).main()
        # for cluster in split_cluster:
        #     tmp_clusters = []
        #     for point in cluster.points:
        #         for i in range(len(self.path)):
        #             if point[0] == self.path[i].centroid[0] and point[1] == self.path[i].centroid[1]:
        #                 tmp_clusters.append(self.path[i])
        #     self.path.append(MultiCluster(cluster.centroid, tmp_clusters))
        # self.path = self.path[-split_cluster_num:]
        #
        # # 聚类聚到只有一个类为止
        # self.path = [MultiCluster(None, self.path)]
        #
        # # 开始计算最短回路
        # self.path[0].find_optimal_circle(27)
        # self.path = self.path[0].points
        #
        # # 绘制
        # cluster_num = len(self.path)
        # colors = plt.cm.rainbow(np.linspace(0, 1, cluster_num))
        # for i in np.arange(cluster_num):
        #     cur_multi_cluster = self.path[i]
        #     plt.scatter(cur_multi_cluster.centroid[0], cur_multi_cluster.centroid[1], color=colors[i],
        #                 s=30, marker='x')
        #     for cluster in cur_multi_cluster.points:
        #         plt.scatter(cluster.centroid[0], cluster.centroid[1], color=colors[i], s=5)
        # x_values = [*[cur_cluster.centroid[0] for cur_cluster in self.path], self.path[0].centroid[0]]
        # y_values = [*[cur_cluster.centroid[1] for cur_cluster in self.path], self.path[0].centroid[1]]
        # plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
        # plt.show()
        #
        # # 找到每个multi-cluster中的起点cluster和终点cluster
        # for i in range(len(self.path)):
        #     self.path[i - 1].tail, self.path[i].head = TSPSolution.find_diff_clusters_connector(self.path[i - 1],
        #                                                                                         self.path[i])
        # for i in range(len(self.path)):
        #     self.path[i].determine_head_and_tail()
        # for i in range(len(self.path)):
        #     self.path[i].find_optimal_path(27)
        #
        # # 最终解构成single cluster的状态
        # tmp_path = []
        # for i in range(len(self.path)):
        #     tmp_path += self.path[i].points
        # self.path = tmp_path

        # 寻找最底层的每个聚类的最短路径
        # 找到每个聚类里的起点和终点
        # for i in range(len(self.path)):
        #     self.path[i - 1].tail, self.path[i].head = TSPSolution.find_diff_clusters_connector(self.path[i - 1],
        #                                                                                         self.path[i])
        # for i in range(len(self.path)):
        #     self.path[i].determine_head_and_tail()
        # for i in range(len(self.path)):
        #     self.path[i].find_optimal_path(27)
        #
        # # 解构成最底层的回路
        # tmp_path = []
        # for i in range(len(self.path)):
        #     tmp_path += self.path[i].points
        # self.path = tmp_path

        # split_cluster_num = min(m.ceil(1.0 * len(self.points) / self.sub_issue_max_size), self.sub_issue_max_size - 1)
        # split_cluster = QMeans(self.points, split_cluster_num).main()
        # split_cluster.find_optimal_circle(self.max_qubit_num)
        # self.path += split_cluster.points
        # for i in range(len(self.path)):
        #     self.path[i].can_be_split(self.sub_issue_max_size)
        #     print(self.path[i].points)
        # is_finish = False
        # while not is_finish:
        #     is_finish = True
        #     # divide current layer clusters to smaller clusters
        #     for i in range(len(self.path)):
        #         if not self.path[i].should_split:
        #             continue
        #
        #         is_finish = False
        #         split_cluster_num = min(m.ceil(1.0 * len(self.path[i].points) / self.sub_issue_max_size),
        #                                 self.sub_issue_max_size)
        #         split_clusters = QMeans(self.path[i].points, split_cluster_num).main()
        #         self.path[i] = split_clusters
        #
        #     # after division, get every cluster's begin and end of centroids
        #     for i in range(len(self.path)):
        #         # get the source and target of every cluster
        #         self.path[i - 1].tail, self.path[i].head = TSPSolution.find_diff_clusters_connector(self.path[i - 1],
        #                                                                                             self.path[i])
        #         if i > 0:
        #             self.path[i - 1].determine_head_and_tail()
        #     self.path[len(self.path) - 1].determine_head_and_tail()
        #
        #     # calculate every cluster's order of centroids
        #     for i in range(len(self.path) - 1, -1, -1):
        #         if self.path[i].class_type == 0:
        #             continue
        #
        #         cur_clusters = self.path.pop(i)
        #         cur_clusters.find_optimal_path(self.max_qubit_num)
        #         self.path[i: i] = cur_clusters.points
        #         for j in range(cur_clusters.point_num):
        #             self.path[i + j].can_be_split(self.sub_issue_max_size)

        # cluster_num = len(self.path)
        # colors = plt.cm.rainbow(np.linspace(0, 1, cluster_num))
        # for i in np.arange(cluster_num):
        #     cur_multi_cluster = self.path[i]
        #     plt.scatter(cur_multi_cluster.centroid[0], cur_multi_cluster.centroid[1], color=colors[i],
        #                 s=30, marker='x')
        #     for point in cur_multi_cluster.points:
        #         plt.scatter(point[0], point[1], color=colors[i], s=5)
        # x_values = [*[cur_cluster.centroid[0] for cur_cluster in self.path], self.path[0].centroid[0]]
        # y_values = [*[cur_cluster.centroid[1] for cur_cluster in self.path], self.path[0].centroid[1]]
        # plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=2, label='折线图')
        # plt.show()

        # when all clusters are divided to minimum, find optimal paths for all clusters respectively
        # for i in range(len(self.path) - 1, -1, -1):
        #     cur_clusters = self.path.pop(i)
        #     cur_clusters.find_optimal_path(self.max_qubit_num)
        #     for j in range(cur_clusters.point_num):
        #         self.path.insert(i + j, self.point_map[cur_clusters.points[j]])


if __name__ == '__main__':
    file_path = 'dataset/xqf131.tsp'
test = TSPSolution()
test.get_data(file_path)
test.divide_sub_issue()
print(test.path)
