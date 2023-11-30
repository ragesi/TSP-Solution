import numpy as np
import math as m
from queue import Queue

from clustering.Q_means import QMeans
from TSP_path.optimal_path import OptimalPath


class Cluster:
    def __init__(self, centroids, points):
        self.centroids = centroids
        self.points = points
        self.centroids_start = None
        self.centroids_end = None


class TSPSolution:
    def __init__(self, file_path):
        self.file_path = file_path
        self.points = []
        self.point_map = dict()
        self.clusters = []
        self.path = []

        self.sub_issue_max_size = 7
        self.max_qubit_num = 29

    def get_data(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        lines = lines[8: -1]
        for line in lines:
            tmp_point = line.strip().split(' ')
            tmp_point = [int(x) for x in tmp_point]
            tmp_tuple = (tmp_point[1], tmp_point[2])
            self.points.append(tmp_tuple)
            self.point_map[tmp_tuple] = tmp_point[0]

    @staticmethod
    def find_diff_clusters_connector(points_1, points_2):
        min_dist = 1000
        res_source = 0
        res_target = 0

        for i in range(len(points_1)):
            for j in range(len(points_2)):
                cur_dist = abs(points_1[i][0] - points_2[j][0]) + abs(points_1[i][1] - points_2[j][1])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    res_source = points_1[i]
                    res_target = points_2[j]

        return res_source, res_target

    def find_shortest_path(self, clusters):
        for i in range(len(clusters)):

    def divide_sub_issue(self):
        if len(self.points) < self.sub_issue_max_size:
            optimal_path = OptimalPath(len(self.points) + 1, [*self.points, self.points[0]], self.max_qubit_num)
            tmp_path = optimal_path.main()
            tmp_path = tmp_path[:-1]
            for tmp_node in tmp_path:
                self.path.append(self.point_map.get(self.points[tmp_node]))
            return

        is_finish = False
        self.clusters.append(Cluster(None, self.points))
        while not is_finish:
            is_finish = True
            for i in range(len(self.clusters)):
                cur_points = self.clusters[i].points
                if len(cur_points) <= self.sub_issue_max_size:
                    continue

                is_finish = False
                next_layer_cluster_num = min(m.ceil(1.0 * len(cur_points) / self.sub_issue_max_size),
                                             self.sub_issue_max_size - 1)
                q_mean = QMeans(cur_points, next_layer_cluster_num)
                cur_centroids, cur_clusters = q_mean.main()
                del self.clusters[i]
                for j in range(len(cur_centroids)):


    def main(self):

        paths = []
        queue = Queue()
        tmp_cluster_num = min(m.ceil(1.0 * len(self.points) / sub_problem_max_size), sub_problem_max_size - 1)
        q_mean = QMeans(self.points, tmp_cluster_num)
        centroids, clusters = q_mean.main()
        optimal_path = OptimalPath(len(centroids) + 1, [*centroids, centroids[0]], max_qubit_num)
        path = optimal_path.main()
        path = path[:-1]
        for i in range(len(path)):
            queue.put(Cluster(centroids[path[i]], clusters[path[i]], centroids[path[i - 1]],
                              centroids[path[(i + 1) % len(path)]]))


def main(centroids, clusters):
    # TSP
    dist_adj = cal_distance(centroids)
    # TODO: 这里要删掉这个node_list参数，同时设置optimal_route的返回值
    optimal_route = OptimalPath(len(centroids), dist_adj, 27)
    output = optimal_route.async_grover()
    path = [centroids[0]]
    for i in range(len(output)):
        path.append(centroids[output[i] + 1])
    path.append(centroids[-1])

    # 找到下一层的中心点
    next_layer_centroids = []
    for i in range(len(clusters)):
        if len(clusters[i]) <= 6:
            next_layer_centroids.append(clusters[i])
        else:
            tmp_cluster_num = min(m.ceil(1.0 * len(clusters[i]) / 6), 6)
            q_mean = QMeans(clusters[i], tmp_cluster_num, 10)
            tmp_clusters = q_mean.clusters
            tmp_centroids = q_mean.centroids
            next_layer_centroids.append(tmp_centroids)

    # 找到下一层中每一个子路径的起点和终点

    cluster_num = min(m.ceil(1.0 * len(clusters) / 6), 6)
    q_mean = QMeans(clusters, cluster_num, 10)
    new_clusters = q_mean.clusters
    new_centroids = q_mean.centroids

    # using TSP to connect all centroids of current layer
    dist_adj = cal_distance(new_centroids)

    for i in range(cluster_num):
