import numpy as np
import math as m
from queue import Queue

from clustering.Q_means import QMeans
from TSP_path.optimal_path import OptimalPath
from cluster import SingleCluster, Clusters


class TSPSolution:
    def __init__(self, file_path):
        self.file_path = file_path
        self.points = []
        self.point_map = dict()
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
        res_begin = 0
        res_end = 0

        for i in range(len(points_1)):
            for j in range(len(points_2)):
                cur_dist = abs(points_1[i][0] - points_2[j][0]) + abs(points_1[i][1] - points_2[j][1])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    res_begin = i
                    res_end = j

        return res_begin, res_end

    def divide_sub_issue(self):
        if len(self.points) < self.sub_issue_max_size:
            optimal_path = OptimalPath(len(self.points) + 1, [*self.points, self.points[0]], self.max_qubit_num)
            tmp_path = optimal_path.main()
            tmp_path = tmp_path[:-1]
            for tmp_node in tmp_path:
                self.path.append(self.point_map.get(self.points[tmp_node]))
            return

        split_cluster_num = min(m.ceil(1.0 * len(self.points) / self.sub_issue_max_size), self.sub_issue_max_size - 1)
        split_centroids, split_points = QMeans(self.points, split_cluster_num).main()
        optimal_path = OptimalPath(len(split_centroids) + 1, [*split_centroids, split_centroids[0]], self.max_qubit_num)
        path = optimal_path.main()[:-1]
        for i in range(len(path)):
            cur_id = path[i]
            self.path.append(SingleCluster(split_centroids[cur_id], split_points[cur_id]))
            self.path[-1].should_split = False if split_points[cur_id] <= self.sub_issue_max_size else True
        is_finish = False
        while not is_finish:
            is_finish = True
            # divide current layer clusters to smaller clusters
            for i in range(len(self.path)):
                if not self.path[i].should_split:
                    continue

                is_finish = False
                split_cluster_num = min(m.ceil(1.0 * len(self.path[i].points) / self.sub_issue_max_size),
                                        self.sub_issue_max_size)
                split_centroids, split_points = QMeans(self.path[i].points, split_cluster_num).main()
                split_clusters = Clusters(split_centroids, split_points)
                self.path[i] = split_clusters

            if is_finish:
                break

            # after division, get every cluster's begin and end of centroids
            for i in range(len(self.path)):
                if type(self.path[i]) == 'SingleCluster':
                    continue

                # get the source and target of every cluster
                begin, end = TSPSolution.find_diff_clusters_connector(self.path[i - 1].get_nodes_in_path,
                                                                          self.path[i].get_nodes_in_path)
                self.path[i - 1].end = begin
                self.path[i].begin = end

            # calculate every cluster's order of centroids
            for i in range(len(self.path)):
                if type(self.path[i]) == 'SingleCluster':
                    continue

                self.path[i].swap()
                cur_centroids = self.path[i].get_nodes_in_path()
                cur_path = OptimalPath(len(cur_centroids), cur_centroids, self.max_qubit_num).main()
                cur_clusters = self.path.pop(i)
                for j in range(len(cur_path)):
                    self.path.insert(j + i, cur_clusters.cluster_list[cur_path[j]])

        # TODO: 当聚类中的节点数量少于3时的解决方法
        # TODO: 优化整体代码架构
        # TODO: 重构QMeans代码，它的代码中应该可以用到cluster结构体
        # when all clusters are divided to minimum, find optimal paths for all clusters respectively
        for i in range(len(self.path)):
            self.path[i].swap()
            cur_path = OptimalPath(len(self.path[i].points), self.path[i].points, self.max_qubit_num).main()
            cur_clusters = self.path.pop(i)
            for j in range(len(cur_path)):
                self.path.insert(j + i, self.point_map[cur_clusters.points[cur_path[j]]])
