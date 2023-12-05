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

        self.sub_issue_max_size = 6
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
            cur_cluster = SingleCluster(None, self.points)
            cur_cluster.find_optimal_circle(self.max_qubit_num)
            cur_order = cur_cluster.get_nodes_in_path()
            for point in cur_order:
                self.path.append(self.point_map.get(point))
            return

        split_cluster_num = min(m.ceil(1.0 * len(self.points) / self.sub_issue_max_size), self.sub_issue_max_size - 1)
        split_cluster = QMeans(self.points, split_cluster_num).main()
        split_cluster.find_optimal_circle(self.max_qubit_num)
        self.path += split_cluster.points
        for i in range(len(self.path)):
            self.path[i].can_be_split(self.sub_issue_max_size)
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
                split_clusters = QMeans(self.path[i].points, split_cluster_num).main()
                self.path[i] = split_clusters

            if is_finish:
                break

            # after division, get every cluster's begin and end of centroids
            for i in range(len(self.path)):
                # get the source and target of every cluster
                begin, end = TSPSolution.find_diff_clusters_connector(self.path[i - 1].get_nodes_in_path,
                                                                      self.path[i].get_nodes_in_path)
                self.path[i - 1].end = begin
                self.path[i].begin = end
                if i > 0:
                    self.path[i - 1].determine_head_and_tail()

            # calculate every cluster's order of centroids
            for i in range(len(self.path)):
                if self.path[i].class_type == 0:
                    continue

                cur_clusters = self.path.pop(i)
                cur_clusters.find_optimal_path(self.max_qubit_num)
                self.path[i: i] = cur_clusters.points
                for j in range(i, i + cur_clusters.point_num):
                    self.path[i + j].can_be_split(self.sub_issue_max_size)

        # when all clusters are divided to minimum, find optimal paths for all clusters respectively
        for i in range(len(self.path)):
            cur_clusters = self.path.pop(i)
            cur_clusters.find_optimal_path(self.sub_issue_max_size)
            for j in range(cur_clusters.point_num):
                self.path.insert(i + j, self.point_map[cur_clusters.points[j]])
