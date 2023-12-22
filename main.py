import math as m

from clustering.Q_means import QMeans
from cluster import SingleCluster


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
        conn_min_dist = abs(points_1[conn_begin][0] - points_2[conn_end][0]) + abs(
            points_1[conn_begin][1] - points_2[conn_end][1])

        for i in range(len(points_1)):
            if i == cluster_1.head:
                continue
            for j in range(len(points_2)):
                if j == cluster_2.tail:
                    continue
                cur_dist = abs(points_1[i][0] - points_2[j][0]) + abs(points_1[i][1] - points_2[j][1])
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

        split_cluster_num = min(m.ceil(1.0 * len(self.points) / self.sub_issue_max_size), self.sub_issue_max_size - 1)
        split_cluster = QMeans(self.points, split_cluster_num).main()
        split_cluster.find_optimal_circle(self.max_qubit_num)
        self.path += split_cluster.points
        for i in range(len(self.path)):
            self.path[i].can_be_split(self.sub_issue_max_size)
            print(self.path[i].points)
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

            # after division, get every cluster's begin and end of centroids
            for i in range(len(self.path)):
                # get the source and target of every cluster
                self.path[i - 1].tail, self.path[i].head = TSPSolution.find_diff_clusters_connector(self.path[i - 1],
                                                                                                    self.path[i])
                if i > 0:
                    self.path[i - 1].determine_head_and_tail()
            self.path[len(self.path) - 1].determine_head_and_tail()

            # calculate every cluster's order of centroids
            for i in range(len(self.path) - 1, -1, -1):
                if self.path[i].class_type == 0:
                    continue

                cur_clusters = self.path.pop(i)
                cur_clusters.find_optimal_path(self.max_qubit_num)
                self.path[i: i] = cur_clusters.points
                for j in range(cur_clusters.point_num):
                    self.path[i + j].can_be_split(self.sub_issue_max_size)

        # when all clusters are divided to minimum, find optimal paths for all clusters respectively
        for i in range(len(self.path) - 1, -1, -1):
            cur_clusters = self.path.pop(i)
            cur_clusters.find_optimal_path(self.max_qubit_num)
            for j in range(cur_clusters.point_num):
                self.path.insert(i + j, self.point_map[cur_clusters.points[j]])


if __name__ == '__main__':
    file_path = 'dataset/pbn423.tsp'
    test = TSPSolution()
    test.get_data(file_path)
    test.divide_sub_issue()
    print(test.path)
