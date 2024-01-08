from TSP_path.optimal_path import OptimalPath
import math as m


def cal_similarity(point1, point2):
    return m.sqrt(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))


def find_optimal_path(points, cur_path, cur_len, opt_path, min_len, is_chosen):
    if len(cur_path) == (len(points) - 1):
        # 只剩下终点
        cur_len += cal_similarity(points[-1], points[cur_path[-1]])
        if cur_len < min_len:
            min_len = cur_len
            for i in range(len(cur_path)):
                opt_path[i] = cur_path[i]
            opt_path[-1] = len(points) - 1
        return min_len

    point_num = len(points)
    for i in range(1, point_num - 1):
        if is_chosen[i]:
            continue
        tmp_len = cal_similarity(points[i], points[cur_path[-1]])
        cur_path.append(i)
        is_chosen[i] = True

        min_len = find_optimal_path(points, cur_path, cur_len + tmp_len, opt_path, min_len, is_chosen)

        is_chosen[i] = False
        cur_path.pop()

    return min_len


class BaseCluster:
    def __init__(self, centroid, point_num, points, class_type):
        self.head = -1
        self.tail = -1
        self.centroid = centroid
        self.points = points
        self.point_num = point_num
        self.class_type = class_type

    def determine_head_and_tail(self):
        if self.point_num == 1:
            return
        tmp_points = [self.points[self.head], self.points[self.tail]]
        for i in range(self.point_num - 1, -1, -1):
            if i == self.head or i == self.tail:
                continue
            tmp_points.insert(1, self.points[i])
        self.points = tmp_points

    def reorder(self, path):
        new_points = []
        for i in range(len(path)):
            new_points.append(self.points[path[i]])
        self.points = new_points

    def path_to_circle(self):
        self.point_num += 1
        self.points.append(self.points[0])

    def circle_to_path(self):
        self.point_num -= 1
        self.points.pop()


class SingleCluster(BaseCluster):
    def __init__(self, centroid, points):
        super(SingleCluster, self).__init__(centroid, len(points), points, 'Single')
        self.should_split = True

    def get_nodes_in_path(self):
        return self.points

    def find_optimal_circle(self, total_qubit_num):
        self.path_to_circle()
        self.find_optimal_path(total_qubit_num)
        self.circle_to_path()

    def find_optimal_path(self, total_qubit_num):
        # path = OptimalPath(self.point_num, self.points, total_qubit_num).main()
        # self.reorder(path)

        cur_path = [0]
        opt_path = [0 for _ in range(self.point_num)]
        min_len = 100000
        is_chosen = [False for _ in range(self.point_num)]
        find_optimal_path(self.points, cur_path, 0, opt_path, min_len, is_chosen)
        self.reorder(opt_path)

    def can_be_split(self, sub_issue_max_size):
        if self.point_num <= sub_issue_max_size:
            self.should_split = False


class MultiCluster(BaseCluster):
    def __init__(self, centroid, clusters):
        super(MultiCluster, self).__init__(centroid, len(clusters), clusters, 'Multi')

    def get_nodes_in_path(self):
        return [cluster.centroid for cluster in self.points]

    def find_optimal_circle(self, total_qubit_num):
        self.path_to_circle()
        self.find_optimal_path(total_qubit_num)
        self.circle_to_path()

    def find_optimal_path(self, total_qubit_num):
        cur_path = [0]
        opt_path = [0 for _ in range(self.point_num)]
        min_len = 100000
        is_chosen = [False for _ in range(self.point_num)]
        find_optimal_path(self.get_nodes_in_path(), cur_path, 0, opt_path, min_len, is_chosen)
        self.reorder(opt_path)


# class Clusters(BaseCluster):
#     def __init__(self, centroids, points):
#         super().__init__(len(centroids), [], 1)
#         for i in range(len(centroids)):
#             self.points.append(SingleCluster(centroids[i], points[i]))
#
#     def get_nodes_in_path(self):
#         return [point.centroid for point in self.points]
#
#     def find_optimal_circle(self, total_qubit_num):
#         self.path_to_circle()
#         self.find_optimal_path(total_qubit_num)
#         self.circle_to_path()
#
#     def find_optimal_path(self, total_qubit_num):
#         # path = OptimalPath(self.point_num, self.get_nodes_in_path(), total_qubit_num).main()
#         # self.reorder(path)
#
#         cur_path = [0]
#         opt_path = [0 for _ in range(self.point_num)]
#         min_len = 100000
#         is_chosen = [False for _ in range(self.point_num)]
#         find_optimal_path(self.get_nodes_in_path(), cur_path, 0, opt_path, min_len, is_chosen)
#         self.reorder(opt_path)
