from scipy.spatial import ConvexHull, QhullError

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
    def __init__(self, centroid, element_num, elements, class_type):
        self.head = None
        self.tail = None
        self.centroid = centroid
        self.elements = elements
        self.element_num = element_num
        self.class_type = class_type

    def determine_head_and_tail(self):
        if self.element_num == 1:
            return
        tmp_elements = [self.head, self.tail]
        for element in self.elements:
            if element != self.head and element != self.tail:
                tmp_elements.insert(-1, element)
        self.elements = tmp_elements

    def reorder(self, path):
        new_elements = []
        for i in range(len(path)):
            new_elements.append(self.elements[path[i]])
        self.elements = new_elements

    def path_to_circle(self):
        self.element_num += 1
        self.elements.append(self.elements[0])

    def circle_to_path(self):
        self.element_num -= 1
        self.elements.pop()


class SingleCluster(BaseCluster):
    def __init__(self, centroid, points):
        super(SingleCluster, self).__init__(centroid, len(points), points, 'Single')
        self.convex_hull = []
        self.find_convex_hull()

    def get_nodes_in_path(self):
        return self.elements

    def find_optimal_circle(self, total_qubit_num):
        self.path_to_circle()
        self.find_optimal_path(total_qubit_num)
        self.circle_to_path()

    def find_optimal_path(self, total_qubit_num):
        # path = OptimalPath(self.point_num, self.points, total_qubit_num).main()
        # self.reorder(path)

        cur_path = [0]
        opt_path = [0 for _ in range(self.element_num)]
        min_len = 100000
        is_chosen = [False for _ in range(self.element_num)]
        find_optimal_path(self.elements, cur_path, 0, opt_path, min_len, is_chosen)
        self.reorder(opt_path)

    def find_convex_hull(self):
        try:
            hull = ConvexHull(self.elements)
            hull_vertices = hull.vertices
            for vertex in hull_vertices:
                self.convex_hull.append(self.elements[vertex])
        except QhullError:
            # 说明这个点集都沿着同一条直线排列
            max_dist = -1
            for i in range(self.element_num):
                tmp_dist = cal_similarity(self.elements[i - 1], self.elements[i])
                if tmp_dist > max_dist:
                    max_dist = tmp_dist
                    self.convex_hull = [self.elements[i - 1], self.elements[i]]

    def get_convex_hull(self):
        return [point for point in self.convex_hull if point != self.head and point != self.tail]

    def get_point_num(self):
        return self.element_num


class MultiCluster(BaseCluster):
    def __init__(self, centroid, clusters):
        super(MultiCluster, self).__init__(centroid, len(clusters), clusters, 'Multi')
        self.point_num = sum(cluster.get_point_num() for cluster in clusters)

    def get_nodes_in_path(self):
        return [cluster.centroid for cluster in self.elements]

    def find_optimal_circle(self, total_qubit_num):
        self.path_to_circle()
        self.find_optimal_path(total_qubit_num)
        self.circle_to_path()

    def find_optimal_path(self, total_qubit_num):
        cur_path = [0]
        opt_path = [0 for _ in range(self.element_num)]
        min_len = 100000
        is_chosen = [False for _ in range(self.element_num)]
        find_optimal_path(self.get_nodes_in_path(), cur_path, 0, opt_path, min_len, is_chosen)
        self.reorder(opt_path)

    def get_point_num(self):
        return self.point_num

    def cal_centroid(self):
        if self.centroid is None:
            self.centroid = [0 for _ in range(2)]
        for cluster in self.elements:
            self.centroid[0] += cluster.centroid[0] * cluster.get_point_num()
            self.centroid[1] += cluster.centroid[1] * cluster.get_point_num()
        self.centroid = [self.centroid[0] / self.point_num, self.centroid[1] / self.point_num]

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
