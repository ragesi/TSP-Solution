import sys
import os

from scipy.spatial import ConvexHull, QhullError

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from entity.base_cluster import BaseCluster
from utils import util
from QCHSA.qchsa_main import ConvexHull as QCH


class SingleCluster(BaseCluster):
    def __init__(self, centroid, points, env, backend, print_detail=False):
        super(SingleCluster, self).__init__(centroid, len(points), points, 'Single', env, backend, print_detail)
        self.convex_hull = []
        self.find_convex_hull()

    def calculate_centroid(self):
        x_sum, y_sum = 0, 0
        for e in self.elements:
            x_sum += e[0]
            y_sum += e[1]
        self.centroid = [x_sum / self.element_num, y_sum / self.element_num]

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
        util.find_optimal_path(self.elements, cur_path, 0, opt_path, min_len, is_chosen)
        self.reorder(opt_path)

    def classical_find_convex_hull(self):
        if len(self.elements) < 3:
            self.convex_hull = self.elements
        else:
            try:
                hull = ConvexHull(self.elements)
                hull_vertices = hull.vertices
                for vertex in hull_vertices:
                    self.convex_hull.append(self.elements[vertex])
            except QhullError:
                # 说明这个点集都沿着同一条直线排列
                max_dist = -1
                for i in range(self.element_num):
                    tmp_dist = util.cal_similarity(self.elements[i - 1], self.elements[i])
                    if tmp_dist > max_dist:
                        max_dist = tmp_dist
                        self.convex_hull = [self.elements[i - 1], self.elements[i]]

    def find_convex_hull(self):
        if len(self.elements) < 3:
            self.convex_hull = self.elements
        else:
            self.convex_hull = QCH(self.elements, self.env, self.backend, self.print_detail).find_convex_hull()

    def get_convex_hull(self):
        if self.element_num > 1:
            return [point for point in self.convex_hull if point != self.head and point != self.tail]
        else:
            return self.convex_hull

    def get_point_num(self):
        return self.element_num