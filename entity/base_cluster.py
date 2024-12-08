import os
import sys
import math as m
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from QUOTA.quota_main import OptimalPath
from utils import util


class BaseCluster:
    def __init__(self, centroid, element_num, elements, class_type, env, backend, print_detail=False):
        self.head = None
        self.tail = None
        self.centroid = centroid
        self.elements = elements
        self.element_num = element_num
        self.class_type = class_type

        self.env = env
        self.backend = backend
        self.print_detail = print_detail

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
