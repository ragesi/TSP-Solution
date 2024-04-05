from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator

import numpy as np
import matplotlib.pyplot as plt

from utils import inner_product, read_dataset
from dataset import test


class ConvexHull:
    def __init__(self, points):
        self.points = points
        self.start = None
        self.base_vec = [1, 0]
        self.base_index = None

        self.init_start()
        self.convex_hull_set = [self.start]

    def init_start(self):
        # finding the point with minimum y
        min_index = 0
        min_y = self.points[0][1]
        for i in range(1, len(self.points)):
            if self.points[i][1] < min_y:
                min_y = self.points[i][1]
                min_index = i
        self.base_index = min_index
        self.start = self.points[min_index]

    def normalization(self, vec):
        normalized_vec = np.array(vec)
        return normalized_vec / np.linalg.norm(normalized_vec)

    def find_convex_hull(self):
        while self.start != self.convex_hull_set[-1] or len(self.convex_hull_set) == 1:
            # finding a new boundary point
            cur_vec_list = list()
            for point in self.points:
                if point != self.convex_hull_set[-1]:
                    if len(self.convex_hull_set) >= 2 and point == self.convex_hull_set[-2]:
                        cur_vec_list.append([-self.base_vec[1], self.base_vec[0]])
                    else:
                        normalize_vec = self.normalization(np.array(point) - np.array(self.convex_hull_set[-1]))
                        normalize_vec = (normalize_vec + self.base_vec) / 2
                        normalize_vec = self.normalization(normalize_vec)
                        cur_vec_list.append(normalize_vec)

            next_hull_index = inner_product.get_max_inner_product(self.base_vec, cur_vec_list)
            if next_hull_index >= self.base_index:
                next_hull_index += 1
            self.convex_hull_set.append(self.points[next_hull_index])
            self.base_vec = self.normalization(np.array(self.convex_hull_set[-1]) - np.array(self.convex_hull_set[-2]))
            self.base_index = next_hull_index

        self.convex_hull_set = self.convex_hull_set[:-1]


if __name__ == '__main__':
    points = test.point_test_for_6
    test = ConvexHull(points)
    test.find_convex_hull()
    print(test.convex_hull_set)

    x_values = [point[0] for point in test.convex_hull_set]
    y_values = [point[1] for point in test.convex_hull_set]
    plt.scatter(x_values, y_values, marker='o', color='r', s=10)

    x_values = [point[0] for point in points if point not in test.convex_hull_set]
    y_values = [point[1] for point in points if point not in test.convex_hull_set]
    plt.scatter(x_values, y_values, marker='o', color='b', s=10)

    plt.show()
