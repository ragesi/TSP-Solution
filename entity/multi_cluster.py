import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from entity.base_cluster import BaseCluster
from utils import util


class MultiCluster(BaseCluster):
    def __init__(self, centroid, clusters, env, backend, print_detail=False):
        super(MultiCluster, self).__init__(centroid, len(clusters), clusters, 'Multi', env, backend, print_detail)
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
        util.find_optimal_path(self.get_nodes_in_path(), cur_path, 0, opt_path, min_len, is_chosen)
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
