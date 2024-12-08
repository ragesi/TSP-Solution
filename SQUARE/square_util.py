import sys
import os
import numpy as np

from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from utils import util


def spectral_clustering(points, cluster_num):
    k_neighbors = int(len(points) / cluster_num)
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nearest_neighbors_matrix = nn.fit(points).kneighbors_graph(mode='distance')

    sc = SpectralClustering(n_clusters=cluster_num, affinity='precomputed')
    y_pred = sc.fit_predict(nearest_neighbors_matrix)

    clusters = [[] for _ in range(cluster_num)]
    for i in range(len(y_pred)):
        clusters[y_pred[i]].append(tuple(points[i]))
    return clusters


def cal_dist_point_to_line(point, line_start, line_end):
    vec_of_point = point - line_start
    vec_of_line = line_end - line_start

    dist_vec_of_point = np.linalg.norm(vec_of_point)
    dist_vec_of_line = np.linalg.norm(vec_of_line)

    # 计算点到直线的垂线距离
    return abs(
        dist_vec_of_point * np.sin(np.arccos(np.dot(vec_of_point, vec_of_line) / dist_vec_of_line / dist_vec_of_point)))


def find_diff_clusters_connector(cluster_1, cluster_2):
    points_1 = cluster_1.get_convex_hull()
    points_2 = cluster_2.get_convex_hull()
    conn_min_dist = 10000
    conn_begin = 0
    conn_end = 0
    for i in range(len(points_1)):
        for j in range(len(points_2)):
            cur_dist = util.cal_similarity(points_1[i], points_2[j])
            if cur_dist < conn_min_dist:
                conn_min_dist = cur_dist
                conn_begin = i
                conn_end = j
    return points_1[conn_begin], points_2[conn_end]


def draw_result(point_num, ordered_cycle):
    # painting
    for i in np.arange(point_num):
        plt.plot([ordered_cycle[i - 1][0], ordered_cycle[i][0]], [ordered_cycle[i - 1][1], ordered_cycle[i][1]],
                 linewidth=1, marker='o', markersize=2)
    plt.show()
