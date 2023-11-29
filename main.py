import numpy as np
import math as m

from clustering.Q_means import QMeans
from TSP_route.optimal_route import OptimalRoute


def get_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = lines[8: -1]
    point_map = dict()
    clusters = []
    for line in lines:
        tmp_point = line.strip().split(' ')
        tmp_point = [int(x) for x in tmp_point]
        tmp_tuple = (tmp_point[1], tmp_point[2])
        clusters.append(tmp_tuple)
        point_map[tmp_tuple] = tmp_point[0]

    return point_map, clusters


def cal_distance(points):
    point_num = len(points)
    dist_adj = np.zeros((point_num - 1, point_num - 1))
    for i in range(point_num - 1):
        for j in range(point_num - 1):
            dist_adj[i][j] = abs(points[j + 1][0] - points[i][0]) + abs(points[j + 1][1] - points[i][1])
    dist_adj[0][-1] = 0.
    return dist_adj


def find_shortest_path(points_1, points_2):
    min_dist = 1000
    res_source = 0
    res_target = 0

    for i in range(len(points_1)):
        for j in range(len(points_2)):
            cur_dist = abs(points_1[i][0] - points_2[j][0]) + abs(points_1[i][1] - points_2[j][1])
            if cur_dist < min_dist:
                min_dist = cur_dist
                res_source = points_1[i]
                res_target = points_2[j]

    return res_source, res_target


def main(points):
    if len(points) <= 5:
        optimal_route = OptimalRoute(len(points))

    cluster_num = min(m.ceil(1.0 * len(points) / 6), 5)
    q_mean = QMeans(points, cluster_num, 10)
    q_mean.main()
    clusters = q_mean.clusters
    centroids = q_mean.centroids




def main(centroids, clusters):
    # TSP
    dist_adj = cal_distance(centroids)
    # TODO: 这里要删掉这个node_list参数，同时设置optimal_route的返回值
    optimal_route = OptimalRoute(len(centroids), [1, 2, 3, 4, 5], dist_adj, 27)
    output = optimal_route.async_grover()
    path = [centroids[0]]
    for i in range(len(output)):
        path.append(centroids[output[i] + 1])
    path.append(centroids[-1])

    # 找到下一层的中心点
    next_layer_centroids = []
    for i in range(len(clusters)):
        if len(clusters[i]) <= 6:
            next_layer_centroids.append(clusters[i])
        else:
            tmp_cluster_num = min(m.ceil(1.0 * len(clusters[i]) / 6), 6)
            q_mean = QMeans(clusters[i], tmp_cluster_num, 10)
            tmp_clusters = q_mean.clusters
            tmp_centroids = q_mean.centroids
            next_layer_centroids.append(tmp_centroids)

    # 找到下一层中每一个子路径的起点和终点


    cluster_num = min(m.ceil(1.0 * len(clusters) / 6), 6)
    q_mean = QMeans(clusters, cluster_num, 10)
    new_clusters = q_mean.clusters
    new_centroids = q_mean.centroids

    # using TSP to connect all centroids of current layer
    dist_adj = cal_distance(new_centroids)

    for i in range(cluster_num):



if __name__ == '__main__':
    # get data
    file_path = '/dataset/xqf131.tsp'
    point_map, clusters = get_data(file_path)
