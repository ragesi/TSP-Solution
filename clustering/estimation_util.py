import numpy as np


def cal_cut_similarity(cluster_1, cluster_2, sigma) -> float:
    """
    calculating the similarity between different clusters
    :param cluster_1: list, the first cluster
    :param cluster_2: list, the second cluster
    :param sigma: float, the standard deviation
    """
    similarity = 0.
    for element_1 in cluster_1.elements:
        for element_2 in cluster_2.elements:
            cur_dist = np.linalg.norm(np.array(element_1) - np.array(element_2))
            similarity += np.exp(-np.square(np.array(cur_dist) / sigma) / 2)
    return similarity


def cal_cut_weights(cluster_1, cluster_2) -> float:
    """
    calculating the weights of different clusters
    """
    weights = 0.
    for element_1 in cluster_1.elements:
        for element_2 in cluster_2.elements:
            weights += np.linalg.norm(np.array(element_1) - np.array(element_2))
    return weights


def cal_similarity(cluster, sigma) -> float:
    """
    calculating the similarity in cluster
    :param cluster: list
    :param sigma: float, the standard deviation
    """
    similarity = 0.
    for i in range(cluster.element_num):
        for j in range(i + 1, cluster.element_num):
            cur_dist = np.linalg.norm(np.array(cluster.elements[i]) - np.array(cluster.elements[j]))
            similarity += np.exp(-np.square(np.array(cur_dist) / sigma) / 2)
    return similarity


def cal_weights(cluster) -> float:
    """
    calculating the weight of cluster
    :param cluster: list
    """
    weights = 0.
    for i in range(cluster.element_num):
        for j in range(i + 1, cluster.element_num):
            weights += np.linalg.norm(np.array(cluster.elements[i]) - np.array(cluster.elements[j]))
    return weights


def estimation_with_similarity(clusters, points, print_detail=False) -> tuple[float, float]:
    """
    estimating the performance of QMeans using two metrics: inter-graph and intra-graph similarity
    :param clusters: list, the result of QMeans
    :param points: list, all cities
    :param print_detail: boolean, whether to print the execution detail
    """
    max_dist = 0.0
    min_dist = 100000.
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            if dist > max_dist:
                max_dist = dist
            if dist < min_dist:
                min_dist = dist
    sigma = (max_dist - min_dist) * 0.15

    if print_detail:
        print("max_dist: ", max_dist)
        print("min_dist: ", min_dist)
        print("sigma: ", sigma)

    similarity = 0.
    cut_similarity = 0.
    for i in range(len(clusters)):
        similarity += cal_similarity(clusters[i], sigma)
        for j in range(i + 1, len(clusters)):
            cut_similarity += cal_cut_similarity(clusters[i], clusters[j], sigma)
    return similarity, cut_similarity


def estimation_with_weight(clusters) -> tuple[float, float]:
    """
    estimating the performance of QMeans using two metrics: inter-graph and intra-graph weight
    :param clusters: list, the result of QMeans
    """
    weights = 0.
    cut_weights = 0.
    for i in range(len(clusters)):
        weights += cal_weights(clusters[i])
        for j in range(i + 1, len(clusters)):
            cut_weights += cal_cut_weights(clusters[i], clusters[j])
    return weights, cut_weights
