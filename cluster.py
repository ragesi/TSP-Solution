class BaseCluster:
    def __init__(self):
        self.begin = 0
        self.end = 0


class SingleCluster(BaseCluster):
    def __init__(self, centroid, points, should_split=True):
        super().__init__()
        self.centroid = centroid
        self.points = points
        self.should_split = should_split
        self.end = len(self.points) - 1

    def swap(self):
        self.points[0], self.points[self.begin] = self.points[self.begin], self.points[0]
        self.points[-1], self.points[self.end] = self.points[self.end], self.points[-1]

    def get_nodes_in_path(self):
        return self.points


class Clusters(BaseCluster):
    def __init__(self, centroids, points):
        super().__init__()
        self.cluster_list = []
        for i in range(len(centroids)):
            self.cluster_list.append(SingleCluster(centroids[i], points[i]))
        self.end = len(self.cluster_list) - 1

    def swap(self):
        self.cluster_list[0], self.cluster_list[self.begin] = self.cluster_list[self.begin], self.cluster_list[0]
        self.cluster_list[-1], self.cluster_list[self.end] = self.cluster_list[self.end], self.cluster_list[-1]

    def reorder(self, path):
        new_clusters = []
        for i in range(len(path)):
            new_clusters.append(self.cluster_list[path[i]])
        self.cluster_list = new_clusters

    def get_nodes_in_path(self):
        return [centroid for centroid in self.cluster_list]
