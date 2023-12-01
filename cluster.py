class BaseCluster:
    def __init__(self, end, points, class_type):
        self.head = 0
        self.tail = end
        self.points = points
        self.class_type = class_type

    def determine_head_and_tail(self):
        self.points[0], self.points[self.head] = self.points[self.head], self.points[0]
        self.points[-1], self.points[self.tail] = self.points[self.tail], self.points[-1]

    def reorder(self, path):
        new_points = []
        for i in range(len(path)):
            new_points.append(self.points[path[i]])
        self.points = new_points


class SingleCluster(BaseCluster):
    def __init__(self, centroid, points):
        super().__init__(len(points) - 1, points, 0)
        self.centroid = centroid
        self.should_split = True

    def get_nodes_in_path(self):
        return self.points


class Clusters(BaseCluster):
    def __init__(self, centroids, points):
        super().__init__(len(centroids) - 1, [], 1)
        for i in range(len(centroids)):
            self.points.append(SingleCluster(centroids[i], points[i]))

    def get_nodes_in_path(self):
        return [centroid for centroid in self.points]
