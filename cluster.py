from TSP_path.optimal_path import OptimalPath


class BaseCluster:
    def __init__(self, point_num, points, class_type):
        self.head = -1
        self.tail = -1
        self.points = points
        self.point_num = point_num
        self.class_type = class_type

    def determine_head_and_tail(self):
        self.points[0], self.points[self.head] = self.points[self.head], self.points[0]
        self.points[-1], self.points[self.tail] = self.points[self.tail], self.points[-1]

    def reorder(self, path):
        new_points = []
        for i in range(len(path)):
            new_points.append(self.points[path[i]])
        self.points = new_points

    def path_to_circle(self):
        self.point_num += 1
        self.points.append(self.points[0])

    def circle_to_path(self):
        self.point_num -= 1
        self.points.pop()


class SingleCluster(BaseCluster):
    def __init__(self, centroid, points):
        super().__init__(len(points), points, 0)
        self.centroid = centroid
        self.should_split = True

    def get_nodes_in_path(self):
        return self.points

    def find_optimal_circle(self, total_qubit_num):
        self.path_to_circle()
        self.find_optimal_path(total_qubit_num)
        self.circle_to_path()

    def find_optimal_path(self, total_qubit_num):
        path = OptimalPath(self.point_num, self.points, total_qubit_num).main()
        self.reorder(path)

    def can_be_split(self, sub_issue_max_size):
        if self.point_num <= sub_issue_max_size:
            self.should_split = False


class Clusters(BaseCluster):
    def __init__(self, centroids, points):
        super().__init__(len(centroids), [], 1)
        for i in range(len(centroids)):
            self.points.append(SingleCluster(centroids[i], points[i]))

    def get_nodes_in_path(self):
        return [centroid for centroid in self.points]

    def find_optimal_circle(self, total_qubit_num):
        self.path_to_circle()
        self.find_optimal_path(total_qubit_num)
        self.circle_to_path()

    def find_optimal_path(self, total_qubit_num):
        path = OptimalPath(self.point_num, self.get_nodes_in_path(), total_qubit_num).main()
        self.reorder(path)
