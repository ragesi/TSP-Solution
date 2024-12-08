import argparse
import sys
import os
import random
import math as m
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import clustering.cut_preparation as prep
from utils import execute, read_dataset
from entity.single_cluster import SingleCluster


# from dataset import test


class QAOACut:
    def __init__(self, points: list, theta: list, lamda: float, norm_threshold: float):
        """
        :param points: list, all cities in TSP
        :param theta: list, parameters of QAOA
        :param lamda: int, the parameter in the cost function of QNCut
        :param norm_threshold: float, the threshold when the degree matrix is normalized
        """
        self.points = points
        self.point_num = len(self.points)
        self.adj_matrix = prep.build_adj_matrix(self.points, self.point_num)
        deg_matrix = prep.build_deg_matrix(self.adj_matrix, self.point_num)
        self.deg_matrix = prep.scaling_up_deg_matrix(deg_matrix, self.point_num)
        self.norm_deg_matrix = prep.scaling_down_deg_matrix(deg_matrix, norm_threshold, self.point_num)

        self.theta = theta
        self.lamda = lamda
        self.min_theta = self.theta

        self.precision = 6
        self.step = 0.01
        self.epsilon = 0.001
        self.delta = 0.001

    def phase_gate_simple(self, gamma: float) -> QuantumCircuit:
        """
        the simple version of phase operator
        :param gamma: float, the parameter in phase operator
        """
        qram = QuantumRegister(self.point_num)
        qc = QuantumCircuit(qram)

        # The cost of cut
        for i in range(self.point_num):
            for j in range(i + 1, self.point_num):
                qc.cx(qram[i], qram[j])
                qc.rz(gamma * self.adj_matrix[i][j], qram[j])
                qc.cx(qram[i], qram[j])

        for i in range(self.point_num):
            qc.rz(-1 * gamma * self.lamda * self.deg_matrix[i], qram[i])

        return qc

    def phase_gate(self, gamma: float) -> QuantumCircuit:
        """
        the complete version of phase operator
        :param gamma: float, the parameter in phase operator
        """
        qram = QuantumRegister(self.point_num)
        eigen_vec = QuantumRegister(1)
        eigen_val = QuantumRegister(self.precision)
        anc = QuantumRegister(1)

        qc = QuantumCircuit(qram, eigen_vec, eigen_val, anc)

        # The cost of cut
        for i in range(self.point_num):
            for j in range(i + 1, self.point_num):
                qc.cx(qram[i], qram[j])
                qc.rz(gamma * self.adj_matrix[i][j], qram[j])
                qc.cx(qram[i], qram[j])

        for i in range(self.point_num):
            qc.rz(-1 * gamma * self.lamda * self.deg_matrix[i], qram[i])

        # The constraint of cut
        qc.append(prep.qpe(self.precision, self.norm_deg_matrix, self.point_num), [*qram, *eigen_vec, *eigen_val, *anc])

        # If the highest qubit of eigen_val is 1, the result is positive
        for i in range(self.point_num):
            qc.crz(-1 * gamma * self.lamda * self.deg_matrix[i], eigen_val[-1], qram[i])
        # Else if the highest qubit is 0, the result is negative
        qc.x(eigen_val[-1])
        for i in range(self.point_num):
            qc.crz(gamma * self.lamda * self.deg_matrix[i], eigen_val[-1], qram[i])
        qc.x(eigen_val[-1])

        return qc

    def mixing_gate(self, beta: float) -> QuantumCircuit:
        """
        the mixing operator in QAOA
        :param beta: float, the parameter in phase operator
        """
        qram = QuantumRegister(self.point_num)
        qc = QuantumCircuit(qram)

        for i in range(self.point_num):
            qc.rx(beta, qram[i])

        return qc

    def qaoa_simple(self) -> QuantumCircuit:
        """
        the simple version of QAOA
        """
        qram = QuantumRegister(self.point_num)
        cl = ClassicalRegister(self.point_num)
        qc = QuantumCircuit(qram, cl)

        p = len(self.theta) // 2
        gamma = self.theta[:p]
        beta = self.theta[p:]

        qc.h(qram)
        for i in range(p):
            qc.append(self.phase_gate_simple(gamma[i]), qram)
            qc.append(self.mixing_gate(beta[i]), qram)

        qc.measure(qram, cl)

        return qc

    def qaoa(self) -> QuantumCircuit:
        """
        the complete version of QAOA
        """
        qram = QuantumRegister(self.point_num)
        eigen_vec = QuantumRegister(1)
        eigen_val = QuantumRegister(self.precision)
        anc = QuantumRegister(1)
        cl = ClassicalRegister(self.point_num)
        qc = QuantumCircuit(qram, eigen_vec, eigen_val, anc, cl)

        p = len(self.theta) // 2
        gamma = self.theta[:p]
        beta = self.theta[p:]

        qc.h(qram)
        for i in range(p):
            qc.append(self.phase_gate(gamma[i]), [*qram, *eigen_vec, *eigen_val, *anc])
            qc.append(self.mixing_gate(beta[i]), qram)

        qc.measure(qram, cl)

        return qc

    def cal_energy(self, state: str) -> float:
        """
        calculating the total energy of the quantum system represented by measurement results
        :param state: the measurement results of the quantum system
        :return: the total energy
        """
        energy = 0
        for i in range(self.point_num):
            for j in range(i + 1, self.point_num):
                if state[i] != state[j]:
                    energy += 0.5 * self.adj_matrix[i][j]

        # constraint
        constraint = 0.
        for k in range(self.point_num):
            if state[k] == '0':
                constraint += self.deg_matrix[k]
            else:
                constraint -= self.deg_matrix[k]
        energy += self.lamda * abs(constraint)

        return energy

    def expectation_value(self, qc: QuantumCircuit, shots: int = 20000) -> float:
        """
        calculating the expectation value of the entire quantum system
        :param qc: quantum circuit
        :param shots: int, the shots times
        """
        job = execute.exec_qcircuit(qc, shots, 'sim', False, None, False)
        output = execute.get_output(job, 'sim')

        energy = 0
        for item in output.items():
            tmp_key = item[0][::-1]
            tmp_val = item[1]
            # 此处在真机上处理时，不需要除以shots，得到的本身就是概率
            energy += self.cal_energy(tmp_key) * (tmp_val / shots)
        energy = round(energy, 4)
        return energy

    def gradient_descent(self):
        """
        optimizing the parameters of QAOA
        """
        energies = []
        for i in range(len(self.theta)):
            self.theta[i] += self.epsilon
            qc = self.qaoa()
            energies.append(self.expectation_value(qc))
            self.theta[i] -= 2 * self.epsilon
            qc = self.qaoa()
            energies.append(self.expectation_value(qc))
            self.theta[i] += self.epsilon
        for i in range(len(self.theta)):
            self.theta[i] -= (energies[i * 2] - energies[i * 2 + 1]) / (2.0 * self.epsilon) * self.step

    def main(self):
        """
        the controller that handles the entire process of QNCut
        """
        energy = 100
        energy_old = 1000
        energy_min = 1000

        s = 0
        while s < 100 and abs(energy - energy_old) > self.delta:
            if s > 0:
                # optimize theta
                self.gradient_descent()
            qc = self.qaoa()

            energy_old = energy
            energy = self.expectation_value(qc)
            if energy < energy_min:
                energy_min = energy
                self.min_theta = self.theta
            s += 1
            print(s, '-th step, F: ', energy, ' theta: ', self.theta)

            # reduce the step size
            if self.step > 0.001:
                self.step *= 0.9


def execute_qncut(points, theta, lamda, norm_threshold, env, backend, print_detail):
    cut = QAOACut(points, theta, lamda, norm_threshold)
    cut.main()

    # getting the optimal outputs
    result_cut = QAOACut(points, cut.min_theta, lamda, norm_threshold)
    qc = result_cut.qaoa()
    job = execute.exec_qcircuit(qc, 20000, env, False, backend, print_detail)
    output = execute.get_output(job, env)

    # finding the output with maximum probability
    max_num = 0
    max_output = ""
    for item in output.items():
        if item[1] > max_num:
            max_num = item[1]
            max_output = item[0]

    if print_detail:
        print("max_output: ", max_output)

    clusters = [[], []]
    for i in range(len(max_output)):
        if max_output[i] == '0':
            clusters[0].append(points[i])
        else:
            clusters[1].append(points[i])

    return [SingleCluster(None, clusters[i], env, backend, print_detail) for i in range(len(clusters))]


def random_theta(theta_num=4):
    theta = list()
    for _ in range(theta_num):
        theta.append(2 * m.pi * random.random())
    return theta


def divide_clusters(points, env, backend, print_detail, cluster_max_size, lamda=6, norm_threshold=0.25):
    clusters = execute_qncut(points, random_theta(), lamda, norm_threshold, env, backend, print_detail)

    i = 0
    while i < len(clusters):
        if clusters[i].element_num <= cluster_max_size:
            i += 1
        else:
            if print_detail:
                print(f"The {i}-th cluster needs to be partitioned again")
            clusters[i: i + 1] = execute_qncut(clusters[i].elements, random_theta(), lamda, norm_threshold, env,
                                               backend, print_detail)

    # calculating the centroid of each cluster
    for cluster in clusters:
        cluster.calculate_centroid()

    if print_detail:
        print("The final result of QNCut: ")
        for i in range(len(clusters)):
            print(f"{i}-th final cluster: centroid: {clusters[i].centroid}")

    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum Normalized Cut')
    parser.add_argument('--file_name', '-f', type=str, default='ulysses16.tsp', help='Dataset of TSP')
    parser.add_argument('--scale', '-s', type=int, default=16, help='The scale of dataset')
    parser.add_argument('--lamda', '-l', type=float, default=6, help='The parameter in QNCut\'s cost function')
    parser.add_argument('--norm_threshold', '-n', type=float, default=0.25,
                        help='the threshold in degree matrix normalization')

    args = parser.parse_args()

    lines = read_dataset.read_dataset(args.file_name, args.scale)
    points = []
    for line in lines:
        point = line.strip().split(' ')[1:]
        points.append([float(point[i]) for i in np.arange(len(point))])

    theta = list()
    for _ in range(4):
        theta.append(2 * m.pi * random.random())
    # lamda = 6
    # max_sum = 0.25

    cut = QAOACut(points, theta, args.lamda, args.norm_threshold)
    cut.main()
    print("min_theta: ", cut.min_theta)
    print("theta: ", cut.theta)
