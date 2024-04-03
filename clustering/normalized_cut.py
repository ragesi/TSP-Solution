from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator

import random
import math as m

import clustering.cut_preparation as prep
from utils import execute, read_dataset


class QAOACut:
    def __init__(self, points, theta, lamda, precision):
        self.points = points
        self.point_num = len(self.points)
        self.adj_matrix = prep.build_adj_matrix(self.points, self.point_num)
        deg_matrix = prep.build_deg_matrix(self.adj_matrix, self.point_num)
        self.deg_matrix = prep.amplify_deg_matrix(deg_matrix, self.point_num)
        self.norm_deg_matrix = prep.normalize_deg_matrix(deg_matrix, self.point_num)

        self.theta = theta
        self.lamda = lamda
        self.precision = precision
        self.min_theta = self.theta

        self.step = 0.01
        self.epsilon = 0.001
        self.delta = 0.005

    def phase_gate(self, gamma):
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

        # The constraint of cut
        qc.append(prep.qpe(self.precision, self.norm_deg_matrix, self.point_num), [*qram, *eigen_vec, *eigen_val, *anc])

        # If the highest qubit of eigen_val is 1, the result is negative
        for i in range(self.point_num):
            qc.crz(gamma * self.lamda * self.deg_matrix[i], eigen_val[-1], qram[i])
        # Else if the highest qubit is 0, the result is positive
        qc.x(eigen_val[-1])
        for i in range(self.point_num):
            qc.crz(-1 * gamma * self.lamda * self.deg_matrix[i], eigen_val[-1], qram[i])
        qc.x(eigen_val[-1])

        return qc

    def mixing_gate(self, beta):
        qram = QuantumRegister(self.point_num)
        qc = QuantumCircuit(qram)

        for i in range(self.point_num):
            qc.rx(beta, qram[i])

        return qc

    def qaoa(self):
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

    def cal_energy(self, state):
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

    def expectation_vale(self, qc, shots=20000):
        job = execute.exec_qcircuit(qc, shots, 'sim', False, None, False)
        output = execute.get_output(job, 'sim')

        energy = 0
        for item in output.items():
            tmp_key = item[0][::-1]
            tmp_val = item[1]
            # TODO: 此处在真机上处理时，不需要除以shots，得到的本身就是概率
            energy += self.cal_energy(tmp_key) * (tmp_val / shots)
        energy = round(energy, 4)
        return energy

    def gradient_descent(self):
        energies = []
        for i in range(len(self.theta)):
            self.theta[i] += self.epsilon
            qc = self.qaoa()
            energies.append(self.expectation_vale(qc))
            self.theta[i] -= 2 * self.epsilon
            qc = self.qaoa()
            energies.append(self.expectation_vale(qc))
            self.theta[i] += self.epsilon
        for i in range(len(self.theta)):
            self.theta[i] -= (energies[i * 2] - energies[i * 2 + 1]) / (2.0 * self.epsilon) * self.step

    def main(self):
        energy = 100
        energy_old = 1000
        energy_min = 1000

        i = 0
        s = 0
        while i < 100 and abs(energy - energy_old) > self.delta:
            qc = self.qaoa()

            energy_old = energy
            energy = self.expectation_vale(qc)
            if energy < energy_min:
                energy_min = energy
                self.min_theta = self.theta
            s += 1
            print(s, '-th step, F: ', energy, ' theta: ', self.theta)

            # optimize theta
            self.gradient_descent()

            # reduce the step size
            if self.step > 0.001:
                self.step *= 0.9
            else:
                i += 1


if __name__ == '__main__':
    lines = read_dataset.read_dataset('ulysses16.tsp', 16)
    points = list()
    for line in lines:
        tmp_point = line.strip().split(' ')
        tmp_point = [float(x) for x in tmp_point]
        tmp_point[0] = int(tmp_point[0])
        points.append([tmp_point[1], tmp_point[2]])

    theta = list()
    for _ in range(4):
        theta.append(2 * m.pi * random.random())

    test = QAOACut(points, theta)
