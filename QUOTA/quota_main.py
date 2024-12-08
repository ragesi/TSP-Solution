# -*- coding: UTF-8 -*-
import random
import time
import argparse
import sys
import os
from typing import Optional

import numpy as np
import math as m
import cmath as cm

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import noise, AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit.providers.fake_provider import Fake27QPulseV1, Fake127QPulseV1, GenericBackendV2
import qiskit.circuit.library as lib

dir_path = os.path.dirname(os.path.realpath(__file__))
root_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, root_dir_path)
from utils import NOT_gate, util, execute
from dataset import test
from QUOTA import quota_util


class OptimalPath:
    def __init__(self, point_num: int, points: list, cycle_type: bool, precision: int, env: str, backend: Optional[str],
                 noisy: bool, print_detail: bool):
        """
        :param point_num: int, the number of cities in the route
        :param points: list, coordinates of all cities
        :param cycle_type: bool, True if QUOTA finds the optimal cycle and False otherwise
        :param precision: int, the precision parameter for calculating and storing the cost of candidates in circuit
        :param env: string, the environment type for implementing the circuit
        :param backend: string, the backend name when running the circuit on real quantum devices
        :param noisy: boolean, whether to add noise to circuit on simulators
        :param print_detail, boolean, whether to print the execution detail
        """
        # determining whether the input is legal
        quota_util.validate_inputs(point_num, points)

        # transforming the input points into the type of cycle
        self.cycle_type = cycle_type
        if self.cycle_type:
            points = quota_util.trans_to_cycle(points)

        self.point_num = point_num
        # there is only one Hamiltonian solution in solution space
        if self.point_num < 4:
            return

        self.points = points
        self.precision = precision
        # the parameters involve the composition of candidate solutions
        self.choice_num, self.step_num = 0, 0
        # the optimal solutions
        self.path = []

        # the number of qubits used to encode all choices in each step
        self.choice_bit_num = 0
        # the number of qubits in all quantum registers
        self.qram_num, self.buffer_num, self.anc_num, self.res_num, self.total_qubit_num = 0, 0, 0, 0, 0

        # the distance adjacency
        self.dist_adj, self.end_dists = None, None

        # the parameters in QUOTA
        self.threshold = 0.0
        self.grover_iter_min_num, self.grover_iter_max_num = 1.0, 1.0
        self.alpha = 6.0 / 5
        self.grover_repeat_num = 0

        # initializing the execution of circuit
        self.env = env
        self.backend = backend
        self.noisy = noisy
        self.job = None

        # fixed circuit
        self.qc_start, self.qc_end = None, None

        self.print_detail = print_detail

        # initialization
        self.initialize_all()

    def initialize_all(self):
        """
        executing all initialization steps
        """
        # initializing candidate solutions
        self.init_candidate_sol()

        # calculating the qubit cost in all quantum registers
        self.init_circuit()

        # normalizing the cost adjacency matrix
        self.init_dist_adj()

        # initializing the parameters in QUOTA
        self.init_param()

        # building up the fixed partial circuit
        self.init_fixed_circuit()

    def init_candidate_sol(self):
        """
        initializing candidate solutions
        """
        # the number of choices in each step, excluding the start and the end
        self.choice_num = self.point_num - 2
        # the number of steps
        self.step_num = self.point_num - 2
        # the optimal solutions
        self.path = [i for i in np.arange(self.step_num)]

    def init_circuit(self):
        """
        setting the number of qubits in all quantum registers
        """
        # the number of qubits used to encode all choices in each step
        self.choice_bit_num = m.ceil(1.0 * m.log2(self.choice_num))

        self.qram_num = self.step_num * self.choice_bit_num
        self.buffer_num = max(self.precision, self.step_num)
        self.anc_num = max(max(self.precision - 1, self.step_num - 2), 3)
        self.res_num = 3
        self.total_qubit_num = self.qram_num + self.buffer_num + self.anc_num + self.res_num

        if self.print_detail:
            print("qram_num: ", self.qram_num)
            print("buffer_num: ", self.buffer_num)
            print("anc_num: ", self.anc_num)
            print("res_num: ", self.res_num)
            print("total_qubit_num: ", self.total_qubit_num)

    def init_dist_adj(self):
        """
        the cost matrix's preprocessing
        """
        dist_adj = np.zeros((self.point_num - 1, self.point_num - 1))
        max_dist = 0
        for i in range(self.point_num - 1):
            for j in range(i, self.point_num - 1):
                if i == 0 and j == self.point_num - 2:
                    continue
                dist_adj[i][j] = abs(self.points[j + 1][0] - self.points[i][0]) + abs(
                    self.points[j + 1][1] - self.points[i][1])
                max_dist = max(max_dist, dist_adj[i][j])
        # max_dist *= 1.2

        order_dists = []
        for i in range(self.point_num - 1):
            for j in range(i, self.point_num - 1):
                if i == 0 and j == self.point_num - 2:
                    continue
                dist_adj[i][j] = max_dist - dist_adj[i][j]
                if i > 0 and j < self.point_num - 2:
                    dist_adj[j + 1][i - 1] = dist_adj[i][j]
                    order_dists.append(dist_adj[i][j])

        # calculate the max distance
        order_dists.sort(reverse=True)
        max_dist = max(dist_adj[0][:-1])
        for i in range(self.step_num - 1):
            max_dist += order_dists[i]
        max_dist += max([row[-1] for row in dist_adj]) + 1

        # normalize the adjacency matrix to 2.0 * m.pi / (2 ** precision)
        base_num = 2 ** self.precision
        for i in np.arange(len(dist_adj)):
            for j in np.arange(len(dist_adj[i])):
                dist_adj[i][j] = round(1.0 * dist_adj[i][j] / max_dist * base_num) / base_num

        # make up the last step's distance
        self.end_dists = np.zeros(self.choice_num)
        for i in np.arange(len(dist_adj)):
            if i == 0:
                continue
            self.end_dists[i - 1] = dist_adj[i][-1]
        # make up the rest of adjacency matrix
        self.dist_adj = dist_adj[:, :self.choice_num]

    def init_param(self):
        """
        initializing the parameters in QUOTA
        """
        # initializing the threshold
        for i in np.arange(len(self.dist_adj) - 1):
            self.threshold += self.dist_adj[i][i]
        self.threshold += self.end_dists[self.choice_num - 1]
        self.threshold = min(self.threshold, 1.0 - (1.0 / 64))
        self.threshold *= 2 ** self.precision
        if self.print_detail:
            print("threshold: ", self.threshold)

        # setting the number of iterations for VPS and MCPS oracles
        for i in np.arange(self.step_num, 0, -1):
            self.grover_iter_min_num *= m.sqrt(1.0 * i / (2 ** self.choice_bit_num))
        self.grover_iter_min_num = m.pi / 4.0 / m.asin(self.grover_iter_min_num)
        self.grover_iter_min_num = max(1.0, self.grover_iter_min_num)
        self.grover_iter_max_num = self.grover_iter_min_num
        if self.print_detail:
            print("iter_num: ", self.grover_iter_min_num)

        # initializing the number of repetitions of quantum circuit
        self.grover_repeat_num = round(m.log(m.sqrt(m.factorial(self.choice_num)), self.alpha))

    def init_fixed_circuit(self):
        """
        in order to reduce the entire implementation time, we build up the fixed partial circuit first,
        after getting the new threshold, we complete the entire circuit
        """
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.buffer_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(self.res_num)
        self.qc_start = QuantumCircuit(qram, buffer, anc, res, name='qc_start')
        self.qc_end = QuantumCircuit(qram, buffer, anc, res, name='qc_end')

        self.qc_start.append(self.check_route_validity(), [*qram, *buffer[:self.step_num], *anc, res[0]])
        self.qc_start.append(self.cal_path_dist(), [*qram, *buffer[:self.precision], *anc])

        self.qc_end.append(self.cal_path_dist().inverse(), [*qram, *buffer[:self.precision], *anc])
        self.qc_end.append(self.check_route_validity().inverse(), [*qram, *buffer[:self.step_num], *anc, res[0]])
        self.qc_end.append(self.grover_diffusion(), [*qram, *anc, res[-1]])

    def check_route_validity(self) -> QuantumCircuit:
        """
        VPS Oracle
        """
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.step_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(1)
        qc = QuantumCircuit(qram, buffer, anc, res, name='check_route_validity')

        for i in np.arange(self.step_num):
            for j in np.arange(self.choice_num):
                qc.append(NOT_gate.equal_to_int_NOT(j, self.choice_bit_num, self.anc_num),
                          [*qram[i * self.choice_bit_num: (i + 1) * self.choice_bit_num], *anc, buffer[j]])
        qc.append(NOT_gate.custom_mcx(self.step_num, self.anc_num), [*buffer, *anc, *res])
        for i in np.arange(self.step_num - 1, -1, -1):
            for j in np.arange(self.choice_num - 1, -1, -1):
                qc.append(NOT_gate.equal_to_int_NOT(j, self.choice_bit_num, self.anc_num).inverse(),
                          [*qram[i * self.choice_bit_num: (i + 1) * self.choice_bit_num], *anc, buffer[j]])

        return qc

    def qpe_u(self, dists: np.ndarray) -> QuantumCircuit:
        """
        the control-U operator in QPE
        :param dists: np.ndarray, a row or a column of the cost matrix
        """
        control = QuantumRegister(self.precision)
        target = QuantumRegister(self.choice_bit_num)
        anc = QuantumRegister(self.anc_num)
        qc = QuantumCircuit(control, target, anc)
        assert np.all(np.isfinite(dists)), "Input dists contains invalid values (NaN or Inf)."

        for i in np.arange(len(dists)):
            qc.append(NOT_gate.equal_to_int_NOT(i, self.choice_bit_num, self.anc_num - 1),
                      [*target, *anc[1:], anc[0]])

            for j in np.arange(self.precision):
                for _ in np.arange(2 ** (self.precision - j - 1)):
                    qc.cp(2.0 * m.pi * dists[i], control[j], anc[0])

            qc.append(NOT_gate.equal_to_int_NOT(i, self.choice_bit_num, self.anc_num - 1).inverse(),
                      [*target, *anc[1:], anc[0]])

        return qc

    def custom_qpe_u(self, dist_adj: np.ndarray) -> QuantumCircuit:
        """
        the control-F-U operator in QPE
        :param dist_adj: np.ndarray, the cost matrix
        """
        control = QuantumRegister(self.precision)
        source = QuantumRegister(self.choice_bit_num)
        target = QuantumRegister(self.choice_bit_num)
        anc = QuantumRegister(self.anc_num)
        qc = QuantumCircuit(control, source, target, anc)

        for i in np.arange(len(dist_adj)):
            qc.append(NOT_gate.equal_to_int_NOT(i, self.choice_bit_num, self.anc_num - 2),
                      [*source, *anc[2:], anc[0]])

            for j in np.arange(len(dist_adj[i])):
                qc.append(NOT_gate.equal_to_int_NOT(j, self.choice_bit_num, self.anc_num - 2),
                          [*target, *anc[2:], anc[1]])

                for k in np.arange(self.precision):
                    qc.ccx(anc[0], control[k], anc[2])
                    for _ in np.arange(2 ** (self.precision - k - 1)):
                        qc.cp(2.0 * m.pi * dist_adj[i][j], anc[2], anc[1])
                    qc.ccx(anc[0], control[k], anc[2])

                qc.append(NOT_gate.equal_to_int_NOT(j, self.choice_bit_num, self.anc_num - 2).inverse(),
                          [*target, *anc[2:], anc[1]])

            qc.append(NOT_gate.equal_to_int_NOT(i, self.choice_bit_num, self.anc_num - 2).inverse(),
                      [*source, *anc[2:], anc[0]])

        return qc

    def grover_diffusion(self) -> QuantumCircuit:
        """
        the Diffusion operator of Grover algorithm
        """
        qram = QuantumRegister(self.qram_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(1)
        qc = QuantumCircuit(qram, anc, res, name='grover_diffusion')

        qc.h(qram)
        qc.x(qram)
        qc.append(NOT_gate.custom_mcx(self.qram_num, self.anc_num), [*qram, *anc, *res])
        qc.x(qram)
        qc.h(qram)

        return qc

    def cal_path_dist(self) -> QuantumCircuit:
        """
        calculating the total cost of each route
        """
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.precision)
        anc = QuantumRegister(self.anc_num)
        qc = QuantumCircuit(qram, buffer, anc, name='cal_distance_qpe')

        # QPE
        qc.h(buffer)
        # for each step
        for i in np.arange(self.step_num):
            if i == 0:
                qc.append(self.qpe_u(self.dist_adj[0]), [*buffer, *qram[:self.choice_bit_num], *anc])
            else:
                qc.append(self.custom_qpe_u(self.dist_adj[1:]),
                          [*buffer, *qram[(i - 1) * self.choice_bit_num: (i + 1) * self.choice_bit_num], *anc])
        # the last step which is not shown in the self.step_num
        qc.append(self.qpe_u(self.end_dists), [*buffer, *qram[-self.choice_bit_num:], *anc])
        qc.append(lib.QFT(self.precision, do_swaps=False, inverse=True), buffer)

        return qc

    def cal_single_route_dist(self, route: list) -> float:
        """
        calculating the cost of the solution that is obtained by quantum circuit
        :param route: list(), the solution obtained by quantum circuit
        """
        dist = 0.0
        for i in np.arange(len(route)):
            if route[i] >= self.choice_num or (i > 0 and route[i - 1] >= self.choice_num):
                continue
            elif i == 0:
                dist += self.dist_adj[0][route[i]]
            else:
                dist += self.dist_adj[route[i - 1] + 1][route[i]]
        dist += self.end_dists[route[-1]] if route[-1] < self.choice_num else 0
        dist *= 2 ** self.precision

        return dist

    def translate_route(self, bin_route: str) -> list:
        """
        transforming the binary result obtained by quantum circuit to solution
        :param bin_route: string, the binary result
        """
        route = []
        for i in np.arange(self.step_num):
            route.insert(0, int(bin_route[i * self.choice_bit_num: (i + 1) * self.choice_bit_num], 2))

        return route

    def async_grover(self):
        """
        executing the quantum circuit and iteratively obtaining the optimal solution
        """
        qram = QuantumRegister(self.qram_num)
        buffer = QuantumRegister(self.buffer_num)
        anc = QuantumRegister(self.anc_num)
        res = QuantumRegister(self.res_num)
        cl = ClassicalRegister(self.qram_num)
        qc = QuantumCircuit(qram, buffer, anc, res, cl)
        # initialization
        qc.h(qram)
        qc.x(res[-1])
        qc.h(res[-1])

        max_iter_bound = m.pi / 4.0 * m.sqrt(2 ** self.qram_num)

        # qc_start, qc_end = self.build_partial_circuit()
        # print(qc_start)
        # print(qc_end)

        if self.job is not None:
            output = execute.get_output(self.job, self.env)
            if self.env == 'sim':
                output = sorted(output.items(), key=lambda item: item[1], reverse=True)[0][0]
            else:
                output = sorted(output.items(), key=lambda item: item[1], reverse=True)
                output = util.int_to_binary(output[0][0], self.qram_num)
            new_path = self.translate_route(output)
            new_threshold = self.cal_single_route_dist(new_path)
            if self.print_detail:
                print("new_path: ", new_path)

            if new_threshold > self.threshold:
                self.threshold = new_threshold
                self.path = new_path
                # self.grover_iter_min_num = min(self.grover_iter_min_num * 1.4, max_iter_bound)
                self.grover_iter_min_num = 1.0 / 2 * (self.grover_iter_max_num + self.grover_iter_min_num)
                # self.grover_iter_max_num = max(self.grover_iter_min_num, self.grover_iter_max_num)
                # self.grover_repeat_num = round(m.log(m.sqrt(m.factorial(self.choice_num)), self.alpha))
                self.grover_repeat_num = round(m.log(m.sqrt(2 ** self.qram_num) / self.grover_iter_max_num, self.alpha))
                if self.print_detail:
                    print("new_threshold: ", new_threshold)
                    print("grover repeat num: ", self.grover_repeat_num)
            else:
                self.grover_repeat_num -= 1
                self.grover_iter_max_num = min(self.alpha * self.grover_iter_max_num, max_iter_bound)
        if self.grover_repeat_num == 0:
            # self.session.close()
            return

        cur_iter_num = random.randint(int(self.grover_iter_min_num), int(self.grover_iter_max_num))
        # print("cur_iter_num: ", cur_iter_num)
        for _ in range(cur_iter_num):
            qc.append(self.qc_start, [i for i in range(self.total_qubit_num)])
            qc.append(lib.IntegerComparator(self.precision, self.threshold, geq=True),
                      [*buffer[:self.precision], res[1], *anc[:self.precision - 1]])
            qc.ccx(res[0], res[1], res[-1])
            qc.append(lib.IntegerComparator(self.precision, self.threshold, geq=True).inverse(),
                      [*buffer[:self.precision], res[1], *anc[:self.precision - 1]])
            qc.append(self.qc_end, [i for i in range(self.total_qubit_num)])

        qc.measure(qram, cl)

        # remote_backend: 32, 63
        shots = 2000
        self.job = execute.exec_qcircuit(qc, shots, self.env, self.noisy, self.backend)

        self.async_grover()

    def main(self) -> list:
        """
        the controller handling the entire process of QUOTA
        :return: the optimal solution
        """
        if self.point_num < 4:
            return [i for i in range(self.point_num)]

        self.async_grover()
        path = [0]
        for i in range(len(self.path)):
            path.append(self.path[i] + 1)
        if self.cycle_type:
            path.append(0)
        else:
            path.append(len(path))
        return path

        # path = [self.points[0]]
        # for i in range(len(self.path)):
        #     path.append(self.points[self.path[i] + 1])
        # path.append(self.points[-1])
        # return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize path')
    parser.add_argument('--scale', '-s', type=int, default=3, help='The number of nodes in TSP graph')
    parser.add_argument('--cycle', '-c', type=bool, default=True, help='let quota find the optimal cycle or path')
    parser.add_argument('--precision', '-p', type=int, default=6, help='The precision of the QPE process')
    parser.add_argument('--env', '-e', type=str, default='sim',
                        help='The environment to run program, parameter: "sim"; "remote_sim"; "real"')
    parser.add_argument('--backend', '-b', type=str, default=None, help='The backend to run program')
    parser.add_argument('--noisy', '-n', type=bool, default=False, help='determining whether to add noisy')
    parser.add_argument('--print_detail', '-pd', type=bool, default=True, help='Print detailed information')

    args = parser.parse_args()
    if args.scale < 3 or args.scale > 7:
        raise ValueError('The scale must be between 3 and 7')
    if args.env != 'sim' and args.env != 'remote_sim' and args.env != 'real':
        raise ValueError('The environment must be either "sim" or "remote_sim" or "real"!')
    if args.env == 'remote_sim' and args.backend != 'ibmq_qasm_simulator' and args.backend != 'simulator_extended_stabilizer':
        raise ValueError('The backend is illegal for remote simulator!')
    if args.env == 'real' and args.backend != 'ibm_brisbane' and args.backend != 'ibm_osaka' and args.backend != 'ibm_kyoto':
        raise ValueError('The backend is illegal for a real quantum computer!')

    test_points_dict = {
        3: test.cycle_test_for_3,
        4: test.cycle_test_for_4,
        5: test.cycle_test_for_5,
        6: test.cycle_test_for_6,
        7: test.cycle_test_for_7
    }
    test_points = test_points_dict[args.scale]
    test = OptimalPath(args.scale + 1, test_points, args.cycle, args.precision, args.env, args.backend, args.noisy,
                       args.print_detail)
    # print(test.dist_adj)
    # print(test.end_dists)

    start_time = time.time()
    path = test.main()
    end_time = time.time()
    print("time: ", end_time - start_time)
    print(path)
