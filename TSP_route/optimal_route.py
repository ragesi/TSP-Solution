# -*- coding: UTF-8 -*-

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

import numpy as np
import math as m

from utils import display_result, NOT_gate, unitary_function, util


class OptimalRoute:
    def __init__(self, node_num, node_list, dis_adj):
        """
        :param node_num: the number of cities in the route
        :param node_list: the list of cities' id,
                          the first element must be the start of route, and the last element must be the end
        :param dis_adj: the distance adjacency matrix of nodes, whose size is [node_num - 1, node_num - 1]
        """
        # TODO: 处理node_list的映射关系
        # the number of nodes in the route
        self.node_num = node_num
        self.node_list = node_list
        # self.dis_adj's size is [node_num - 1, 4], filling 0 to make-up, which excludes the last step's distance
        self.dis_adj = np.zeros((self.node_num - 1, 4))
        self.end_dis = np.zeros(4)
        # the number of choices for every step, excluding the start and the end
        self.choice_num = self.node_num - 2
        # the number of steps that have choices
        self.step_num = self.node_num - 2
        # the number of choice encoding bits in binary
        self.choice_code_num = m.ceil(1.0 * m.sqrt(self.choice_num))
        # the precision of route distance
        self.precision = 6
        self.thresholds = []
        self.illegal_choice_list = []
        self.illegal_route_list = []

        self.q_choice = None
        self.q_detail = None
        self.q_ancilla = None
        self.q_result = None
        self.c = None
        self.qc = None

        self.init_dis_adj(dis_adj)
        self.init_Circuit()
        # get the parameter of algorithm, which is used in each iteration
        self.get_illegal_choice_param()
        self.get_illegal_route_param()

    def init_dis_adj(self, dis_adj):
        """
        the distance matrix's preprocessing
        :param dis_adj: matrix representing the distance between every city
        """
        # calculate the total distance
        total_distance = 0.0
        for i in np.arange(len(dis_adj)):
            for j in np.arange(i, len(dis_adj[i])):
                total_distance += dis_adj[i][j]
        # normalize the adjacency matrix to 2.0 * m.pi / 64
        for i in np.arange(len(dis_adj)):
            for j in np.arange(len(dis_adj[i])):
                dis_adj[i][j] = round(1.0 * dis_adj[i][j] / total_distance * 60)
                dis_adj[i][j] = 2.0 * m.pi * dis_adj[i][j] / 64

        # calculating the threshold
        threshold = 1.0 / (2 ** self.precision)
        for i in np.arange(len(dis_adj)):
            threshold += (dis_adj[i][i] / 2.0 / m.pi)
        threshold = util.decimal_to_binary(threshold, self.precision)
        self.thresholds.append(threshold)

        # make up the last step's distance
        for i in np.arange(len(dis_adj)):
            if i == 0:
                dis_adj[i].pop()
                continue
            self.end_dis[i - 1] = dis_adj[i].pop()
        # make up the rest of adjacency matrix
        self.dis_adj[:, :self.node_num - 2] = dis_adj

    def init_Circuit(self):
        """
        init the quantum circuit for detecting the legal route, which has 4 parts
        1. choice register: store the choice of every step
        2. detail register: store the temporary info of processing
        3. ancilla register: to assist the process of algorithm
        4. result register: to record the result of every stage of algorithm
        """
        total_num = 27
        # for the end node, it doesn't have choice; for the node_list[-2], it only has one choice for the end
        choice_num = self.step_num * self.choice_code_num
        detail_num = 6
        result_num = 4
        ancilla_num = total_num - choice_num - detail_num - result_num
        print("choice_num: ", choice_num)
        print("detail_num: ", detail_num)
        print("ancilla_num: ", ancilla_num)
        print("result_num: ", result_num)

        self.q_choice = QuantumRegister(choice_num, name='choice')
        self.q_detail = QuantumRegister(detail_num, name='detail')
        self.q_ancilla = QuantumRegister(ancilla_num, name='ancilla')
        self.q_result = QuantumRegister(result_num, name='result')
        self.c = ClassicalRegister(choice_num + 1, name='c')
        self.qc = QuantumCircuit(self.q_choice, self.q_detail, self.q_ancilla, self.q_result, self.c)

        for i in np.arange(choice_num):
            self.qc.h(self.q_choice[i])
        self.qc.x(self.q_result[3])
        self.qc.h(self.q_result[3])

    def get_illegal_choice_param(self):
        """
        make up the list of illegal choice parameter, which is used in the process of is_legal_choice
        """
        size = 0
        for i in np.arange(2 ** self.choice_code_num - 1, self.choice_num - 1, -1):
            for j in np.arange(self.step_num):
                self.illegal_choice_list.append(
                    [self.q_choice[j * self.choice_code_num: (j + 1) * self.choice_code_num], i, self.q_detail[size]])
                size += 1

    def get_illegal_route_param(self):
        """
        make up the list of illegal route parameter, which is used in the process of is_legal_route
        """
        size = 0
        for i in np.arange(self.step_num):
            for j in np.arange(i + 1, self.step_num):
                self.illegal_route_list.append([self.q_choice[i * self.choice_code_num: (i + 1) * self.choice_code_num],
                                                self.q_choice[j * self.choice_code_num: (j + 1) * self.choice_code_num],
                                                self.q_detail[size]])
                size += 1

    def is_legal_choice(self, dgr):
        """
        determine whether each step's choice exceeds the range of choice_list
        :param dgr: determine whether it is the function's inverse operation or not
        """
        size = len(self.illegal_choice_list)
        if dgr:
            self.illegal_choice_list.reverse()
        for i in np.arange(size):
            NOT_gate.equal_to_int_NOT(self.qc, self.illegal_choice_list[i][0], self.illegal_choice_list[i][1],
                                      self.illegal_choice_list[i][2], self.q_ancilla, self.choice_code_num)

        NOT_gate.zero_NOT(self.qc, self.q_detail[0: size], self.q_result[0], self.q_ancilla, size)

        for i in np.arange(size - 1, -1, -1):
            NOT_gate.equal_to_int_NOT(self.qc, self.illegal_choice_list[i][0], self.illegal_choice_list[i][1],
                                      self.illegal_choice_list[i][2], self.q_ancilla, self.choice_code_num)
        if dgr:
            self.illegal_choice_list.reverse()

    def is_legal_route(self, dgr):
        """
        determine if the choices in the steps are different
        """
        size = len(self.illegal_route_list)
        if dgr:
            self.illegal_route_list.reverse()
        for i in np.arange(size):
            NOT_gate.equal_NOT(self.qc, self.illegal_route_list[i][0], self.illegal_route_list[i][1],
                               self.illegal_route_list[i][2], self.q_ancilla, self.choice_code_num)

        NOT_gate.zero_NOT(self.qc, self.q_detail[0: size], self.q_result[1], self.q_ancilla, size)

        for i in np.arange(size - 1, -1, -1):
            NOT_gate.equal_NOT(self.qc, self.illegal_route_list[i][0], self.illegal_route_list[i][1],
                               self.illegal_route_list[i][2], self.q_ancilla, self.choice_code_num)
        if dgr:
            self.illegal_route_list.reverse()

    def cal_distance(self):
        """
        to calculate the total distance of every route
        """
        # QPE
        for i in np.arange(self.precision):
            self.qc.h(self.q_detail[i])
        # for every bit of detail register (precision register)
        for i in np.arange(self.precision):
            for _ in np.arange(2 ** i):
                # one calculation of distance
                for k in np.arange(self.step_num):
                    if k == 0:
                        unitary_function.QPE_U(self.qc, self.q_detail[i], self.q_choice[k * 2: (k + 1) * 2],
                                               self.q_ancilla[0], self.dis_adj[0])
                    else:
                        unitary_function.n_QPE_U(self.qc, self.q_detail[i], self.q_choice[(k - 1) * 2: k * 2],
                                                 self.q_choice[k * 2: (k + 1) * 2], self.q_ancilla, self.dis_adj[1:])

                unitary_function.QPE_U(self.qc, self.q_detail[i], self.q_choice[-2:], self.q_ancilla[0], self.end_dis)

        unitary_function.QFT_dgr(self.qc, self.q_detail[0: self.precision], self.precision)

    def cal_distance_dgr(self):
        """
        the inverse of the calculating distance operator
        """
        unitary_function.QFT(self.qc, self.q_detail[0: self.precision], self.precision)

        for i in np.arange(self.precision - 1, -1, -1):
            for _ in np.arange(2 ** i):
                unitary_function.QPE_U_dgr(self.qc, self.q_detail[i], self.q_choice[-2:], self.q_ancilla[0],
                                           self.end_dis)

                for k in np.arange(self.step_num - 1, -1, -1):
                    if k == 0:
                        unitary_function.QPE_U_dgr(self.qc, self.q_detail[i], self.q_choice[k * 2: (k + 1) * 2],
                                                   self.q_ancilla[0], self.dis_adj[0])
                    else:
                        unitary_function.n_QPE_U_dgr(self.qc, self.q_detail[i], self.q_choice[(k - 1) * 2: k * 2],
                                                     self.q_choice[k * 2: (k + 1) * 2], self.q_ancilla,
                                                     self.dis_adj[1:])

        for i in np.arange(self.precision - 1, -1, -1):
            self.qc.h(self.q_detail[i])

    def is_shortest_route(self, threshold):
        """
        calculate the distance of every route, and determine whether it is less than the threshold
        :param threshold: integer
        """
        self.cal_distance()

        # determine whether the distance of every route is less than threshold
        NOT_gate.less_than_int_NOT(self.qc, self.q_detail, threshold, self.q_result[2], self.q_ancilla, self.precision)

    def is_shortest_route_dgr(self, threshold):
        """
        the inverse of the calculating distance operator
        """
        NOT_gate.less_than_int_NOT_dgr(self.qc, self.q_detail, threshold, self.q_result[2], self.q_ancilla,
                                       self.precision)

        self.cal_distance_dgr()

    def Grover(self, threshold):
        """
        an iteration of Grover algorithm
        """
        # run three judgment processes
        self.is_legal_choice(dgr=False)
        self.is_legal_route(dgr=False)
        self.is_shortest_route(threshold=threshold)

        NOT_gate.n_NOT(self.qc, self.q_result[0: 3], self.q_result[3], self.q_ancilla, 3)

        self.is_shortest_route_dgr(threshold=threshold)
        self.is_legal_route(dgr=True)
        self.is_legal_choice(dgr=True)

        # Grover Diffusion
        unitary_function.Grover_diffusion(self.qc, self.q_choice, self.q_result[3], self.q_ancilla, len(self.q_choice))

    def main(self):
        """
        the entire process of choose optimal route
        """
        # TODO: 此处需要计算阈值，但因为距离数据已经经过了区间映射和归一化，所以不知道阈值设置多少合适，留待以后处理
        iter_times = round(m.sqrt(len(self.q_choice)))
        iter_times = 2
        print("iter_times: ", iter_times)

        for i in np.arange(iter_times):
            self.Grover(threshold=self.thresholds[0])

        for i in np.arange(len(self.q_choice)):
            self.qc.measure(self.q_choice[i], self.c[i])
        self.qc.measure(self.q_result[1], self.c[-1])
        # TODO：此处根据阈值和执行情况设置shots次数
        val = display_result.Measurement(self.qc, return_M=True, print_M=False, shots=100)
        print(len(val))
        print(val)
        num = 0
        max_item = ""
        # all_one_num = 0
        for item in val.items():
            if item[1] > num:
                num = item[1]
                max_item = item[0]
        #     if item[0][-1] == '1':
        #         print(item)
        #         all_one_num += 1
        # print("all_one_num: ", all_one_num)
        print(max_item, num)


if __name__ == '__main__':
    # 2.49次
    # adj = [[1, 2, 3, 4, 0],
    #        [1, 2, 3, 4, 5],
    #        [1, 2, 3, 4, 5],
    #        [1, 2, 3, 4, 5],
    #        [1, 2, 3, 4, 5]]
    # 2.49次
    # adj = [[1, 2, 3, 0],
    #        [1, 2, 3, 4],
    #        [1, 2, 3, 4],
    #        [1, 2, 3, 4]]
    # 2.24次
    # adj = [[1.5, 3.2, 0],
    #        [0, 2.4, 9],
    #        [2.4, 0, 1.2]]
    adj = [[2, 2.1, 0],
           [0, 2.2, 3],
           [2.2, 0, 2.4]]
    # adj = [[2, 2, 0],
    #        [0, 2, 2],
    #        [2, 0, 2]]
    test = OptimalRoute(4, [0, 1, 2, 3], adj)
    test.main()