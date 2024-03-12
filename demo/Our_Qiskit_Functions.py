from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute
from qiskit.extensions.simulator import snapshot
from qiskit.tools.visualization import circuit_drawer
import numpy as np
import math as m
import scipy as sci
import random
import time
import matplotlib
import matplotlib.pyplot as plt

S_simulator = Aer.backends(name='statevector_simulator')[0]
M_simulator = Aer.backends(name='qasm_simulator')[0]


#==================================================
#----------- Displaying Results -----------
#==================================================

def Wavefunction(obj, **kwargs):
    '''
     Prints a tidier versrion of the array statevector corresponding to the wavefuntion of a QuantumCircuit object
     Keyword Arguments: precision (integer) - the decimal precision for amplitudes
                        column (Bool) - prints each state in a vertical column
                        systems (array of integers) - seperates the qubits into different states
                        show_systems (array of Bools) - indictates which qubit systems to print
    '''

    if (type(obj) == QuantumCircuit):
        statevec = execute(obj, S_simulator, shots=1).result().get_statevector()
    if (type(obj) == np.ndarray):
        statevec = obj
    sys = False
    NL = False
    dec = 5
    if 'precision' in kwargs:
        dec = int(kwargs['precision'])
    if 'column' in kwargs:
        NL = kwargs['column']
    if 'systems' in kwargs:
        systems = kwargs['systems']
        sys = True
        last_sys = int(len(systems) - 1)
        show_systems = []
        for s_chk in np.arange(len(systems)):
            if (type(systems[s_chk]) != int):
                raise Exception('systems must be an array of all integers')
        if 'show_systems' in kwargs:
            show_systems = kwargs['show_systems']
            if (len(systems) != len(show_systems)):
                raise Exception('systems and show_systems need to be arrays of equal length')
            for ls in np.arange(len(show_systems)):
                if ((show_systems[ls] != True) and (show_systems[ls] != False)):
                    raise Exception('show_systems must be an array of Truth Values')
                if (show_systems[ls] == True):
                    last_sys = int(ls)
        else:
            for ss in np.arange(len(systems)):
                show_systems.append(True)
    wavefunction = ''
    qubits = int(m.log(len(statevec), 2))
    for i in range(int(len(statevec))):
        # print(wavefunction)
        value = round(statevec[i].real, dec) + round(statevec[i].imag, dec) * 1j
        if ((value.real != 0) or (value.imag != 0)):
            state = list(Binary(int(i), int(2 ** qubits), 'R'))
            state.reverse()
            state_str = ''
            # print(state)
            if (sys == True):  # Systems and SharSystems
                k = 0
                for s in np.arange(len(systems)):
                    if (show_systems[s] == True):
                        if (int(s) != last_sys):
                            state.insert(int(k + systems[s]), '>|')
                            k = int(k + systems[s] + 1)
                        else:
                            k = int(k + systems[s])
                    else:
                        for s2 in np.arange(systems[s]):
                            del state[int(k)]
            for j in np.arange(len(state)):
                if (type(state[j]) != str):
                    state_str = state_str + str(int(state[j]))
                else:
                    state_str = state_str + state[j]
            # print(state_str)
            # print(value)
            if ((value.real != 0) and (value.imag != 0)):
                if (value.imag > 0):
                    wavefunction = wavefunction + str(value.real) + '+' + str(value.imag) + 'j |' + state_str + '>   '
                else:
                    wavefunction = wavefunction + str(value.real) + '' + str(value.imag) + 'j |' + state_str + '>   '
            if ((value.real != 0) and (value.imag == 0)):
                wavefunction = wavefunction + str(value.real) + '  |' + state_str + '>   '
            if ((value.real == 0) and (value.imag != 0)):
                wavefunction = wavefunction + str(value.imag) + 'j |' + state_str + '>   '
            if (NL):
                wavefunction = wavefunction + '\n'
    # print(NL)

    print(wavefunction)


def Measurement(quantumcircuit, **kwargs):
    '''
     Executes a measurement(s) of a QuantumCircuit object for tidier printing
     Keyword Arguments: shots (integer) - number of trials to execute for the measurement(s)
                        return_M (Bool) - indictaes whether to return the Dictionary object containng measurement resul
                        print_M (Bool) - indictaes whether to print the measurement results
                        column (Bool) - prints each state in a vertical column
     '''
    p_M = True
    S = 1
    ret = False
    NL = False
    if 'shots' in kwargs:
        S = int(kwargs['shots'])
    if 'return_M' in kwargs:
        ret = kwargs['return_M']
    if 'print_M' in kwargs:
        p_M = kwargs['print_M']
    if 'column' in kwargs:
        NL = kwargs['column']
    M1 = execute(quantumcircuit, M_simulator, shots=S).result().get_counts(quantumcircuit)
    M2 = {}
    k1 = list(M1.keys())
    v1 = list(M1.values())
    for k in np.arange(len(k1)):
        key_list = list(k1[k])
        new_key = ''
        for j in np.arange(len(key_list)):
            new_key = new_key + key_list[len(key_list) - (j + 1)]
        M2[new_key] = v1[k]
    if (p_M):
        k2 = list(M2.keys())
        v2 = list(M2.values())
        measurements = ''
        for i in np.arange(len(k2)):
            m_str = str(v2[i]) + '|'
            for j in np.arange(len(k2[i])):
                if (k2[i][j] == '0'):
                    m_str = m_str + '0'
                if (k2[i][j] == '1'):
                    m_str = m_str + '1'
                if (k2[i][j] == ' '):
                    m_str = m_str + '>|'
            m_str = m_str + '>   '
            if (NL):
                m_str = m_str + '\n'
            measurements = measurements + m_str
        print(measurements)
        return measurements
    if (ret):
        return M2


def Most_Probable(M,N):
    '''
     Input: M (Dictionary) N (integer)
     Returns the N most probable states accoding to the measurement counts stored in M
    '''
    count = []
    state = []
    if (len(M) < N):
        N = len(M)
    for k in np.arange(N):
        count.append(0)
        state.append(0)
    for m in np.arange(len(M)):
        new = True
        for n in np.arange(N):
            if ((list(M.values())[int(m)] > count[int(n)]) and (new)):
                for i in np.arange(int(N - (n + 1))):
                    count[-int(1 + i)] = count[-int(1 + i + 1)]
                    state[-int(1 + i)] = state[-int(1 + i + 1)]
                count[int(n)] = list(M.values())[m]
                state[int(n)] = list(M.keys())[m]
                new = False
    return count, state


#===============================================
#----------- Math Operations -----------
#===============================================


def Oplus(bit1, bit2):
    '''Adds too bits of O's and 1's (modulo 2)'''
    bit = np.zeros(len(bit1))
    for i in np.arange(len(bit)):
        if ((bit1[i] + bit2[i]) % 2 == 0):
            bit[i] = 0
        else:
            bit[i] = 1
    return bit


def Binary(N, total, LSB):
    """
    Input: N (integer)    total (integer)    LSB (string)
    Returns the base-2 binary equivilant of N according to left or right least significant bit notation
    """
    qubits = int(m.log(total, 2))
    b_num = np.zeros(qubits)
    for i in np.arange(qubits):
        if( N / ((2) ** (qubits - i - 1)) >= 1 ):
            if(LSB == "R"):
                b_num[i] = 1
            if(LSB == "L"):
                b_num[int(qubits - (i + 1))] = 1
            N = N - 2 ** (qubits - i - 1)
    B = []
    for j in np.arange(len(b_num)):
        B.append(int(b_num[j]))
    return B


def From_Binary(S, LSB):
    """
    Input: S (string or array)  LSB (string)
    Converts a base-2 binary number to base-10 according to left or right least significant bit notation
    """
    num = 0
    for i in np.arange(len(S)):
        if(LSB == "R"):
            num = num + int(S[int(0 - (i + 1))]) * 2 ** (i)
        if(LSB == "L"):
            num = num + int(S[int(i)]) * 2 ** (i)
    return num


def B2D(in_bi):
    len_in = len(in_bi)
    in_bi = in_bi[::-1]
    dec = 0
    for i in range(0, len_in):
        if in_bi[i] != '0':
            dec += 2 ** i
    return dec


#=================================================
#----------- Custom Operations -----------
#=================================================


def x_Transformation(qc, qreg, state):
    # Tranforms the state of the system, applying X gates according to as in the vector 'state'
    '''
     Input: qc (QuantumCircuit) qreg (QuantumRegister) state (array)
     Applies the neccessary X gates to transform 'state' to the state of all 1's
    '''
    for j in np.arange(len(state)):
        if (int(state[j]) == 0):
            qc.x(qreg[int(j)])


def n_NOT(qc, control, target, anc):
    # performs an n-NOT gate
    '''
     Input: qc (QuantumCircuit) control (QuantumRegister) target (QuantumRegister[integer]) anc (QuantumRegister)
     Applies the neccessary CCX gates to perform a higher order control-X operation on the target qubit
    '''
    n = len(control)
    instructions = []
    active_ancilla = []
    q_unused = []
    q = 0
    a = 0
    while ((n > 0) or (len(q_unused) != 0) or (len(active_ancilla) != 0)):
        if (n > 0):
            if ((n - 2) >= 0):
                instructions.append([control[q], control[q + 1], anc[a]])
                active_ancilla.append(a)
                a += 1
                q += 2
                n = n - 2
            if ((n - 2) == -1):
                q_unused.append(q)
                n = n - 1
        elif (len(q_unused) != 0):
            if (len(active_ancilla) != 1):
                instructions.append([control[q], anc[active_ancilla[0]], anc[a]])
                del active_ancilla[0]
                del q_unused[0]
                active_ancilla.append(a)
                a = a + 1
            else:
                instructions.append([control[q], anc[active_ancilla[0]], target])
                del active_ancilla[0]
                del q_unused[0]
        elif (len(active_ancilla) != 0):
            if (len(active_ancilla) > 2):
                instructions.append([anc[active_ancilla[0]], anc[active_ancilla[1]], anc[a]])
                active_ancilla.append(a)
                del active_ancilla[0]
                del active_ancilla[0]
                a = a + 1
            elif (len(active_ancilla) == 2):
                instructions.append([anc[active_ancilla[0]], anc[active_ancilla[1]], target])
                del active_ancilla[0]
                del active_ancilla[0]
    for i in np.arange(len(instructions)):
        qc.ccx(instructions[i][0], instructions[i][1], instructions[i][2])
    del instructions[-1]
    for i in np.arange(len(instructions)):
        qc.ccx(instructions[0 - (i + 1)][0], instructions[0 - (i + 1)][1], instructions[0 - (i + 1)][2])


def Control_Instruction(qc, vec):
    # Ammends the proper quantum circuit instruction based on the input 'vec'
    # Used for the function 'n_Control_U
    if (vec[0] == 'X'):
        qc.cx(ver[1], vec[2])
    if (vec[0] == 'Z'):
        qc.cz(ver[1], vec[2])
    if (vec[0] == 'PRASE'):
        qc.cu1(vec[2], vec[1], vec[3])
    if (vec[0] == 'SWAP'):
        qc.cswap(vec[1], vec[2], vec[3])


def X_Transformation(qc, qreg, state):
    for j in np.arange(len(state)):
        if (int(state[j]) == 0):
            qc.x(qreg[int(j)])


def sinmons_solver(E, N):
    '''Returns an array of s_prime candidates
    '''
    s_primes = []
    for s in np.ararge(1, 2 ** N):
        sp = Binary(int(s), 2 ** N)
        candidate = True
        for e in np.arange(len(E)):
            value = 0
            for i in np.arange(N):
                value = value + sp[i] * E[e][i]
            if (value % 2 == 1):
                candidate = False
        if (candidate):
            s_primes.append(sp)
    return s_primes


def Grover_Oracle(mark, qc, q, an1, an2):
    '''
    Input: mark (array)    qc (QuantumCircuit)    q (QuantumRegister)
           an1 (QuantumRegister)    an2 (QuantumRegister)
    picks out the marked state and applies a negative phase
    '''
    qc.h(an1[0])
    X_Transformation(qc, q, mark)
    if (len(mark) > 2):
        n_NOT(qc, q, an1[0], an2)
    if (len(mark) == 2):
        qc.ccx(q[0], q[1], an1[0])
    X_Transformation(qc, q, mark)
    qc.h(an1[0])


def Grover_Diffusion(mark, qc, q, an1, an2):
    '''
    Input: mark (array)    qc (QuantumCircuit)    q (QuantumRegister)
           an1 (QuantumRegister)    an2 (QuantumRegister)
    ammends the instructions for a Grover Diffusion Operation to the Quartumcircuit
    '''
    zeros_state = []
    for i in np.arange(len(mark)):
        zeros_state.append(0)
        qc.h(q[int(i)])
    Grover_Oracle(zeros_state, qc, q, an1, an2)
    for j in np.arange(len(mark)):
        qc.h(q[int(j)])


def Grover(Q, marked):
    '''
    Amends all the instructions for a Grover Search
    '''
    q = QuantumRegister(Q, name='q')
    an1 = QuantumRegister(1, name='anc')
    an2 = QuantumRegister(Q - 2, name='nanc')
    c = ClassicalRegister(Q, name='c')
    qc = QuantumCircuit(q, an1, an2, c, name='qc')
    for j in np.arange(Q):
        qc.h(q[int(j)])
    qc.x(an1[0])
    iterations = round(m.pi / 4 * 2 ** (Q / 2.0))
    for i in np.arange(iterations):
        Grover_Oracle(marked, qc, q, an1, an2)
        Grover_Diffusion(marked, qc, q, an1, an2)
    return qc, q, an1, an2, c


#==============================================
#-----------------  QFT -----------------------
#==============================================


def QFT(qc, q, qubits, **kwargs):
    '''
     Input: qc (QuantumCircuit) q (QuantumRegister) qubits (integer)
     Keyword Arguments: swap (Bool) - Adds SWAP gates after all of the phase gates have been applied
     Assigns all the gate operations for a Quantum Fourier Transformation
    '''
    R_phis = [0]
    for i in np.arange(2, int(qubits + 1)):
        R_phis.append(2 / (2 ** (i)) * m.pi)
    for j in np.arange(int(qubits)):
        qc.h(q[int(j)])
        for k in np.arange(int(qubits - (j + 1))):
            qc.cp(R_phis[k + 1], q[int(j + k + 1)], q[int(j)])
    if 'swap' in kwargs:
        if (kwargs['swap'] == True):
            for s in np.arange(m.floor(qubits / 2.0)):
                qc.swap(q[int(s)], q[int(qubits - 1 - s)])


def QFT_dgr(qc, q, qubits, **kwargs):
    '''
     Input: qc (QuantumCircuit) q (QuantumRegister) qubits (integer)
     Keyword Arguments: swap (Bool) - Adds SWAP gates after all of the phase gates have been applied
     Assigns all the gate operations for a Quantum Fourier Transformation
    '''
    if 'swap' in kwargs:
        if (kwargs['swap'] == True):
            for s in np.arange(m.floor(qubits / 2.0)):
                qc.swap(q[int(s)], q[int(qubits - 1 - s)])
    R_phis = [0]
    for i in np.arange(2, int(qubits + 1)):
        R_phis.append(-2 / (2 ** (i)) * m.pi)
    for j in np.arange(int(qubits)):
        for k in np.arange(int(j)):
            qc.cp(R_phis[int(j - k)], q[int(qubits - (k + 1))], q[int(qubits - (j + 1))])
        qc.h(q[int(qubits - (j + 1))])


def DFT(x, **kwargs):
    '''
     Input: x (array)
     Keyword Arguments: inverse (Bool) - if True, performs a Inverse Discrete Fourier Transformation instead
     Computes a classical Discrete Fourier Transformation on the array of values x, returning a new array of transformed val
    '''
    p = -1.0
    if 'inverse' in kwargs:
        P = kwargs['inverse']
        if (P == True):
            p = 1.0
    L = len(x)
    X = []
    for i in np.arange(L):
        value = 0
        for j in np.arange(L):
            value = value + x[j] * np.exp(p * 2 * m.pi * 1.0j * (int(i * j) / (L * 1.0)))
        X.append(value)
    for k in np.arange(len(X)):
        re = round(X[k].real, 5)
        im = round(X[k].imag, 5)
        if ((abs(im) == 0) and (abs(re) != 0)):
            X[k] = re
        elif ((abs(re) == 0) and (abs(im) != 0)):
            X[k] = im * 1.0j
        elif ((abs(re) == 0) and (abs(im) == 0)):
            X[k] = 0
        else:
            X[k] = re + im * 1.0j
    return X


def Quantum_Adder(qc, Qa, Qb, A, B):
    """
    Input: qc (QuantunCircuit)   Qa (QuantumRegister)   Qb (QuantumRegister)    A (array)   B (array)
    Appends all of the gate operations for a QFT based addition of two states A and B
    """
    Q = len(B)
    for n in np.arange(Q):
        if(A[n] == 1):
            qc.x(Qa[int(n + 1)])
        if(B[n] == 1):
            qc.x(Qb[int(n)])
    QFT(qc, Qa, Q + 1)
    p = 1
    for j in np.arange(Q):
        qc.cp(m.pi / (2 ** p), Qb[int(j)], Qa[0])
        p = p + 1
    for i in np.arange(1, Q + 1):
        p = 0
        for jj in np.arange(i - 1, Q):
            qc.cp(m.pi / (2 ** p), Qb[int(jj)], Qa[int(i)]) 
            p = p + 1
    QFT_dgr(qc, Qa, Q + 1)
    
    
#==============================================
#-----------------  QPE -----------------------
#==============================================


def QPE_phi(MP):
    """
    Input: array( [[float, float], [string, string]] )
    Takes in the two most probable states and their probabilities, returns phi and the approximate theta for QPE
    """
    ms = [[], []]
    for i in np.arange(2):
        for j in np.arange(len(MP[1][i])):
            ms[i].append(int(MP[1][i][j]))
    n = int(len(ms[0]))
    MS1 = From_Binary(ms[0], "R")
    MS2 = From_Binary(ms[1], "R")
    PHI = [99, 0]
    for k in np.arange(1, 5000):
        phi = k / 5000
        prob = 1 / (2 ** (2 * n)) * abs(
            (-1 + np.exp(2.0j * m.pi * phi)) / (-1 + np.exp(2.0j * m.pi * phi / (2 ** n)))) ** 2
        if (abs(prob - MP[0][0]) < abs(PHI[0] - MP[0][0])):
            PHI[0] = prob
            PHI[1] = phi
    if( (MS1 < MS2) and ( (MS1 != 0) and (MS2 != (2 ** n - 1)) ) ):
        theta = (MS1 + PHI[1]) / (2 ** n)
    elif ((MS1 > MS2) and (MS1 != 0)):
        theta = (MS1 - PHI[1]) / (2 ** n)
    else:
        theta = 1 + (MS1 - PHI[1]) / (2 ** n)
    return PHI[1], theta


#==============================================
#---------------  Q-Means ---------------------
#==============================================


def k_Data(k, n):
    """
    Input: k (integer)    n (integer)
    Creates a random set of data loosely centered around k locations
    """
    Centers = []
    for i in np.arange(k):
        Centers.append( [1.5 + np.random.rand() * 5, 1.5 * random.random() * 5] )
    count = round((0.7 * n) / k)
    Data = []
    for j in np.arange(len(Centers)):
        for j2 in np.arange(count):
            r = random.random() * 1.5
            Data.append( [Centers[j][0] + r * np.cos(random.random() * 2 * m.pi), Centers[j][1] + r * np.sin(random.random() * 2 * m.pi)] )
    diff = int(n - k * count)
    for j2 in np.arange(diff):
        Data.append( [random.random() * 8, random.random() * 8] )
    return Data


def Initial_Centroids(k, D):
    """
    Input: k (integer)    D (array)
    Picks k data points at random from the list D
    """
    D_copy = []
    for i in np.arange(len(D)):
        D_copy.append(D[i])
    Centroids = []
    for j in np.arange(k):
        p = random.randint(0, int(len(D_copy) - 1))
        Centroids.append( [ D_copy[p][0], D_copy[p][1] ] )
        D_copy.remove(D_copy[p])
    return Centroids


def Update_Centroids(CT, CL):
    """
    Input: CT (array)    CL (array)
    Based on the data within each cluster, computes and returns new Centroids using mean coordinate values
    """
    old_Centroids = []
    for c0 in np.arange(len(CT)):
        old_Centroids.append(CT[c0])
    Centroids = []
    for c1 in np.arange(len(CL)):
        mean_x = 0
        mean_y = 0
        for c2 in np.arange(len(CL[c1])):
            mean_x = mean_x + CL[c1][c2][0] / len(CL[c1])
            mean_y = mean_y + CL[c1][c2][1] / len(CL[c1])
        Centroids.append( [mean_x, mean_y] )
    return Centroids, old_Centroids


def Update_Clusters(D, CT, CL):
    """
    Input: D (array)    CT (array)    CL (array)
    Using all data points and Centroids, computes and returns the new array of Clusters
    """
    old_Clusters = []
    for c0 in np.arange(len(CL)):
        old_Clusters.append(CL[c0])
    Clusters = []
    for c1 in np.arange(len(CT)):
        Clusters.append([])
    for d in np.arange(len(D)):
        closest = 'c'
        distance = 1000000
        for c2 in np.arange(len(Clusters)):
            Dist = m.sqrt( (CT[c2][0] - D[d][0]) ** 2 + (CT[c2][1] - D[d][1]) ** 2 )
            if(Dist < distance):
                distance = Dist
                closest = int(c2)
        Clusters[closest].append(D[d])
    return Clusters, old_Clusters


def Check_Termination(CL, oCL):
    """
    Input: CL (array)    oCL (array)
    Returns True or False based on whether the Update_Clusters function has caused any data points to change clusters
    """
    terminate = True
    for c1 in np.arange(len(oCL)):
        for c2 in np.arange(len(oCL[c1])):
            P_found = False
            for c3 in np.arange(len(CL[c1])):
                if(CL[c1][c3] == oCL[c1][c2]):
                    P_found = True
            if(P_found == False):
                terminate = False
    return terminate


def Draw_Data(CL, CT, oCT, fig, ax, colors, colors2):
    """
    Input: CL (array)    CT (array)    oCT (array)    fig (matplotlib figure)
           ax (figure subplot)    colors (array of color strings)
           colors2 (array of color strings)
    Using the arrays Clusters, Centroids, and old Centroids, draws and colors each data point according to its cluster
    """
    for j1 in np.arange(len(CL)):
        ax.scatter(oCT[j1][0], oCT[j1][1], color='white', marker='s', s=80)
    for cc in np.arange(len(CL)):
        for ccc in np.arange(len(CL[cc])):
            ax.scatter(CL[cc][ccc][0], CL[cc][ccc][1], color=colors[cc], s=10)
    for j2 in np.arange(len(CL)):
        ax.scatter(CT[j2][0], CT[j2][1], color=colors2[j2], marker='x', s=50)
    fig.canvas.draw()
    time.sleep(1)


def SWAP_Test(qc, control, q1, q2, classical, shots):
    """
    Input: qc (QuantumCircuit)    Control (QuantumRegister[i])
           q1 (QuantumRegister[i])    q2 (QuantumRegister[i])
           classical (ClassicalRegister[i])    shots (integer)
    Appends the necessary gates for 2-Qubit SWAP Test and returns the number of |0> state counts
    """
    qc.h(control)
    qc.cswap(control, q1, q2)
    qc.h(control)
    qc.measure(control, classical)
    D = {'0': 0}
    D.update(Measurement(qc, shots=shots, return_M=True, print_M=False))
    return D['0']


def Bloch_State(point, p_range):
    """
    Input: point (array)    p_range (array)
    Returns the corresponding theta and phi values of the data point p, according to min / max paramters of P
    """
    delta_x = (point[0] - p_range[0]) / (1.0 * p_range[1] - p_range[0])
    delta_y = (point[1] - p_range[2]) / (1.0 * p_range[3] - p_range[2])
    theta = np.pi / 2 * (delta_x + delta_y)
    phi = np.pi / 2 * (delta_x - delta_y + 1)
    return theta, phi


def Q_Update_Clusters(data, centroids, old_clusters, p_range, shots):
    """
    Input: data (array)    centroids (array)    old_clusters (array)
           p_range (array)    shots (integer)
    Using all data points, Centroids, uses the SWAP Test to compute and return the new array of Clusters
    """
    clusters = []
    for i in np.arange(len(centroids)):
        clusters.append([])

    for i in np.arange(len(data)):
        # consider every point in data set
        closest = -1
        distance = 0
        tmp_x = sorted([data[i][0], *[centroid[0] for centroid in centroids]])
        tmp_y = sorted([data[i][1], *[centroid[1] for centroid in centroids]])
        p_range = [tmp_x[0], tmp_x[-1], tmp_y[0], tmp_y[-1]]
        p_theta, p_phi = Bloch_State(data[i], p_range)
        for j in np.arange(len(clusters)):
            c_theta, c_phi = Bloch_State(centroids[j], p_range)

            q = QuantumRegister(3, name='q')
            c = ClassicalRegister(1, name='c')
            qc = QuantumCircuit(q, c, name='qc')
            qc.u(p_theta, p_phi, 0, q[1])
            qc.u(c_theta, c_phi, 0, q[2])
            tmp = SWAP_Test(qc, q[0], q[1], q[2], c[0], shots)
            if(tmp > distance):
                distance = tmp
                closest = int(j)
        clusters[closest].append(data[i])
    return clusters, old_clusters