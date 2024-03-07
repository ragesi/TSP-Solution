"""
This file is meant for matrix exponentials and their corresponding complex space operations
"""
import numpy as np
from sympy.matrices import Matrix
import math


def get_exponential_unitary(matrix):
    """
    The only operation I need, the rest are helper functions.

    formula: Unitary_matrix = e^(2*pi*i*matrix) approximated to iterations (count)
    """
    if Matrix(matrix).is_diagonalizable():
        return get_exact_exponential(matrix)
    else:
        return get_approximation_exponential(matrix)

def convert_seperate_real_and_imaginary_sympy_to_numpy(r,i):
    x = np.zeros(shape=np.array(r).shape)
    y = np.zeros(shape=np.array(r).shape)
    for a in range(len(y)):
        for b in range(len(y[a])):
            x[a][b] = float(r[a+b])
            y[a][b] = float(i[a+b])
    return (x + y*1j).copy()

def convert_sympy_to_numpy_imaginary_matrix(m):
    r,i = m.as_real_imag() # seperate into real and imaginary parts
    return convert_seperate_real_and_imaginary_sympy_to_numpy(r,i)

def get_exact_exponential(matrix):
    (P, D) = Matrix(matrix).diagonalize()
    
    D_exp = Matrix(np.diag([math.cos(val*2*math.pi) + 1j*math.sin(val*2*math.pi) for val in np.diag(np.array(D))]))

    exponentiated_matrix = P * D_exp * P.inv()
    real_portion,imaginary_portion = exponentiated_matrix.as_real_imag()

    return convert_seperate_real_and_imaginary_sympy_to_numpy(real_portion, imaginary_portion)


def get_approximation_exponential(matrix, iterations=100):
    n = len(matrix)
    final_matrix = np.eye(n) + np.zeros((n,n))*1j
    for k in range(iterations):
        constant = ((2*math.pi*1j)**k)/math.factorial(k)
        # constant = 1/math.factorial(k)
        final_matrix += constant*np.linalg.matrix_power(matrix,k)
    # print(f"final_matrix.real:{final_matrix.real}")
    # print(f"final_matrix.imag:{final_matrix.imag}")
    # print(f"final_matrix:{final_matrix}")
    return final_matrix