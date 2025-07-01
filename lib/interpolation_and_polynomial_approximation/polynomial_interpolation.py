from lib.colors import bcolors
from lib.matrix_utility import *


def GaussJordanElimination(matrix, vector):
    """
    Solves a system of linear equations using Gauss-Jordan elimination.

    Parameters:
    matrix (list of lists): Coefficient matrix A (n x n).
    vector (list): Right-hand side vector b (size n).

    Returns:
    list: Solution vector x such that Ax = b.
    """
    matrix, vector = RowXchange(matrix, vector)
    invert = InverseMatrix(matrix, vector)
    return MulMatrixVector(invert, vector)


def UMatrix(matrix, vector):
    """
    Computes the upper triangular matrix U in LU decomposition of matrix A.

    Parameters:
    matrix (list of lists): Input matrix A (n x n).
    vector (list): Right-hand side vector (used for pivoting).

    Returns:
    list of lists: Upper triangular matrix U.
    """
    U = MakeIMatrix(len(matrix), len(matrix))
    for i in range(len(matrix[0])):
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i]) / matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)
    U = MultiplyMatrix(U, matrix)
    return U


def LMatrix(matrix, vector):
    """
    Computes the lower triangular matrix L in LU decomposition of matrix A.

    Parameters:
    matrix (list of lists): Input matrix A (n x n).
    vector (list): Right-hand side vector (used for pivoting).

    Returns:
    list of lists: Lower triangular matrix L.
    """
    L = MakeIMatrix(len(matrix), len(matrix))
    for i in range(len(matrix[0])):
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i]) / matrix[i][i]
            L[j][i] = (matrix[j][i]) / matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)
    return L


def SolveLU(matrix, vector):
    """
    Solves a system of linear equations using LU decomposition.

    Parameters:
    matrix (list of lists): Coefficient matrix A (n x n).
    vector (list): Right-hand side vector b.

    Returns:
    list: Solution vector x such that Ax = b, computed via LU decomposition.
    """
    matrixU = UMatrix(matrix)
    matrixL = LMatrix(matrix)
    return MultiplyMatrix(InverseMatrix(matrixU), MultiplyMatrix(InverseMatrix(matrixL), vector))


def solveMatrix(matrixA, vectorb):
    """
    Solves a system of equations using either Gauss-Jordan or LU decomposition,
    depending on whether the matrix is singular.

    Parameters:
    matrixA (list of lists): Coefficient matrix A (n x n).
    vectorb (list): Right-hand side vector b.

    Returns:
    list: Solution or reconstructed matrix, depending on method used.
    """
    detA = Determinant(matrixA, 1)
    print(bcolors.YELLOW, "\nDET(A) = ", detA)

    if detA != 0:
        print("CondA = ", Cond(matrixA, InverseMatrix(matrixA, vectorb)), bcolors.ENDC)
        print(bcolors.OKBLUE, "\nnon-Singular matrix - Perform GaussJordanElimination", bcolors.ENDC)
        result = GaussJordanElimination(matrixA, vectorb)
        print(np.array(result))
        return result
    else:
        print("Singular matrix - Perform LU Decomposition\n")
        print("matrix U: \n")
        print(np.array(UMatrix(matrixA, vectorb)))
        print("\nmatrix L: \n")
        print(np.array(LMatrix(matrixA, vectorb)))
        print("\nmatrix A=LU: \n")
        result = MultiplyMatrix(LMatrix(matrixA, vectorb), UMatrix(matrixA, vectorb))
        print(np.array(result))
        return result


def polynomialInterpolation(table_points, x):
    """
    Performs polynomial interpolation on given points and evaluates the resulting
    polynomial at a specific x value.

    Parameters:
    table_points (list of tuples): Points in the form (x, y).
    x (float): The x-value at which to evaluate the interpolated polynomial.

    Returns:
    float: The interpolated y-value at x.
    """
    matrix = [[point[0] ** i for i in range(len(table_points))] for point in table_points]
    b = [[point[1]] for point in table_points]

    print(bcolors.OKBLUE, "The matrix obtained from the points: ", bcolors.ENDC, '\n', np.array(matrix))
    print(bcolors.OKBLUE, "\nb vector: ", bcolors.ENDC, '\n', np.array(b))
    matrixSol = solveMatrix(matrix, b)

    result = sum([matrixSol[i][0] * (x ** i) for i in range(len(matrixSol))])
    print(bcolors.OKBLUE, "\nThe polynom:", bcolors.ENDC)
    print('P(X) = ' + '+'.join([f'({matrixSol[i][0]}) * x^{i} ' for i in range(len(matrixSol))]))
    print(bcolors.OKGREEN, f"\nThe Result of P(X={x}) is:", bcolors.ENDC)
    print(result)
    return result


if __name__ == '__main__':

    table_points = [(0, 0), (1, 0.8415), (2, 0.9093), (3, 0.1411), (4, -0.7568), (5, -0.9589), (6, -0.2794)]
    x = 1.28
    print(bcolors.OKBLUE, "----------------- Interpolation & Extrapolation Methods -----------------\n", bcolors.ENDC)
    print(bcolors.OKBLUE, "Table Points: ", bcolors.ENDC, table_points)
    print(bcolors.OKBLUE, "Finding an approximation to the point: ", bcolors.ENDC, x, '\n')
    polynomialInterpolation(table_points, x)
    print(bcolors.OKBLUE, "\n---------------------------------------------------------------------------\n", bcolors.ENDC)
