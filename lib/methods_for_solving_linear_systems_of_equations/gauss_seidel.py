import numpy as np
from numpy.linalg import norm

from lib.colors import bcolors
from lib.matrix_utility import is_diagonally_dominant

def gauss_seidel(A, b, X0, TOL=1e-16, N=200):
    """
    Solves the linear system Ax = b using the Gauss-Seidel iterative method.

    Parameters:
    A (np.ndarray): Coefficient matrix of size (n x n).
    b (np.ndarray): Right-hand side vector of length n.
    X0 (np.ndarray): Initial guess vector for the solution.
    TOL (float): Tolerance for the stopping criterion based on infinity norm of difference (default 1e-16).
    N (int): Maximum number of iterations allowed (default 200).

    Returns:
    tuple: Approximate solution vector as a tuple after convergence or maximum iterations.

    Notes:
    - Prints iteration number and the approximate solution vector at each iteration.
    - Checks for diagonal dominance of the matrix and prints a message if true.
    - Stops when the infinity norm of the difference between consecutive iterates is less than TOL.
    - If the method does not converge within N iterations, returns the last approximation.
    """
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('matrix is diagonally dominant - performing gauss seidel algorithm\n')

    print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, n + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


if __name__ == '__main__':

    A = np.array([[3, -1, 1], [0, 1, -1], [1, 1, -2]])
    b = np.array([4, -1, -3])
    X0 = np.zeros_like(b)

    solution = gauss_seidel(A, b, X0)
    print(bcolors.OKBLUE, "\nApproximate solution:", solution)
