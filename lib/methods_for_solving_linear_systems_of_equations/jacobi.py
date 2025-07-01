import numpy as np
from numpy.linalg import norm

from lib.colors import bcolors
from lib.matrix_utility import is_diagonally_dominant, DominantDiagonalFix, is_square_matrix

def jacobi_iterative(A, b, X0, TOL=1e-16, N=200):
    """
    Solves the linear system Ax = b using the Jacobi iterative method.

    Parameters:
    A (np.ndarray): Coefficient matrix of size (n x n).
    b (np.ndarray): Right-hand side vector of length n.
    X0 (np.ndarray): Initial guess vector for the solution.
    TOL (float, optional): Tolerance for convergence based on infinity norm difference (default 1e-16).
    N (int, optional): Maximum number of iterations (default 200).

    Returns:
    tuple: Approximate solution vector as a tuple after convergence or maximum iterations.

    Notes:
    - Prints iteration number and current approximation vector at each step.
    - Checks if matrix is diagonally dominant and prints a notice.
    - Stops if the infinity norm of (x_new - x_old) is less than TOL.
    - If convergence is not reached within N iterations, returns last approximation.
    """
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('matrix is diagonally dominant - preforming jacobi algorithm\n')

    print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, n + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    while k <= N:
        x = np.zeros(n, dtype=np.double)
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * X0[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


if __name__ == "__main__":

    A = np.array([[3, -1, 1], [0, 1, -1], [1, 1, -2]])
    b = np.array([4, -1, -3])

    x = np.zeros_like(b, dtype=np.double)
    solution = jacobi_iterative(A, b, x)

    print(bcolors.OKBLUE, "\nApproximate solution:", solution)
