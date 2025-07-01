import numpy as np

from lib.colors import bcolors
from lib.matrix_utility import swap_rows_elementary_matrix, row_addition_elementary_matrix


def lu(A):
    """
    Performs LU Decomposition of a square matrix A using elementary matrices and partial pivoting.

    Parameters:
    A (list of lists or np.ndarray): The square matrix to decompose (N x N).

    Returns:
    tuple:
        L (np.ndarray): Lower triangular matrix (N x N).
        U (np.ndarray): Upper triangular matrix (N x N).

    Raises:
    ValueError: If the matrix is singular and LU decomposition cannot be performed.

    Notes:
    - Uses elementary matrices to perform row swaps and row additions.
    - Prints the elementary matrices and intermediate matrices after each operation.
    """
    N = len(A)
    L = np.eye(N)  # Create an identity matrix of size N x N

    for i in range(N):
        # Partial Pivoting: Find pivot row with largest absolute value in column i
        pivot_row = i
        v_max = A[pivot_row][i]
        for j in range(i + 1, N):
            if abs(A[j][i]) > v_max:
                v_max = A[j][i]
                pivot_row = j

        # Check for singularity
        if A[i][pivot_row] == 0:
            raise ValueError("can't perform LU Decomposition")

        # Swap rows if needed
        if pivot_row != i:
            e_matrix = swap_rows_elementary_matrix(N, i, pivot_row)
            print(f"elementary matrix for swap between row {i} to row {pivot_row} :\n {e_matrix} \n")
            A = np.dot(e_matrix, A)
            print(f"The matrix after elementary operation :\n {A}")
            print(bcolors.OKGREEN,"---------------------------------------------------------------------------", bcolors.ENDC)

        for j in range(i + 1, N):
            # Compute multiplier and corresponding elementary matrix for elimination
            m = -A[j][i] / A[i][i]
            e_matrix = row_addition_elementary_matrix(N, j, i, m)
            e_inverse = np.linalg.inv(e_matrix)
            L = np.dot(L, e_inverse)
            A = np.dot(e_matrix, A)
            print(f"elementary matrix to zero the element in row {j} below the pivot in column {i} :\n {e_matrix} \n")
            print(f"The matrix after elementary operation :\n {A}")
            print(bcolors.OKGREEN,"---------------------------------------------------------------------------", bcolors.ENDC)

    U = A
    return L, U


def backward_substitution(mat):
    """
    Performs backward substitution on an augmented upper triangular matrix to find solution vector.

    Parameters:
    mat (list of lists or np.ndarray): Augmented upper-triangular matrix (N x N+1).

    Returns:
    np.ndarray: Solution vector x of length N.
    """
    N = len(mat)
    x = np.zeros(N)  # An array to store solution

    for i in range(N - 1, -1, -1):
        x[i] = mat[i][N]
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]
        x[i] = x[i] / mat[i][i]

    return x


def lu_solve(A_b):
    """
    Solves a system of linear equations given by the augmented matrix A_b using LU decomposition.

    Parameters:
    A_b (list of lists or np.ndarray): Augmented matrix of size N x (N+1) representing the system.

    Prints:
    - The L and U matrices.
    - The solution vector.
    """
    L, U = lu(A_b)
    print("Lower triangular matrix L:\n", L)
    print("Upper triangular matrix U:\n", U)

    result = backward_substitution(U)
    print(bcolors.OKBLUE, "\nSolution for the system:")
    for x in result:
        print("{:.6f}".format(x))


if __name__ == '__main__':

    A_b = [
        [1, -1, 2, -1, -8],
        [2, -2, 3, -3, -20],
        [1, 1, 1, 0, -2],
        [1, -1, 4, 3, 4]
    ]

    lu_solve(A_b)
