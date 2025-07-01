import numpy as np
from lib.colors import bcolors
from lib.matrix_utility import swap_row


def gaussianElimination(mat):
    """
    Solves a system of linear equations using Gaussian elimination with partial pivoting.

    Parameters:
    mat (list of lists): Augmented matrix representing the system (n x (n+1)).

    Returns:
    numpy.ndarray or str:
        - A NumPy array containing the solution vector if the system is consistent and non-singular.
        - A string message indicating if the matrix is singular or system is inconsistent/has infinitely many solutions.
    """
    N = len(mat)

    singular_flag = forward_substitution(mat)

    if singular_flag != -1:
        if mat[singular_flag][N]:
            return "Singular matrix (Inconsistent System)"
        else:
            return "Singular matrix (May have infinitely many solutions)"

    # if matrix is non-singular: get solution to system using backward substitution
    return backward_substitution(mat)


def forward_substitution(mat):
    """
    Performs forward elimination with partial pivoting to convert the matrix into upper triangular form.

    Parameters:
    mat (list of lists): Augmented matrix (n x (n+1)).

    Returns:
    int:
        - -1 if matrix is non-singular.
        - The index of the singular row if the matrix is singular.
    """
    N = len(mat)
    for k in range(N):
        # Partial Pivoting: Find the pivot row with the largest absolute value in column k
        pivot_row = k
        v_max = abs(mat[pivot_row][k])
        for i in range(k + 1, N):
            if abs(mat[i][k]) > v_max:
                v_max = abs(mat[i][k])
                pivot_row = i

        # Check for singularity
        if abs(mat[pivot_row][k]) == 0:
            return k  # matrix is singular

        # Swap current row with pivot row if necessary
        if pivot_row != k:
            swap_row(mat, k, pivot_row)

        # Eliminate entries below pivot
        for i in range(k + 1, N):
            m = mat[i][k] / mat[k][k]
            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j] * m
            mat[i][k] = 0  # fill lower triangular matrix with zeros

    return -1


def backward_substitution(mat):
    """
    Performs backward substitution on an upper triangular matrix to find the solution vector.

    Parameters:
    mat (list of lists): Augmented upper-triangular matrix (n x (n+1)).

    Returns:
    numpy.ndarray: Solution vector x.
    """
    N = len(mat)
    x = np.zeros(N)

    for i in range(N - 1, -1, -1):
        x[i] = mat[i][N]
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]
        x[i] = x[i] / mat[i][i]

    return x


if __name__ == '__main__':
    A_b = [
        [1, -1, 2, -1, -8],
        [2, -2, 3, -3, -20],
        [1, 1, 1, 0, -2],
        [1, -1, 4, 3, 4]
    ]

    result = gaussianElimination(A_b)
    if isinstance(result, str):
        print(result)
    else:
        print(bcolors.OKBLUE, "\nSolution for the system:")
        for x in result:
            print("{:.6f}".format(x))
