import numpy as np
from lib.matrix.inverse_matrix import inverse
from lib.colors import bcolors
from lib.matrix_utility import print_matrix


def norm(mat):
    """
    Calculates the matrix norm based on the maximum absolute row sum (infinity norm).

    Parameters:
    mat (list of lists or np.ndarray): The input square matrix.

    Returns:
    float: The maximum row sum norm of the matrix.
    """
    size = len(mat)
    max_row = 0
    for row in range(size):
        sum_row = 0
        for col in range(size):
            sum_row += abs(mat[row][col])
        if sum_row > max_row:
            max_row = sum_row
    return max_row


def condition_number(A):
    """
    Computes the condition number of a square matrix A using the infinity norm.

    The condition number is defined as: ||A|| * ||A^(-1)||
    where ||·|| is the matrix norm (maximum absolute row sum in this implementation).

    Parameters:
    A (np.ndarray): A square matrix.

    Returns:
    float: The condition number of the matrix A.

    Side Effects:
    - Prints intermediate matrices and norms for visualization.
    """
    # Step 1: Calculate the max norm (infinity norm) of A
    norm_A = norm(A)

    # Step 2: Calculate the inverse of A
    A_inv = inverse(A)

    # Step 3: Calculate the max norm of the inverse of A
    norm_A_inv = norm(A_inv)

    # Step 4: Compute the condition number
    cond = norm_A * norm_A_inv

    print(bcolors.OKBLUE, "A:", bcolors.ENDC)
    print_matrix(A)

    print(bcolors.OKBLUE, "inverse of A:", bcolors.ENDC)
    print_matrix(A_inv)

    print(bcolors.OKBLUE, "Max Norm of A:", bcolors.ENDC, norm_A, "\n")

    print(bcolors.OKBLUE, "max norm of the inverse of A:", bcolors.ENDC, norm_A_inv)

    return cond


if __name__ == '__main__':
    A = np.array([[2, 1.7, -2.5],
                  [1.24, -2, -0.5],
                  [3, 0.2, 1]])
    cond = condition_number(A)

    print(bcolors.OKGREEN, "\n condition number: ", cond, bcolors.ENDC)
