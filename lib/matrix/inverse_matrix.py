from lib.colors import bcolors
from lib.matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix
import numpy as np


def inverse(matrix):
    """
    Finds the inverse of a non-singular square matrix using elementary row operations.

    The function applies a series of elementary row operations to transform the input
    matrix into the identity matrix. Simultaneously, it applies the same operations
    to the identity matrix to construct the inverse.

    Parameters:
    matrix (np.ndarray): A square (n x n) NumPy array representing the matrix to invert.

    Returns:
    np.ndarray: The inverse of the input matrix.

    Raises:
    ValueError: If the matrix is not square or is singular (i.e., has a zero on the diagonal
                during the elimination process).
    """
    print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n", bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            raise ValueError("matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------", bcolors.ENDC)
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------", bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)

    return identity


if __name__ == '__main__':

    A = np.array([[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 6]])

    try:
        A_inverse = inverse(A)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print("=====================================================================================================================", bcolors.ENDC)

    except ValueError as e:
        print(str(e))
