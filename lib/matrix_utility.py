import numpy as np

def print_matrix(matrix):
    """
    Print the matrix row by row.

    Parameters:
        matrix (list of lists): matrix to print
    """
    for row in matrix:
        for element in row:
            print(element, end=" ")
        print()
    print()

def MaxNorm(matrix):
    """
    Calculate the maximum row sum norm (infinity norm) of the matrix.

    Parameters:
        matrix (list of lists): input matrix

    Returns:
        float: maximum absolute row sum
    """
    max_norm = 0
    for i in range(len(matrix)):
        norm = sum(abs(matrix[i][j]) for j in range(len(matrix)))
        if norm > max_norm:
            max_norm = norm
    return max_norm

def swap_row(mat, i, j):
    """
    Swap rows i and j in matrix mat (in-place).

    Parameters:
        mat (list of lists): augmented matrix with n rows and n+1 columns
        i (int): first row index
        j (int): second row index
    """
    N = len(mat)
    for k in range(N + 1):
        mat[i][k], mat[j][k] = mat[j][k], mat[i][k]

def is_diagonally_dominant(mat):
    """
    Check if a matrix is diagonally dominant.

    Parameters:
        mat (list of lists or numpy array): square matrix

    Returns:
        bool: True if diagonally dominant, False otherwise
    """
    if mat is None:
        return False
    d = np.diag(np.abs(mat))
    s = np.sum(np.abs(mat), axis=1) - d
    return np.all(d > s)

def is_square_matrix(mat):
    """
    Check if a matrix is square.

    Parameters:
        mat (list of lists): matrix to check

    Returns:
        bool: True if square, False otherwise
    """
    if mat is None:
        return False
    rows = len(mat)
    return all(len(row) == rows for row in mat)

def reorder_dominant_diagonal(matrix):
    """
    Reorder matrix rows and columns based on descending order of diagonal values.

    Parameters:
        matrix (numpy array): square matrix

    Returns:
        numpy array: reordered matrix
    """
    n = len(matrix)
    permutation = np.argsort(np.diag(matrix))[::-1]
    return matrix[permutation][:, permutation]

def DominantDiagonalFix(matrix):
    """
    Attempt to reorder matrix rows to achieve a diagonally dominant matrix.

    Parameters:
        matrix (list of lists): matrix to reorder

    Returns:
        list of lists: reordered matrix or original if not possible
    """
    dom = [0]*len(matrix)
    result = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] > sum(abs(int(x)) for x in matrix[i]) - matrix[i][j]:
                dom[i] = j
    for i in range(len(matrix)):
        result.append([])
        if i not in dom:
            print("Couldn't find dominant diagonal.")
            return matrix
    for i, j in enumerate(dom):
        result[j] = matrix[i]
    return result

def swap_rows_elementary_matrix(n, row1, row2):
    """
    Create an elementary matrix to swap two rows.

    Parameters:
        n (int): size of identity matrix
        row1 (int): first row index
        row2 (int): second row index

    Returns:
        numpy array: elementary matrix representing the row swap
    """
    E = np.identity(n)
    E[[row1, row2]] = E[[row2, row1]]
    return np.array(E)

def matrix_multiply(A, B):
    """
    Multiply two matrices A and B.

    Parameters:
        A (list of lists): matrix A
        B (list of lists): matrix B

    Returns:
        numpy array: result of multiplication
    """
    if len(A[0]) != len(B):
        raise ValueError("matrix dimensions are incompatible for multiplication.")
    result = [[sum(A[i][k] * B[k][j] for k in range(len(B)))
               for j in range(len(B[0]))] for i in range(len(A))]
    return np.array(result)

def row_addition_elementary_matrix(n, target_row, source_row, scalar=1.0):
    """
    Create an elementary matrix for row addition operation.

    Parameters:
        n (int): matrix size
        target_row (int): row to be replaced
        source_row (int): row to add from
        scalar (float): multiplier for source_row

    Returns:
        numpy array: elementary matrix representing row addition
    """
    if target_row < 0 or source_row < 0 or target_row >= n or source_row >= n:
        raise ValueError("Invalid row indices.")
    if target_row == source_row:
        raise ValueError("Source and target rows cannot be the same.")
    E = np.identity(n)
    E[target_row, source_row] = scalar
    return np.array(E)

def scalar_multiplication_elementary_matrix(n, row_index, scalar):
    """
    Create an elementary matrix for multiplying a row by a scalar.

    Parameters:
        n (int): matrix size
        row_index (int): row to multiply
        scalar (float): scalar value

    Returns:
        numpy array: elementary matrix representing scalar multiplication
    """
    if row_index < 0 or row_index >= n:
        raise ValueError("Invalid row index.")
    if scalar == 0:
        raise ValueError("Scalar cannot be zero for row multiplication.")
    E = np.identity(n)
    E[row_index, row_index] = scalar
    return np.array(E)

def Determinant(matrix, mul):
    """
    Compute the determinant of a matrix recursively.

    Parameters:
        matrix (list of lists): square matrix
        mul (int or float): multiplier (use 1 when calling initially)

    Returns:
        float: determinant value
    """
    width = len(matrix)
    if width == 1:
        return mul * matrix[0][0]
    else:
        sign = -1
        det = 0
        for i in range(width):
            m = [ [matrix[j][k] for k in range(width) if k != i]
                  for j in range(1, width) ]
            sign *= -1
            det += mul * Determinant(m, sign * matrix[0][i])
    return det

def partial_pivoting(A, i, N):
    """
    Perform partial pivoting on matrix A at column i.

    Parameters:
        A (numpy array): matrix
        i (int): pivot column index
        N (int): matrix dimension

    Returns:
        None or str: "Singular matrix" if pivot is zero
    """
    pivot_row = i
    v_max = A[pivot_row][i]
    for j in range(i + 1, N):
        if abs(A[j][i]) > v_max:
            v_max = A[j][i]
            pivot_row = j
    if A[i][pivot_row] == 0:
        return "Singular matrix"
    if pivot_row != i:
        e_matrix = swap_rows_elementary_matrix(N, i, pivot_row)
        print(f"elementary matrix for swap between row {i} to row {pivot_row} :\n {e_matrix} \n")
        A = np.dot(e_matrix, A)
        print(f"The matrix after elementary operation :\n {A}")
        print("------------------------------------------------------------------")

def MultiplyMatrix(matrixA, matrixB):
    """
    Multiply two matrices (list of lists).

    Parameters:
        matrixA (list of lists)
        matrixB (list of lists)

    Returns:
        list of lists: product matrix
    """
    return [[sum(matrixA[i][k] * matrixB[k][j] for k in range(len(matrixB)))
             for j in range(len(matrixB[0]))] for i in range(len(matrixA))]

def MakeIMatrix(cols, rows):
    """
    Create an identity matrix with given dimensions.

    Parameters:
        cols (int): number of columns
        rows (int): number of rows

    Returns:
        list of lists: identity matrix
    """
    return [[1 if x == y else 0 for y in range(cols)] for x in range(rows)]

def MulMatrixVector(InversedMat, b_vector):
    """
    Multiply matrix by vector.

    Parameters:
        InversedMat (list of lists): matrix
        b_vector (list of lists): vector as column matrix

    Returns:
        list of lists: result vector (column matrix)
    """
    result = [[0] for _ in b_vector]
    for i in range(len(InversedMat)):
        for k in range(len(b_vector)):
            result[i][0] += InversedMat[i][k] * b_vector[k][0]
    return result

def RowXchageZero(matrix, vector):
    """
    Swap rows in matrix and vector if diagonal element is zero.

    Parameters:
        matrix (list of lists)
        vector (list of lists)

    Returns:
        tuple: (matrix, vector) after row swaps
    """
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][i] == 0:
                matrix[i], matrix[j] = matrix[j], matrix[i]
                vector[i], vector[j] = vector[j], vector[i]
    return [matrix, vector]

def Cond(matrix, invert):
    """
    Calculate and print condition number of matrix.

    Parameters:
        matrix (list of lists)
        invert (list of lists): inverse matrix

    Returns:
        float: condition number
    """
    print("|| A ||max = ", MaxNorm(matrix))
    print("|| A(-1) ||max = ", MaxNorm(invert))
    return MaxNorm(matrix) * MaxNorm(invert)

def InverseMatrix(matrix, vector):
    """
    Compute inverse of matrix using elementary row operations.

    Parameters:
        matrix (list of lists)
        vector (list of lists)

    Returns:
        list of lists: inverse matrix if exists, else None
    """
    if Determinant(matrix, 1) == 0:
        print("Error, Singular matrix\n")
        return
    result = MakeIMatrix(len(matrix), len(matrix))
    for i in range(len(matrix[0])):
        matrix, vector = RowXchange(matrix, vector)
        elementary = MakeIMatrix(len(matrix[0]), len(matrix))
        elementary[i][i] = 1 / matrix[i][i]
        result = MultiplyMatrix(elementary, result)
        matrix = MultiplyMatrix(elementary, matrix)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -matrix[j][i]
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)
    for i in range(len(matrix[0]) - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -matrix[j][i]
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)
    return result

def RowXchange(matrix, vector):
    """
    Perform partial pivoting row swaps on matrix and vector.

    Parameters:
        matrix (list of lists)
        vector (list of lists)

    Returns:
        tuple: (matrix, vector) after row swaps
    """
    for i in range(len(matrix)):
        max_val = abs(matrix[i][i])
        for j in range(i, len(matrix)):
            if abs(matrix[j][i]) > max_val:
                matrix[i], matrix[j] = matrix[j], matrix[i]
                vector[i], vector[j] = vector[j], vector[i]
                max_val = abs(matrix[i][i])
    return [matrix, vector]
