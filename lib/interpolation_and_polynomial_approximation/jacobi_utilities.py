def checkIfSquare(mat):
    """
    Checks if the given matrix is square.

    Parameters:
    mat (list): A matrix represented as a list of lists.

    Returns:
    bool: True if the matrix is square, False otherwise.
    """
    rows = len(mat)
    for i in mat:
        if len(i) != rows:
            return False
    return True


def isDDM(m, n):
    """
    Checks whether the given matrix is diagonally dominant.

    Parameters:
    m (list): The matrix to check.
    n (int): The number of rows (and columns) in the matrix.

    Returns:
    bool: True if the matrix is diagonally dominant, False otherwise.
    """
    for i in range(0, n):
        row_sum = 0
        for j in range(0, n):
            row_sum += abs(m[i][j])
        row_sum -= abs(m[i][i])
        if abs(m[i][i]) < row_sum:
            return False
    return True


def rowSum(row, n, x):
    """
    Calculates the dot product of a matrix row and a vector.

    Parameters:
    row (list): A row from the matrix.
    n (int): The number of elements in the row.
    x (list): A vector of variables.

    Returns:
    float: The resulting sum.
    """
    sum1 = 0
    for i in range(n):
        sum1 += row[i] * x[i]
    return sum1


def checkResult(result, last_result, n, epsilon):
    """
    Checks whether the difference between two result vectors is within a given tolerance.

    Parameters:
    result (list): The current result vector.
    last_result (list): The previous result vector.
    n (int): The number of elements in the vectors.
    epsilon (float): The acceptable difference threshold.

    Returns:
    bool: True if the result has converged, False otherwise.
    """
    for i in range(n):
        if abs(result[i] - last_result[i]) > epsilon:
            return False
    return True


def Jacobi(mat, b, epsilon=0.000001):
    """
    Solves a system of linear equations using the Jacobi iterative method.

    Parameters:
    mat (list): The coefficient matrix (must be square).
    b (list): The right-hand side result vector.
    epsilon (float): The convergence threshold (default is 1e-6).

    Returns:
    list or str: The solution vector if successful, or an error message if input is invalid.
    """
    n = len(mat)
    if not checkIfSquare(mat):
        return "matrix is not square"
    if len(b) != n:
        return "b is not in the right size"

    if not isDDM(mat, n):
        print("matrix is not Diagonally Dominant")

    last_result = [0 for _ in range(n)]
    result = last_result.copy()

    print("all results:\nx\t\ty\t  z")
    count = 0
    while True:
        for i in range(n):
            result[i] = b[i] - (rowSum(mat[i], n, last_result) - mat[i][i] * last_result[i])
            result[i] /= mat[i][i]

        print("i = " + str(count) + ": " + str(result))
        count += 1

        if checkResult(result, last_result, n, epsilon):
            return result

        last_result = result.copy()
