from lib.colors import bcolors


def lagrange_interpolation(x_data, y_data, x):
    """
    Performs Lagrange polynomial interpolation for a given set of data points.

    Parameters:
    x_data (list of float): The x-coordinates of the known data points.
    y_data (list of float): The y-coordinates of the known data points.
    x (float): The x-value at which to evaluate the interpolated polynomial.

    Returns:
    float: The interpolated y-value corresponding to the input x.

    Notes:
    - Assumes that all x_data values are distinct.
    - The function constructs the interpolation polynomial using the Lagrange basis method.
    """
    n = len(x_data)
    result = 0.0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result


if __name__ == '__main__':

    x_data = [1, 2, 5]
    y_data = [1, 0, 2]
    x_interpolate = 3  # The x-value where you want to interpolate
    y_interpolate = lagrange_interpolation(x_data, y_data, x_interpolate)
    print(bcolors.OKBLUE, "\nInterpolated value at x =", x_interpolate, "is y =", y_interpolate, bcolors.ENDC)

