from lib.colors import bcolors


def linearInterpolation(table_points, point):
    """
    Performs linear interpolation (or extrapolation) for a given point
    using a table of known (x, y) data points.

    Parameters:
    table_points (list of tuples): List of (x, y) coordinate pairs, assumed to be sorted by x.
    point (float): The x-value for which the function approximates a y-value.

    Behavior:
    - If the point lies within the range of table_points, interpolation is used.
    - If the point is outside the range, extrapolation is performed using the first two points.
    - The result is printed with color formatting using bcolors.

    Returns:
    None
    """
    p = []
    result = 0
    flag = 1
    for i in range(len(table_points)):
        p.append(table_points[i][0])
    for i in range(len(p) - 1):
        if i <= point <= i + 1:
            x1 = table_points[i][0]
            x2 = table_points[i + 1][0]
            y1 = table_points[i][1]
            y2 = table_points[i + 1][1]
            result = (((y1 - y2) / (x1 - x2)) * point) + ((y2 * x1) - (y1 * x2)) / (x1 - x2)
            print(bcolors.OKGREEN, "\nThe approximation (interpolation) of the point ", point, " is: ", bcolors.ENDC, round(result, 4))
            flag = 0
    if flag:
        x1 = table_points[0][0]
        x2 = table_points[1][0]
        y1 = table_points[0][1]
        y2 = table_points[1][1]
        m = (y1 - y2) / (x1 - x2)
        result = y1 + m * (point - x1)
        print(bcolors.OKGREEN, "\nThe approximation (extrapolation) of the point ", point, " is: ", bcolors.ENDC, round(result, 4))


if __name__ == '__main__':

    table_points = [(0, 0), (1, 0.8415), (2, 0.9093), (3, 0.1411), (4, -0.7568), (5, -0.9589), (6, -0.2794)]
    x = 1.28
    print(bcolors.OKBLUE, "----------------- Interpolation & Extrapolation Methods -----------------\n", bcolors.ENDC)
    print(bcolors.OKBLUE, "Table Points: ", bcolors.ENDC, table_points)
    print(bcolors.OKBLUE, "Finding an approximation to the point: ", bcolors.ENDC, x)
    linearInterpolation(table_points, x)
    print(bcolors.OKBLUE, "\n---------------------------------------------------------------------------\n", bcolors.ENDC)
