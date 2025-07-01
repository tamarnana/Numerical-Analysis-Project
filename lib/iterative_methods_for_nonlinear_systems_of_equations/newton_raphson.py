from lib.colors import bcolors


def newton_raphson(f, df, p0, TOL, N=50):
    """
    Approximates a root of the function f using the Newton-Raphson method.

    Parameters:
    f (function): The function whose root is to be found.
    df (function): Derivative of the function f.
    p0 (float): Initial guess for the root.
    TOL (float): Tolerance for stopping criterion.
    N (int): Maximum number of iterations (default is 50).

    Returns:
    float: An approximate root of f.

    Notes:
    - If the derivative at any point becomes zero, the method stops early.
    - The method may fail to converge if f or df is not well-behaved near the root.
    """
    print("{:<10} {:<15} {:<15} ".format("Iteration", "po", "p1"))
    for i in range(N):
        if df(p0) == 0:
            print("Derivative is zero at p0, method cannot continue.")
            return

        p = p0 - f(p0) / df(p0)

        if abs(p - p0) < TOL:
            return p  # Procedure completed successfully

        print("{:<10} {:<15.9f} {:<15.9f} ".format(i, p0, p))
        p0 = p
    return p


if __name__ == '__main__':
    f = lambda x: x**3 - 3*x**2
    df = lambda x: 3*x**2 - 6*x
    p0 = -5
    TOL = 1e-6
    N = 100
    roots = newton_raphson(f, df, p0, TOL, N)
    print(bcolors.OKBLUE, "\nThe equation f(x) has an approximate root at x = {:<15.9f} ".format(roots), bcolors.ENDC)
