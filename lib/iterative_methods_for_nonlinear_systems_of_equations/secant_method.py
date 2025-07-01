from lib.colors import bcolors


def secant_method(f, x0, x1, TOL, N=50):
    """
    Approximates a root of the function f using the Secant method.

    Parameters:
    f (function): The function whose root is to be found.
    x0 (float): First initial guess.
    x1 (float): Second initial guess.
    TOL (float): Tolerance for convergence.
    N (int): Maximum number of iterations (default is 50).

    Returns:
    float: An approximate root of f, or the best estimate after N iterations.

    Notes:
    - The method does not require knowledge of the derivative of f.
    - If the function values at x0 and x1 are equal, the method cannot proceed.
    """
    print("{:<10} {:<15} {:<15} {:<15}".format("Iteration", "xo", "x1", "p"))
    for i in range(N):
        if f(x1) - f(x0) == 0:
            print(" method cannot continue.")
            return

        p = x0 - f(x0) * ((x1 - x0) / (f(x1) - f(x0)))

        if abs(p - x1) < TOL:
            return p  # Procedure completed successfully

        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f}".format(i, x0, x1, p))
        x0 = x1
        x1 = p
    return p


if __name__ == '__main__':
    f = lambda x: x**2 - 5*x + 2
    x0 = 80
    x1 = 100
    TOL = 1e-6
    N = 20
    roots = secant_method(f, x0, x1, TOL, N)
    print(bcolors.OKBLUE, f"\n The equation f(x) has an approximate root at x = {roots}", bcolors.ENDC)
