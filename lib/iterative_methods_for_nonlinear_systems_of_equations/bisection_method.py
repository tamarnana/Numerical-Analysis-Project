import math
import numpy as np
from lib.colors import bcolors


def max_steps(a, b, err):
    """
    Calculates the minimum number of iterations required to achieve the desired accuracy
    in the bisection method.

    Parameters:
    a (float): Start of the interval.
    b (float): End of the interval.
    err (float): Tolerable error (epsilon).

    Returns:
    int: Estimated maximum number of iterations required.
    """
    s = int(np.floor(-np.log2(err / (b - a)) / np.log2(2) - 1))
    return s


def bisection_method(f, a, b, tol=1e-6):
    """
    Finds an approximate root of the function f in the interval [a, b] using
    the bisection method, a reliable root-finding technique for continuous functions.

    Parameters:
    f (function): Continuous function such that f(a)*f(b) < 0.
    a (float): Start of the interval.
    b (float): End of the interval.
    tol (float): Tolerable error. Default is 1e-6.

    Returns:
    float: Approximate root of the function f in the interval [a, b].

    Raises:
    Exception: If f(a) and f(b) have the same sign (i.e., no root guaranteed in the interval).
    """
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The scalars a and b do not bound a root")

    c, k = 0, 0
    steps = max_steps(a, b, tol)

    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))

    while abs(b - a) > tol and k < steps:
        c = a + (b - a) / 2  # Midpoint
        if f(c) == 0:
            return c  # Exact root found

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            k, a, b, f(a), f(b), c, f(c)))
        k += 1

    return c


if __name__ == '__main__':
    f = lambda x: x**2 - 4 * math.sin(x)
    roots = bisection_method(f, 1, 3)
    print(bcolors.OKBLUE, f"\nThe equation f(x) has an approximate root at x = {roots}", bcolors.ENDC)
