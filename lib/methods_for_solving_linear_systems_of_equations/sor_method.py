def sor(m, w=1.25, x0=None, eps=1e-5, max_iteration=100):
    """
    Solves a system of linear equations using the Successive Over-Relaxation (SOR) method.

    Parameters
    ----------
    m : list of list of floats
        Augmented matrix of coefficients and constants (size n x n+1).
    w : float, optional
        Relaxation factor (weight), default is 1.25.
    x0 : list of floats, optional
        Initial guess vector. If None, initialized to zero vector.
    eps : float, optional
        Error tolerance for convergence, default is 1e-5.
    max_iteration : int, optional
        Maximum number of iterations, default is 100.

    Returns
    -------
    list of floats
        Approximate solution vector of the system.

    Raises
    ------
    ValueError
        If the solution does not converge within max_iteration iterations.

    Notes
    -----
    Prints the intermediate iterative results at each iteration.
    """
    n = len(m)
    x0 = [0] * n if x0 is None else x0
    x1 = x0[:]

    for __ in range(max_iteration):
        for i in range(n):
            s = sum(-m[i][j] * x1[j] for j in range(n) if i != j)
            x1[i] = w * (m[i][n] + s) / m[i][i] + (1 - w) * x0[i]

        if all(abs(x1[i] - x0[i]) < eps for i in range(n)):
            return x1
        x0 = x1[:]
        print("The iterative result:", x1)
    raise ValueError('Solution does not converge')


if __name__ == '__main__':
    m = [[4, 3.2, 0.5, 9.2], [2.2, 3, -0.3, 0.9], [-3.1, -0.2, 4, 7]]
    print(sor(m))
