import numpy as np
from numba import njit, float64, int32, complex128
import graph as graph


# questa funzione risolve il sistema di equazioni invece di invertire matrici
@njit()
def solve_matrix(n, lower_diagonal, main_diagonal, upper_diagonal, solution_vector):

    """Solve systems of equations through Thomas Algorithm instead of inverting matrices. It returns
       the same solution of np.linalg.solve"""

    w = np.zeros((n - 1), dtype=np.complex128)
    g = np.zeros((n), dtype=np.complex128)
    result = np.zeros((n), dtype=np.complex128)

    w[0] = upper_diagonal[0] / main_diagonal[0]
    g[0] = solution_vector[0] / main_diagonal[0]

    for i in range(1, n - 1):
        w[i] = upper_diagonal[i] / (main_diagonal[i] - lower_diagonal[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (solution_vector[i] - lower_diagonal[i - 1] * g[i - 1]) / (
            main_diagonal[i] - lower_diagonal[i - 1] * w[i - 1]
        )
    result[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        result[i - 1] = g[i - 1] - w[i - 1] * result[i]
    return result  # restituisce la stessa soluzione che con linalg.solve


def potential(Nx, Ny):
    R = 1
    L = 2
    V0 = 1e9
    ymin = -0.5 * L - R
    ymax = 0.5 * L + R
    xmin = -R
    xmax = R
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    # V = np.zeros((Nx, Ny))
    V0 = 1e10
    par = 0.98
    radius = (par) / 2

    # stadium potential function

    # stadium potential function
    F = np.zeros([Ny, Nx])

    for i in range(Nx):
        for j in range(Ny):
            if abs(x[i]) == R or abs(y[j]) == R + 0.5 * L:
                F[j, i] = V0
            cond_0 = (abs(y[j]) - 0.5 * L) > 0
            cond_1 = np.sqrt((abs(y[j]) - 0.5 * L) ** 2 + x[i] ** 2) >= R
            if cond_0 and cond_1:
                F[j, i] = V0
    return F

# cond_2 = 0.3 < np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2) < radius fa comparire un cilindro vuoto
