import numpy as np
from numba import njit, float64, int32, complex128


# @njit(
#     complex128[:](int32, complex128[:], complex128[:], complex128[:], complex128[:]),
#     fastmath=True,
#     cache=True
# )  # questa funzione risolve il sistema di equazioni invece di invertire matrici
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
