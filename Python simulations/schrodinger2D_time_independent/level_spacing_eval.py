import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy import sparse
import matplotlib.pyplot as plt
import time
import math


def spacing_predictions(eigen, spacing_kind, data="float"):

    """Calculate the spacing distributions. Theoretical details reported in the repository"""
    # spacing_kind = 1

    if data == "float":
        eigenfloat = [eigen[i].astype(float) for i in range(len(eigen))]
    if data == "complex":
        eigenfloat = [eigen[i].astype(complex) for i in range(len(eigen))]

    n_evl = len(eigenfloat)
    eigenv = np.array(eigenfloat)
    eigenv = eigenv.real
    eigenv = np.sort(eigenv)  # ordered set of eigenvalues is necessary
    # print(eigenv)

    if spacing_kind == 1:
        spacing = []
        for e in range(1, n_evl - 1):
            spacing.append(
                max(eigenv[e + 1] - eigenv[e], eigenv[e] - eigenv[e - 1])
            )  # si può usare anche min per...
        mean = np.mean(np.array(spacing))
        print("Mean Level Spacing", mean)
        return np.array(spacing) / mean

    # this kind of spacing calculus is adapted for many bodies simulations, it is totally unuseful in this case, ma chissene la lascio lo stesso
    if spacing_kind == 2:
        r_vec = []
        s_vec = np.zeros(len(eigen))
        for i in range(len(eigen) - 1):
            s_vec[i] = eigen[i + 1] - eigen[i]
        mean = np.mean(s_vec)
        for n in range(1, len(s_vec)):
            # r_vec.append(min(s_vec[n], s_vec[n - 1]) / max(s_vec[n], s_vec[n - 1]))
            r_vec.append(min(s_vec[n] / s_vec[n - 1], s_vec[n - 1] / s_vec[n]))
        r_m = np.mean(np.array(r_vec))
        return r_vec / r_m

    if spacing_kind == 3:
        s = []
        for i in range(1, n_evl - 1):
            s.append(min(eigenv[i + 1] - eigenv[i], eigenv[i] - eigenv[i - 1]))
        return s / np.mean(np.array(s))

    if spacing_kind == 4:
        s = []
        for i in range(1, n_evl - 2):
            s.append(max(eigenv[i + 2] - eigenv[i + 1], eigenv[i + 1] - eigenv[i]))
            mean = np.mean(np.array(s))
        return s / mean


def distribution(sp, kind):

    """Plot theoretical distributions of GSE, GOE, GUE ensemble distributions picking the min and max values of the spacing array
    calculated in the main program"""
    s = np.linspace(0, max(sp), len(sp))
    p = np.zeros(len(s))

    if kind == "GOE":
        for i in range(len(p)):
            p[i] = np.pi / 2 * s[i] * np.exp(-np.pi / 4 * s[i] ** 2)

    if kind == "GUE":
        for i in range(len(p)):
            p[i] = (32 / np.pi ** 2) * s[i] ** 2 * np.exp(-4 / np.pi * s[i] ** 2)

    if kind == "GSE":
        for i in range(len(p)):
            p[i] = (
                2 ** 18
                / (3 ** 6 * np.pi ** 3)
                * s[i] ** 4
                * np.exp(-(64 / (9 * np.pi)) * s[i] ** 2)
            )
    if kind == "Poisson":
        for i in range(len(p)):
            p[i] = np.exp(-s[i])

    if kind == "GOE FN":  # lasciamo perdere va...

        a = (27 / 8) * np.pi
        for i in range(len(p)):
            p[i] = (
                (a / np.pi)
                * s[i]
                * np.exp(-2 * a * s[i] ** 2)
                * (
                    np.pi
                    * np.exp((3 * a / 2) * s[i] ** 2)
                    * (a * s[i] ** 2 - 3)
                    * (
                        math.erf(np.sqrt(a / 6) * s[i])
                        - math.erf(np.sqrt(3 * a / 2) * s[i])
                        + np.sqrt(6 * np.pi * a)
                        * s[i]
                        * (np.exp((4 * a / 3) * s[i] ** 2) - 3)
                    )
                )
            )

    return p


def density_of_states(eigenvalues):
    E0 = eigenvalues[0]
    dn_dE = []
    rho = []
    N = len(eigenvalues)
    prefactor = 1 / (np.pi * np.sqrt(N) * E0)
    for e in eigenvalues:
        dn_dE.append(1 / (2 * E0) * (np.sqrt(1 / (e / E0))))
        rho.append(prefactor * np.sqrt(1 - (e ** 2 / (4 * E0 ** 2 * N))))

    return np.array(dn_dE)


if __name__ == "__main__":

    eig = np.loadtxt("eigenvalues_Henon.txt", dtype=complex)
    dndE = density_of_states(eig.real)
    plt.plot(eig / eig[0], dndE)
    plt.show()
