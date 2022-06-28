import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy import sparse
import matplotlib.pyplot as plt
import time


def spacing_predictions(eigen, data="float"):

    '''Calculate the spacing distributions. Theoretical details reported in the repository'''
    spacing_kind = 1

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
        print(spacing)
        mean = np.mean(np.array(spacing))
        return np.array(spacing) / mean

    #this kind of spacing calculus is adapted for many bodies simulations, it is totally unuseful in this case, ma chissene la lascio lo stesso
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
    
    if spacing_kind==3:
        s=[]
        for i in range(1, n_evl-1):
            s.append(eigenv[i+1]-eigenv[i])
        m=np.mean(np.array(s))
        return s/m


def distribution(sp, kind):

    '''Plot theoretical distributions of GSE, GOE, GUE ensemble distributions picking the min and max values of the spacing array
    calculated in the main program'''
    
    s = np.linspace(min(sp), max(sp), len(sp))
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

    return p
