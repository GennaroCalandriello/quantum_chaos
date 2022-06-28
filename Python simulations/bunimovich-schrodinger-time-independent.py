import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as lnlg
import time
import level_spacing_eval as lvl
import module.graph as grph

Nx, Ny = 300, 300
'''This program resolves Schroedinger equation time independent by vectorizing the matrices of potential and kinetic energy. Details
on theory will be added to repository. This system represents a Bunimovich billiard, which is classically and quantum mechanically chaotic'''

def schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, stadium_pot, neigs, E0=0.0):

    '''Solve in the following steps:
    1. Construct meshgrid
    2. Evaluate potential
    3. Construct the two part of the Hamiltonian, Hx, Hy, and the two identity matrices Ix, Iy for the Kronecker sum 
    4. Find the eigenvalues and eigenfunctions
    Basically one can plot all |psi|^2 eigenvectors obtaining the chaotic structure of the Bunimovich billiard'''

    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]

    V = stadium_pot(x, y)

    Hx = sparse.lil_matrix(2 * np.eye(Nx))
    for i in range(Nx - 1):
        Hx[i, i + 1] = -1
        Hx[i + 1, i] = -1
    Hx = Hx / dx ** 2

    Hy = sparse.lil_matrix(np.eye(Ny))
    for i in range(Ny - 1):
        Hy[i + 1, i] = -1
        Hy[i, i + 1] = -1
    Hy = Hy / dy ** 2

    Ix = sparse.lil_matrix(np.eye(Nx))
    Iy = sparse.lil_matrix(np.eye(Ny))

    # Kronecker sum
    H = sparse.kron(Iy, Hx) + sparse.kron(Hy, Ix)

    H = H.tolil()
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]

    # convert to sparse csc matrix form and calculate eigenvalues
    H = H.tocsc()
    [eigenvalues, eigenstates] = lnlg.eigs(H, k=neigs, sigma=E0)

    return eigenvalues, eigenstates


def twoD_to_oneD(Nx, Ny, F): 

    '''Vectorization function'''

    V = np.zeros(Nx * Ny)
    count = 0
    for i in range(Nx):
        for j in range(Ny):
            V[count] = F[i, j]
            count += 1
    return V


def oneD_to_twoD(Nx, Ny, psi):

    '''Transform vectors in matrices'''

    count = 0
    PSI = np.zeros([Nx, Ny], dtype="complex")
    for i in range(Nx):
        for j in range(Ny):
            PSI[i, j] = psi[count]
            count += 1
    return PSI


def evaluation_wavefunction(xmin, xmax, Nx, ymin, ymax, Ny, potential, neigs, E0):

    '''Evaluate wavefunction and save the eigenvalues and eigenvectors'''

    eigenvalues, eigenstates= schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, potential, neigs, E0)
    
    #saving eingenvalues and eigenstates
    np.savetxt("eigenvalues.txt", eigenvalues)
    # np.savetxt("eigenvectors.txt", eigenstates)


def domain_circle(x, y, x0, y0):

    '''Define a circle domain, obviously'''
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return r


def potential(x, y):

    '''Construct the Bunimovich stadium potential. The B. stadium is constructed with two circles and a rectangle with the proportion
    reported in the image on the repository'''

    x1s, y1s, radius1s = 0.2, 0.5, 0.2
    x2s, y2s, radius2s = 0.8, 0.5, 0.2
    idx_list = []                       #index for imposing V values on the first circle
    idx_list_0 = []                     #index for imposing V values on the second circle
    idx_list_rectangle = []             #index for imposing V values on the intermediate rectangle
    V = np.zeros([Nx, Ny])

    for i in range(Nx):
        for j in range(Ny):
            r1 = domain_circle(x[i], y[j], x1s, y1s)
            r2 = domain_circle(x[i], y[j], x2s, y2s)
            if not r1 < radius1s and not r2 < radius2s:
                idx_list.append((i, j))
            if not r1 < radius1s and not r2 < radius2s:
                idx_list_0.append((i, j))
            if not x[i] < 0.2 and not x[i] > 0.8 and not y[j] < 0.3 and not y[j] > 0.7:
                # if x[i] < 0.2 or x[i] > 0.8 or y[j] < 0.3 or y[j] > 0.7:
                idx_list_rectangle.append((i, j))

    for i, j in idx_list:
        V[i, j] = 1e5
    for i, j in idx_list_0:
        V[i, j] = 1e5
    for i, j in idx_list_rectangle:
        V[i, j] = 0
    grph.static_plot(x, y, V)
    V = twoD_to_oneD(Nx, Ny, V)

    return V


if __name__ == "__main__":
    executor = True

    if executor:
        evaluation_wavefunction(0, 1, Nx, 0, 1, Ny, potential, 4000, 100)

    eigen = np.loadtxt("eigenvalues.txt", dtype=str)
    spacing = lvl.spacing_predictions(eigen, "complex") #calculate the spacing, given the array of eigenvalues
    bins = int(2 + 3.332 * np.log(len(spacing)))
    p = lvl.distribution(spacing, "GOE")                #construct the 3 functions of the 3 ensemble spacing distributions
    p1 = lvl.distribution(spacing, "GSE")
    p2 = lvl.distribution(spacing, "GUE")
    bins = int(5 + 3.332 * np.log(len(spacing)))

    #Plotting the spacing obtained with the 3 ensemble theoretical distributions
    plt.figure()
    plt.hist(
        spacing,
        bins,
        density=True,
        histtype="step",
        fill=False,
        color="b",
        label="Spacing distribution",
    )
    plt.plot(
        np.linspace(min(spacing), max(spacing), len(p)),
        p,
        "--",
        color="green",
        label="GOE prediction",
    )
    plt.plot(
        np.linspace(min(spacing), max(spacing), len(p)),
        p1,
        "--",
        color="red",
        label="GSE prediction",
    )
    plt.plot(
        np.linspace(min(spacing), max(spacing), len(p)),
        p2,
        "--",
        color="blue",
        label="GUE prediction",
    )
    plt.title(f"Spacing of Bunimovich billiard vs RMT predictions", fontsize=20)
    plt.legend()
    plt.xlabel("s", fontsize=14)
    plt.ylabel("P(s)", fontsize=14)
    plt.show()
