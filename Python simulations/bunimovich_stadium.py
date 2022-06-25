import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as lnlg


def schrodinger2D(
    xmin, xmax, Nx, ymin, ymax, Ny, Vfun2D, params, neigs, E0=0.0, findpsi=False
):
    x = np.linspace(xmin, xmax, Nx)
    dx = x[1] - x[0]
    y = np.linspace(ymin, ymax, Ny)
    dy = y[1] - y[0]

    V = Vfun2D(x, y, params)

    # derivative in x direction (?)
    Hx = sparse.lil_matrix(2 * np.eye(Nx))
    for i in range(Nx - 1):
        Hx[i, i + 1] = -1
        Hx[i + 1, i] = -1
    Hx = Hx / dx ** 2
    print(Hx)

    Hy = sparse.lil_matrix(np.eye(Ny))
    for i in range(Ny - 1):
        Hy[i, i + 1] = -1
        Hy[i + 1, i] = -1
    Hy = Hy / dy ** 2

    # combining Hilbert space with Krokecker product
    Ix = sparse.lil_matrix(np.eye(Nx))
    Iy = sparse.lil_matrix(np.eye(Ny))
    H = sparse.kron(Iy, Hx) + sparse.kron(Hy, Ix)

    # reconvert to sparse matrix lil form
    H = H.tolil()

    # adding potential energy
    for i in range(Nx * Ny):
        H[i, i] = H[i, i] + V[i]

    # convert to sparse matrix csc form and calculate eigenvalues
    H = H.tocsc()
    [eigenvalue, eigenvector] = lnlg.eigs(H, k=neigs, sigma=E0)

    if findpsi == False:
        return eigenvalue
    else:
        return eigenvalue, eigenvector, x, y


def twoD_to_oneD(Nx, Ny, F):

    V = np.zeros(Nx * Ny)
    vindex = 0

    for i in range(Ny):
        for j in range(Nx):
            V[vindex] = F[i, j]
            vindex += 1
    return V


def oneD_to_twoD(Nx, Ny, psi):
    vindex = 0
    PSI = np.zeros([Ny, Nx], dtype="complex")
    for i in range(Ny):
        for j in range(Nx):
            PSI[i, j] = psi[vindex]
            vindex += 1
    return PSI


def eval_wavefunctions(
    xmin, xmax, Nx, ymin, ymax, Ny, Vfun, params, neigs, E0, findpsi
):
    H = schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, Vfun, params, neigs, E0, findpsi)
    # get eigenenergies:
    evl = H[0]
    indices = np.argsort(evl)
    print("Energies eigenvalue: ")
    for i, j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, np.real(j)))

    # eigenvectors
    evt = H[1]

    plt.figure(figsize=(8, 8))
    # unpack the vector into 2 dimensions for plotting
    for n in range(neigs):
        psi = evt[:, n]
        PSI = oneD_to_twoD(Nx, Ny, psi)
        PSI = np.abs(PSI) ** 2
        plt.subplot(2, int(neigs / 2), n + 1)
        plt.pcolormesh(np.flipud(PSI), cmap="terrain")
        plt.axis("equal")
        plt.axis("off")
    plt.show()
    for n in range(neigs):
        psi = evt[:, n]
        PSI = oneD_to_twoD(Nx, Ny, psi)
        PSI = np.abs(PSI) ** 2
        plt.pcolormesh(np.flipud(PSI), cmap="terrain")
        plt.axis("equal")
        plt.axis("off")
        plt.show()


def Vfun(X, Y, params):
    R = params[0]
    L = params[1]
    V0 = params[2]

    # stadium potential function
    Nx = len(X)
    Ny = len(Y)
    [x, y] = np.meshgrid(X, Y)
    F = np.zeros([Ny, Nx])

    for i in range(Nx):
        for j in range(Ny):
            if abs(X[i]) == R or abs(Y[j]) == R + 0.5 * L:
                F[j, i] = V0
            cond_0 = (abs(Y[j]) - 0.5 * L) > 0
            cond_1 = np.sqrt((abs(Y[j]) - 0.5 * L) ** 2 + X[i] ** 2) >= R
            if cond_0 and cond_1:
                F[j, i] = V0

    # fold the 2D matrix to a 1D array
    V = twoD_to_oneD(Nx, Ny, F)
    return V


def stadium_wavefunctions_plot(R=1, L=2, V0=1e6, neigs=6, E0=100):
    # R=stadium radius
    # L=stadium length
    # V0=stadium wall potential
    ymin = -0.5 * L - R
    ymax = 0.5 * L + R
    xmin = -R
    xmax = R
    params = [R, L, V0]

    Ny = 50
    Nx = int(Ny * 2 * R / (2.0 * R + L))

    eval_wavefunctions(
        xmin, xmax, Nx, ymin, ymax, Ny, Vfun, params, neigs, E0, findpsi=True
    )


stadium_wavefunctions_plot(1, 2, 1e6, 4, 1000)

