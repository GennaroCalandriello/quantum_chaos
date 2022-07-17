import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as lnlg
import time
import level_spacing_eval as lvl
import module.graph as grph
import random


potentials_list = [
    "square",  # 0
    "Sinai",  # 1
    "Bunimovich",  # 2
    "mickey",  # 3
    "Ehrenfest",  # 4
    "Anderson",  # 5
    "Henon",  # 6
    "cardioid",  # 7
    "harm_osc",  # 8
]


kindpot = potentials_list[2]
plot = False
executor = False
plot_potential = "n"

Nx, Ny = 500, 500
xmin, xmax, ymin, ymax = 0, 1, 0, 1
n_eigen = 900  # how much eigenvalues and eigenvectors do you want?
E0 = 0

"""This program resolves Schroedinger equation time independent by vectorizing the matrices of potential and kinetic energy. Details
on theory will be added to repository. This system represents a Bunimovich billiard, which is classically and quantum mechanically chaotic"""


def schrodinger2D(xmin, xmax, Nx, ymin, ymax, Ny, stadium_pot, neigs, E0):

    """Solve in the following steps:
    1. Construct meshgrid
    2. Evaluate potential
    3. Construct the two part of the Hamiltonian, Hx, Hy, and the two identity matrices Ix, Iy for the Kronecker sum 
    4. Find the eigenvalues and eigenfunctions
    Basically one can plot all |psi|^2 eigenvectors obtaining the chaotic structure for various potentials"""

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

    """Vectorization function"""

    V = np.zeros(Nx * Ny)
    count = 0
    for i in range(Nx):
        for j in range(Ny):
            V[count] = F[i, j]
            count += 1
    return V


def oneD_to_twoD(Nx, Ny, psi):

    """Transform vectors in matrices"""

    count = 0
    PSI = np.zeros([Nx, Ny], dtype="complex")
    for i in range(Nx):
        for j in range(Ny):
            PSI[i, j] = psi[count]
            count += 1
    return PSI


def evaluation_wavefunction(xmin, xmax, Nx, ymin, ymax, Ny, potential, neigs, E0):

    """Evaluate wavefunction and save the eigenvalues and eigenvectors"""

    eigenvalues, eigenstates = schrodinger2D(
        xmin, xmax, Nx, ymin, ymax, Ny, potential, neigs, E0
    )

    # saving eingenvalues and eigenstates
    np.savetxt(f"eigenvalues_{kindpot}.txt", eigenvalues)
    # np.savetxt(f"eigenvectors_{kindpot}.txt", eigenstates)

    if plot:
        for n in range(neigs):
            psi = eigenstates[:, n]
            PSI = oneD_to_twoD(Nx, Ny, psi)
            PSI = np.abs(PSI) ** 2
            plt.pcolormesh(np.flipud(PSI), cmap="seismic")
            plt.axis("equal")
            plt.axis("off")
            plt.show()


def domain_circle(x, y, x0, y0):

    """Define a circle domain, obviously"""
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return r


def domain_cardioid(x, y, x0, y0):

    """Define a cardioid domain"""
    # cite as: Cardioid. Encyclopedia of Mathematics. URL: http://encyclopediaofmath.org/index.php?title=Cardioid&oldid=32369
    r = ((x - x0) ** 2 + (y - y0) ** 2) / (
        2 * np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - 2 * (x - x0)
    )
    return r


def potential(x, y):

    """Construct the Bunimovich & Sinai stadium potential. The B. stadium is constructed with two circles and a rectangle with the proportion
    reported in the image on the repository"""

    x1s, y1s, radius1s = 0.2, 0.5, 0.2
    x2s, y2s, radius2s = 0.8, 0.5, 0.2
    x0, y0, radius = 0.5, 0.5, 0.15
    x03, y03, radius3 = 0.3, 0.5, 0.08
    x02, y02, radius2 = 0.5, 0.3, 0.08

    idx_list = []  # index for imposing V values on the first circle
    idx_list_0 = []  # index for imposing V values on the second circle
    idx_list_rectangle = []  # index for imposing V values on the intermediate rectangle
    V = np.zeros([Nx, Ny])

    if kindpot == "square":

        V[1, :] = 1e10
        V[:, 1] = 1e10
        V[Nx - 1, :] = 1e10
        V[:, Ny - 1] = 1e10

    if kindpot == "Sinai":
        for i in range(Nx):
            for j in range(Ny):
                r = domain_circle(x[i], y[j], x0, y0)
                if r < radius:
                    idx_list.append((i, j))

        for i, j in idx_list:
            V[i, j] = 1e10

        V[1, :] = 1e10
        V[:, 1] = 1e10
        V[Nx - 1, :] = 1e10
        V[:, Ny - 1] = 1e10

    if kindpot == "Bunimovich":  # non toccare porcodd*o che sennò va tutto a p*ttane
        for i in range(Nx):
            for j in range(Ny):
                r1 = domain_circle(x[i], y[j], x1s, y1s)
                r2 = domain_circle(x[i], y[j], x2s, y2s)
                if not r1 < radius1s and not r2 < radius2s:
                    idx_list.append((i, j))
                if not r1 < radius1s and not r2 < radius2s:
                    idx_list_0.append((i, j))
                if (
                    not x[i] < 0.2
                    and not x[i] > 0.8
                    and not y[j] < 0.3
                    and not y[j] > 0.7
                ):
                    idx_list_rectangle.append((i, j))

        for i, j in idx_list:
            V[i, j] = 1e10
        for i, j in idx_list_0:
            V[i, j] = 1e10
        for i, j in idx_list_rectangle:
            V[i, j] = 0

    if kindpot == "mickey":
        for i in range(Nx):
            for j in range(Ny):
                r = domain_circle(x[i], y[j], x0, y0)
                r1 = domain_circle(x[i], y[j], x02, y02)
                r2 = domain_circle(x[i], y[j], x03, y03)
                if r < radius or r1 < radius2 or r2 < radius3:  # r1 < radius2:
                    idx_list.append((i, j))

        for i, j in idx_list:
            V[i, j] = 1e10
        V[0, :] = 1e10
        V[:, 0] = 1e10
        V[Nx - 1, :] = 1e10
        V[:, Ny - 1] = 1e10

    if kindpot == "Ehrenfest":
        for i in range(Nx):
            for j in range(Ny):
                r1 = domain_circle(x[i], y[j], x1s, y1s)
                r2 = domain_circle(x[i], y[j], x2s, y2s)
                if not r1 < radius1s and not r2 < radius2s:
                    idx_list.append((i, j))
                if not r1 < radius1s and not r2 < radius2s:
                    idx_list_0.append((i, j))
                if (
                    not x[i] < 0.2
                    and not x[i] > 0.8
                    and not y[j] < 0.45
                    and not y[j] > 0.55
                ):
                    idx_list_rectangle.append((i, j))

        for i, j in idx_list:
            V[i, j] = 1e10
        for i, j in idx_list_0:
            V[i, j] = 1e10
        for i, j in idx_list_rectangle:
            V[i, j] = 0

    if kindpot == "Anderson":

        """Random potential in a certain region to visualize the Anderson Localization"""

        rad = 0.15
        radmin = 0.1
        for i in range(Nx):
            for j in range(Ny):
                # if i % 10 and j % 10 == 0:
                xs, ys = random.uniform(0.44, 0.6), random.uniform(0, 1)
                r = domain_circle(x[i], y[j], xs, ys)
                if radmin < r < rad:
                    idx_list.append((i, j))

        for i, j in idx_list:
            V[i, j] = 1e10

        V[0, :] = 1e10
        V[:, 0] = 1e10
        V[Nx - 1, :] = 1e10
        V[:, Ny - 1] = 1e10

    if kindpot == "Henon":
        lamb = 1.0
        sh = 0
        for i in range(Nx):
            for j in range(Ny):
                V[i, j] = (
                    (1 / 2) * ((x[i] - sh) ** 2 + (y[j] - sh) ** 2)
                    + lamb * (((x[i] - sh) ** 2) * (y[j] - sh) - ((y[j] - sh) ** 3) / 3)
                ) * 2e2

        V[0, :] = 1e10
        V[:, 0] = 1e10
        V[Nx - 1, :] = 1e10
        V[:, Ny - 1] = 1e10

    if kindpot == "cardioid":

        """Cardioid potential, impose xmin=ymin=0 & xmax=ymax=1"""

        rad, x0, y0 = 0.16, 0.67, 0.5  # se li modifichi te la vedi tu eh!
        card_list = []
        for i in range(Nx):
            for j in range(Ny):
                r = domain_cardioid(x[i], y[j], x0, y0)
                if rad < r:
                    card_list.append((i, j))

        for i, j in card_list:
            V[i, j] = 1e10

    if kindpot == "harm_osc":
        k = 0.4
        m = 0.1
        for i in range(Nx):
            for j in range(Ny):
                V[i, j] = (1 / 2) * (k / m) * x[i] ** 2 + y[j] ** 2 / (2 * m)

    if plot_potential == "y":
        grph.static_plot(x, y, V)
    V = twoD_to_oneD(Nx, Ny, V)

    return V


if __name__ == "__main__":

    start = time.time()

    if executor:

        evaluation_wavefunction(
            xmin, xmax, Nx, ymin, ymax, Ny, potential, n_eigen, E0
        )  # here

    print(f"Total time {np.abs(round(time.time()-start))}")

    # eigen = np.loadtxt(f"eigenvalues_{kindpot}.txt", dtype=complex)
    eigen = np.loadtxt(f"unfolded_spectrum_{kindpot}.txt", dtype=complex)

    spacing = lvl.spacing_predictions(
        eigen, 1, "complex"
    )  # calculate the spacing, given the array of eigenvalues
    spacing1 = lvl.spacing_predictions(eigen, 2, "complex",)
    p = lvl.distribution(
        spacing, "GOE"
    )  # construct the 3 functions of the 3 ensemble spacing distributions

    p1 = lvl.distribution(spacing, "GSE")
    p2 = lvl.distribution(spacing, "GUE")
    p3 = lvl.distribution(spacing, "Poisson")

    bins = int(1 + 3.332 * np.log(len(spacing)))

    # Plotting the spacing obtained with the 3 ensemble theoretical distributions

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
    # plt.plot(
    #     np.linspace(min(spacing), max(spacing), len(p)),
    #     p1,
    #     "-.",
    #     color="red",
    #     label="GSE prediction",
    # )
    # plt.plot(
    #     np.linspace(min(spacing), max(spacing), len(p)),
    #     p2,
    #     "--",
    #     color="blue",
    #     label="GUE prediction",
    # )
    plt.plot(
        np.linspace(min(spacing), max(spacing), len(p)),
        p3,
        "-.",
        color="yellow",
        label="Poisson prediction",
    )
    plt.title(f"Spacing of {kindpot} billiard vs RMT predictions", fontsize=20)
    plt.legend()
    plt.xlabel("s", fontsize=14)
    plt.ylabel("P(s)", fontsize=14)
    plt.show()
