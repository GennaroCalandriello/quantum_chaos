import numpy as np
import matplotlib.pyplot as plt
from numba import njit, complex128
import scipy.linalg as SLA
import os
import shutil
import random

import function as fnc
import graph as graph

Nx = 800
Ny = Nx
tmax = 3000
xmax = 1.0
ymax = 1.0
dt = 4e-5
x = np.linspace(0, xmax, Nx)
y = np.linspace(0, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
D = 0.4
alphac = D * dt * 1j / 2 / dx ** 2
alphar = dt * 1 / 4 / dx ** 2

pot_list = [
    "mickey",
    "Sinai",
    "Bunimovich",
    "Ehrenfest",
    "Anderson",
    "Henon",
    "cardioid",
]
graph_potential = False


def domain_circle(x, y, x0, y0):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return r


def domain_cardioid(x, y, x0, y0):

    """Define a cardioid domain"""
    # cite as: Cardioid. Encyclopedia of Mathematics. URL: http://encyclopediaofmath.org/index.php?title=Cardioid&oldid=32369
    r = ((x - x0) ** 2 + (y - y0) ** 2) / (
        2 * np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - 2 * (x - x0)
    )
    return r


def potential(V, x, y, kind=pot_list[1]):

    """Function defining the shape of the potential. Potentials treated are: mickey, Sinai, Bunimovich stadium,
    Ehrenfest, Henon, Anderson, cardioid"""

    idx_list = []
    idx_list_0 = []
    idx_list_rectangle = []

    # for mikey potential
    x0, y0, radius = 0.5, 0.5, 0.15
    x02, y02, radius2 = 0.5, 0.3, 0.08
    x03, y03, radius3 = 0.3, 0.5, 0.08

    x1s, y1s, radius1s = 0.2, 0.5, 0.2
    x2s, y2s, radius2s = 0.8, 0.5, 0.2

    if kind == "mickey":

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

    if kind == "Sinai":
        for i in range(Nx):
            for j in range(Ny):
                r = domain_circle(x[i], y[j], x0, y0)
                if r < radius:
                    idx_list.append((i, j))

        for i, j in idx_list:
            V[i, j] = 1e10
        V[0, :] = 1e10
        V[:, 0] = 1e10
        V[Nx - 1, :] = 1e10
        V[:, Ny - 1] = 1e10

    if kind == "Bunimovich":

        # non toccare dioporco che poi non si trova

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
                    # V[i, j] = 1e6
                    idx_list_rectangle.append((i, j))

        for i, j in idx_list:
            V[i, j] = 1e6
        for i, j in idx_list_0:
            V[i, j] = 1e6
        for i, j in idx_list_rectangle:
            V[i, j] = 0

    if kind == "Ehrenfest":
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

    if kind == "Anderson":

        """Random potential in a certain region to visualize the Anderson Localization"""

        rad = 0.15
        radmin = 0.1
        for i in range(Nx):
            for j in range(Ny):
                if i % 10 and j % 10 == 0:
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

    if kind == "Henon":

        lamb = 1.0
        sh = 0.7
        for i in range(Nx):
            for j in range(Ny):
                # V[i, j] = (
                #     (1 / 2) * (x[i] ** 2 + y[j] ** 2)
                #     + lamb * ((x[i] ** 2) * y[j] - (y[j] ** 3) / 3)
                # )
                V[i, j] = (
                    (1 / 2) * ((i * dx - sh) ** 2 + (j * dy - sh) ** 2)
                    + lamb
                    * (((dx * i - sh) ** 2) * (dy * j - sh) - ((dy * j - sh) ** 3) / 3)
                ) * 1e3

        V[0, :] = 1e10
        V[:, 0] = 1e10
        V[Nx - 1, :] = 1e10
        V[:, Ny - 1] = 1e10

    if kind == "cardioid":
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

    return V


###--------------------Initial coherent states-----------------------------###


def initial_coherent_state(psi, xy0, pxpy0, stdv):

    """Function to initialize the coherent state"""

    for ii in range(Nx):
        for jj in range(Ny):
            psi[ii, jj] = np.exp(
                1j * (pxpy0[0] * (ii - xy0[0]) + pxpy0[1] * (jj - xy0[1])) * dx
                - ((ii - xy0[0]) ** 2 + (jj - xy0[1]) ** 2) * dx ** 2 / (2 * stdv ** 2)
            )
    psi = psi / SLA.norm(psi)

    return psi


# parameters of psi and psi2
xy0, pxpy0, stdv = [Nx / 6, Ny / 2], [Nx / 3, 0], 0.07
xy02, pxpy02, stdv2 = [Nx / 2, Ny / 2], [-3 * Nx / 7, -4 * Nx / 9], 0.06

# initializing 2D array of psi and V
psi = np.zeros((Nx, Ny), dtype=np.complex128)
psi2 = psi.copy()
V = np.zeros((Nx, Ny))

psi = initial_coherent_state(psi, xy0, pxpy0, stdv)
psi2 = initial_coherent_state(psi2, xy02, pxpy02, stdv2)

V = potential(V, x, y)

if graph_potential:
    graph.static_plot(x, y, np.abs(psi) ** 2, 0)

###-----------upper and lower diagonals---------------------------###

up_diag1 = -np.ones(Nx - 1, dtype=np.complex128) * alphac
low_diag1 = up_diag1[::-1]


up_diag2 = -np.ones(Nx - 1, dtype=np.complex128) * alphac
low_diag2 = up_diag2[::-1]

###--------------------------------------------------------------------
@njit()
def ADI_x_y(psi, main, step):

    """ADI function for implicit on x and explicit on y"""

    for i in range(1, Nx):
        for j in range(1, Ny):
            main[j] = 1 + 2 * alphac + 1j * (dt / 2) * V[i, j]
        for j in range(1, Ny - 1):
            step[j] = (1 - 2 * alphac - 1j * (dt / 2) * V[i, j]) * psi[
                i, j
            ] + alphac * (psi[i, min(Nx - 1, j + 1)] + psi[i, max(0, j - 1)])

        temp = fnc.solve_matrix(Nx, low_diag1, main, up_diag1, step)
        for j in range(1, Nx):
            psi[i, j] = temp[j]
    return psi


@njit()
def ADI_y_x(psi, main, step):

    """ADI function for implicit on y and explicit on x"""

    for j in range(1, Nx):
        for i in range(1, Ny):
            main[i] = 1 + 2 * alphac + 1j * dt / 2 * V[i, j]
        for i in range(1, Nx):
            step[i] = (1 - 2 * alphac - 1j * dt / 2 * V[i, j]) * psi[i, j] + alphac * (
                psi[min(Nx - 1, i + 1), j] + psi[max(0, i - 1), j]
            )

        temp = fnc.solve_matrix(Nx, low_diag2, main, up_diag2, step)
        for i in range(1, Nx):
            psi[i, j] = temp[i]
    return psi


###-------------------------------------------------------------------

if __name__ == "__main__":

    INTV = 6

    psitilde = psi + psi2

    exe = True
    vtk = True
    toobig = False  # if meshgrids are too small and the array psi_total is too big, write wtk files at each time%INTV step without saving in the 3D array

    if not toobig:
        psi_total = np.zeros((int(tmax / INTV) + 3, Nx, Ny)).astype(complex)

    if exe:

        if vtk:
            path = "datacard/datavtk"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        # graph.static_plot(x, y, np.absolute(psi) ** 2, 0)
        c = 0
        for t in range(tmax):

            """Time evolution for loops"""

            main = np.ones(Nx).astype(complex)
            step = np.zeros(Nx).astype(complex)
            step1 = step.copy()
            if t % INTV == 0:
                print(f"time step: {t}")

            psi = ADI_x_y(psi, main, step)
            main = np.ones(Nx).astype(complex)
            psi = ADI_y_x(psi, main, step1)
            psi = psi / SLA.norm(psi)

            if toobig:
                if t % INTV == 0:
                    print(f"Writing vtk animation for {t}")
                    graph.writeVtk(c, np.abs(psi) ** 2, Nx, dx, path)
                    c += 1

            if not toobig:
                if t % INTV == 0:
                    psi_total[c] = psi
                    c += 1

        # graph.animate_matplotlib(x, y, np.absolute(psi_total) ** 2)

        if not toobig:
            print("Writing vtk animation")
            for i in range(len(psi_total)):
                graph.writeVtk(i, np.abs(psi_total[i]) ** 2, Nx, dx, path)

