import numpy as np
import matplotlib.pyplot as plt
from numba import njit, complex128
import scipy.linalg as SLA
import os
import shutil

import function as fnc
import graph as graph

Nx = 400
Ny = Nx
tmax = 1000
xmax = 1.0
ymax = 1.0
dt = 5e-6
x = np.linspace(0, xmax, Nx)
y = np.linspace(0, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
alphac = dt * 1j / 2 / dx ** 2
alphar = dt * 1 / 4 / dx ** 2


V = np.zeros((Nx, Ny))
V[120:124, 90::] = 1e15
V[120:124, 86:90] = 0
V[120:124, 76:86] = 1e15
V[120:124, 72:76] = 0
V[120:124, 0:72] = 1e15
# graph.static_plot(x, y, V, 0)

###----------------------------to isolate a circle domain:-------------###
### Sinai potential
# x0, y0, radius = 0.5, 0.5, 0.2


# def domain(x, y):
#     r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
#     return r


# idx_list = []

# for i in range(Nx):
#     for j in range(Ny):
#         r = domain(x[i], y[j])
#         if r < radius:
#             idx_list.append((i, j))


# for i, j in idx_list:
#     V[i, j] = 0
###
###------------------------------------------------------------------------###
###--------------------Initial coherent states-----------------------------###
psi = np.zeros((Nx, Ny), dtype=np.complex128)
psi2 = psi.copy()
### parameters of initial momenta and positions
stdv = 0.07
xy0 = [Nx / 8, Ny / 8]
pxpy0 = [-4 * Nx / 13, 6 * Nx / 10]

stdv2 = 0.09
xy02 = [Nx / 2, Ny / 2]
pxpy02 = [-3 * Nx / 17, 4 * Nx / 15]

for ii in range(Nx):
    for jj in range(Ny):
        psi[ii, jj] = np.exp(
            1j * (pxpy0[0] * (ii - xy0[0]) + pxpy0[1] * (jj - xy0[1])) * dx
            - ((ii - xy0[0]) ** 2 + (jj - xy0[1]) ** 2) * dx ** 2 / (2 * stdv ** 2)
        )

psi = psi / SLA.norm(psi)

for ii in range(Nx):
    for jj in range(Ny):
        psi2[ii, jj] = np.exp(
            1j * (pxpy02[0] * (ii - xy02[0]) + pxpy02[1] * (jj - xy02[1])) * dx
            - ((ii - xy02[0]) ** 2 + (jj - xy02[1]) ** 2) * dx ** 2 / (2 * stdv2 ** 2)
        )

psi2 = psi2 / SLA.norm(psi2)
###---------------------------------------------------------------###
###-----------upper and lower diagonals---------------------------###

up_diag1 = -np.ones(Nx - 1, dtype=np.complex128) * alphac
low_diag1 = up_diag1[::-1]


up_diag2 = -np.ones(Nx - 1, dtype=np.complex128) * alphac
low_diag2 = up_diag2[::-1]

###--------------------------------------------------------------------
@njit()
def ADI_x_y(psi, main, step):

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            main[j] = 1 + 2 * alphac + 1j * (dt / 2) * V[i, j] / 2
        for j in range(1, Ny - 1):
            step[j] = (1 - 2 * alphac) * psi[i, j] + alphac * (
                psi[i, min(Nx - 1, j + 1)] + psi[i, max(0, j - 1)]
            )

        temp = fnc.solve_matrix(Nx, low_diag1, main, up_diag1, step)
        for j in range(Nx):
            psi[i, j] = temp[j]
    return psi


###############ho messo 1j vicino a V[i, j]
@njit()
def ADI_y_x(psi, main, step):

    for j in range(1, Nx - 1):
        for i in range(1, Ny - 1):
            main[i] = 1 + 2 * alphac + 1j * dt / 2 * V[j, i] / 2
        for i in range(1, Nx - 1):
            step[i] = (1 - 2 * alphac) * psi[i, j] + alphac * (
                psi[min(Nx - 1, i + 1), j] + psi[max(0, i - 1), j]
            )

        temp = fnc.solve_matrix(Nx, low_diag2, main, up_diag2, step)
        for i in range(Nx):
            psi[i, j] = temp[i]
    return psi


###-------------------------------------------------------------------

if __name__ == "__main__":

    INTV = 7
    psi_total = np.zeros((int(tmax / INTV) + 3, Nx, Ny)).astype(complex)
    psitilde = psi + psi2

    exe = True
    vtk = True

    if exe:
        psi = psitilde
        # graph.static_plot(x, y, np.absolute(psi) ** 2, 0)
        c = 0
        for t in range(tmax):
            main = np.ones(Nx).astype(complex)
            step = np.zeros(Nx).astype(complex)
            step1 = step.copy()
            if t % INTV == 0:
                print(f"time step: {t}")

            psi = ADI_x_y(psi, main, step)
            main = np.ones(Nx).astype(complex)
            psi = ADI_y_x(psi, main, step1)
            psi = psi / SLA.norm(psi)

            if t % INTV == 0:
                psi_total[c] = psi
                c += 1

        graph.animate_matplotlib(x, y, np.absolute(psi_total) ** 2)

    if vtk:
        if os.path.exists("datavtk"):
            shutil.rmtree("datavtk")
        os.makedirs("datavtk")
        for i in range(len(psi_total)):
            graph.writeVtk(i, np.abs(psi_total[i]) ** 2, Nx, dx, "datavtk")
