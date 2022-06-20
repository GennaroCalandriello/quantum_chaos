import numpy as np
import matplotlib.pyplot as plt
from numba import njit, complex128
import scipy.linalg as SLA
import os
import shutil
import time
import function as fnc
import graph as graph

Nx = 200
Ny = Nx
tmax = 4000
dt = 1e-5
xmax = 1.0
ymax = 1.0
x = np.linspace(0, xmax, Nx)
y = np.linspace(0, ymax, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
alphax = 1j * dt / (2 * dx ** 2)
alphay = alphax
alphareal = dt / (2 * dx ** 2)

stdv = 0.07
xy0 = [Nx / 2, Ny / 4]
pxpy0 = [-4 * Nx / 17, 6 * Nx / 23]

stdv2 = 0.09
xy02 = [Nx / 2, Ny / 2]
pxpy02 = [-3 * Nx / 17, 4 * Nx / 15]

#
V = np.zeros((Nx, Ny))
# V[120:124, 90::] = 1e6
# V[120:124, 86:90] = 0
# V[120:124, 76:86] = 1e6
# V[120:124, 72:76] = 0
# V[120:124, 0:72] = 1e6
# V = V.T
# V[140:160, 140:160] = 1e15
# V[100:105, 100:105] = 1e15
# graph.static_plot(x, y, V, 0)
# time.sleep(222)

###----------------some parameters to isolate a circle domain:-------------###
x0, y0, radius = 0.5, 0.5, 0.2


def domain(x, y):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return r


idx_list = []

for i in range(Nx):
    for j in range(Ny):
        r = domain(x[i], y[j])
        if r < radius:
            idx_list.append((i, j))


for i, j in idx_list:
    V[i, j] = 1e6

# graph.static_plot(x, y, V, 0)
###------------------------------------------------------------------------###

psi = np.zeros((Nx, Ny), dtype=np.complex128)
psi2 = psi.copy()

for ii in range(Nx):
    for jj in range(Ny):
        psi[ii, jj] = np.exp(
            1j * (pxpy0[0] * (ii - xy0[0]) + pxpy0[1] * (jj - xy0[1])) * dx
            - ((ii - xy0[0]) ** 2 + (jj - xy0[1]) ** 2) * dx ** 2 / (2 * stdv ** 2)
        )

for ii in range(Nx):
    for jj in range(Ny):
        psi2[ii, jj] = np.exp(
            1j * (pxpy02[0] * (ii - xy02[0]) + pxpy02[1] * (jj - xy02[1])) * dx
            - ((ii - xy02[0]) ** 2 + (jj - xy02[1]) ** 2) * dx ** 2 / (2 * stdv2 ** 2)
        )

psi = psi / SLA.norm(psi)
psi2 = psi2 / SLA.norm(psi2)

up_diag = np.ones(Nx - 1, dtype=np.complex128) * alphax
low_diag = up_diag[::-1]
# graph.static_plot(x, y, np.absolute(psi) ** 2, 0)


@njit()
def ADI_x_y(psi, main, step):
    for i in range(Nx):
        for j in range(Ny):
            main[j] = 1 + alphax * 2 + 1j * (dt / 2) * V[i, j]
        for j in range(Ny):
            step[j] = (
                psi[i, j]
                + alphay
                * (psi[i, min(Ny - 1, j + 1)] - 2 * psi[i, j] + psi[i, max(0, j - 1)])
                - 1j * (dt / 2) * V[i, j] * psi[i, j]
            )

        temp = fnc.solve_matrix(Nx, low_diag, main, up_diag, step)
        for j in range(Ny):
            psi[i, j] = temp[j]
    return psi


@njit()
def ADI_y_x(psi, main, step):
    for j in range(Ny):
        for i in range(Nx):
            main[i] = 1 + 2 * alphay + 1j * (dt / 2) * V[i, j]
        for i in range(Nx):
            step[i] = (
                psi[i, j]
                + alphax
                * (psi[min(Nx - 1, i + 1), j] - 2 * psi[i, j] + psi[max(0, i - 1), j])
                - 1j * dt * V[i, j] * psi[i, j] / 2
            )

        temp = fnc.solve_matrix(Ny, low_diag, main, up_diag, step)
        for i in range(Nx):
            psi[i, j] = temp[i]
    return psi


if __name__ == "__main__":

    INTV = 3
    psi_total = np.zeros((int(tmax / INTV) + 3, Nx, Ny)).astype(complex)
    # psi = np.zeros(np.shape(psi)).astype(complex)

    exe = True
    vtk = True

    if exe:
        c = 0
        # psitilde = psi + 1.5 * psi2
        psitilde = psi  # riassegno se non voglio altri pacchetti
        # graph.static_plot(x, y, np.absolute(psitilde) ** 2, 0)
        for t in range(tmax):
            main = np.ones(Nx).astype(complex)
            step = np.zeros(Nx).astype(complex)
            step1 = step.copy()
            if t % INTV == 0:
                print(f"time step: {t}")
            psitilde = ADI_x_y(psitilde, main, step)
            # psi1 = psi1 / SLA.norm(psi1)
            psitilde = ADI_y_x(psitilde, main, step1)
            psitilde = psitilde / SLA.norm(psitilde)

            if t % INTV == 0:
                psi_total[c] = psitilde
                c += 1

        graph.animate_matplotlib(x, y, np.absolute(psi_total) ** 2)

    if vtk:
        if os.path.exists("datavtk"):
            shutil.rmtree("datavtk")
        os.makedirs("datavtk")
        for i in range(len(psi_total)):
            graph.writeVtk(i, np.absolute(psi_total[i]) ** 2, Nx, dx, "datavtk")


# psi[i, j]+ alphax* (psi[min(Nx - 1, i + 1), j] - 2 * psi[i, j] + psi[max(0, i - 1), j])- 1j * dt * V[i, j] * psi[i, j] / 2


# psi[i, j]+ alphay* (psi[i, min(Ny - 1, j + 1)] - 2 * psi[i, j] + psi[i, max(0, j - 1)])- 1j * (dt / 2) * V[i, j] * psi[i, j]

