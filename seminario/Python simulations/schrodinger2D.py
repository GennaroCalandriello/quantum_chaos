import numpy as np
import matplotlib.pyplot as plt
import module.graph as graph
from numba import njit, float64, int32
from matplotlib import animation
import module.graph as graph

Nx = 300
Ny = Nx
Nt = 10000
Nt2D = 1000
L = 2 * np.pi
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
dt = 1e-7
dx = x[1] - x[0]
dy = y[1] - y[0]
INTV = 20

mu, sigma = 1 / 2, 1 / 10
Cx = dt / dx ** 2
Cy = dt / dy ** 2
print(Cx)


psi02D = np.zeros((Nx, Ny))
V2D = np.zeros((Nx, Ny))

for i in range(Nx - 1):
    for j in range(Ny - 1):
        # psi02D[i, j] = np.sqrt(2) * np.sin(5 * np.pi * x[i] * np.pi * y[j])
        psi02D[i, j] = -1e1 * np.exp(
            -((x[i] - mu) ** 2 + (y[j] - mu) ** 2) / (2 * sigma ** 2)
        )


@njit()
def module2(x, y):
    return np.sqrt(x ** 2 + y ** 2)


@njit()
def boundary_billiard(x, y):
    LAMBDA = 0.3
    if (x > -0.5 * LAMBDA) and (x < 0.5 * LAMBDA) and (y > -1.0) and (y < 1.0):
        return (x, y)
    if module2(x + 0.5 * LAMBDA, y) < 1.0:
        return (x, y)
    if module2(x - 0.5 * LAMBDA, y) < 1.0:
        return (x, y)
    else:
        return 0


@njit()
def boundary_billiard2(x, y, psi):
    LAMBDA = 0.3
    if (x > -0.5 * LAMBDA) and (x < 0.5 * LAMBDA) and (y > -1.0) and (y < 1.0):
        return psi
    if module2(x + 0.5 * LAMBDA, y) < 1.0:
        return psi
    if module2(x - 0.5 * LAMBDA, y) < 1.0:
        return psi
    else:
        psi = 0
        return psi


@njit()
def compute_psi2D(psi, psi_tot):
    B_COND = "periodic"
    c = 0
    px = 20.0
    py = 0.0
    xs = -1.2
    ys = 0.0
    psi_new = psi.copy()
    #  phase = (px * (xy[0] - x) + py * (xy[1] - y)) / scalex;
    phase = 0

    for t in range(Nt - 1):
        if t % INTV == 0:
            print("time step: ", t)
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                # psi_bc = boundary_billiard2(x[i], y[j], psi[i, j])
                # psi[i, j] = psi_bc
                if B_COND == "Dirichlet":
                    iplus = i + 1
                    if iplus == Nx - 1:
                        iplus = Nx - 2
                    iminus = i - 1
                    if iminus == 0:
                        iminus = 1
                    jplus = j + 1
                    if jplus == Ny - 1:
                        jplus = Ny - 2
                    jminus = j - 1
                    if jminus == 0:
                        jminus = 1
                if B_COND == "periodic":

                    iplus = (i + 1) % Nx
                    iminus = (i - 1) % Nx
                    if iminus < 0:
                        iminus += Nx
                    jplus = (j + 1) % Ny
                    jminus = (j - 1) % Ny
                    if jminus < 0:
                        jminus += Ny
                # iplus, jplus, iminus, jminus = i + 1, j + 1, i - 1, j - 1
                psi_new[i, j] = (
                    psi[i, j]
                    + Cx * (psi[iplus, j] - 2 * psi[i, j] + psi[iminus, j])
                    + Cy * (psi[i, jplus] - 2 * psi[i, j] + psi[i, jminus])
                    # - dt * psi[i, j]
                )
                psi_new[i, j] = psi_new[i, j] * np.cos(x[i] + y[j])  # np.cos(phase)

        psi_new[0, :] = psi_new[1, :]
        psi_new[:, 0] = psi_new[:, 1]
        psi_new[Nx - 1, :] = psi_new[Nx - 2, :]
        psi_new[:, Ny - 1] = psi_new[:, Ny - 2]

        if t % INTV == 0:
            psi_tot[c] = psi_new
            c += 1

        psi = psi_new

    return psi_tot


#  init_coherent_state(-1.2, 0.0, 20.0, 0.0, 0.25, phi, psi, xy_in);
psi_tot = np.zeros((int(Nt / INTV), Nx, Ny)).astype(complex)
psi_total = compute_psi2D(psi02D.astype(complex), psi_tot)

for c in range(len(psi_total)):
    print("scrive")
    graph.writeVtk(c, np.absolute(psi_total[c]) ** 2, Nx, dx, "data2D")
graph.animate_matplotlib(x, y, np.absolute(psi_total) ** 2)

