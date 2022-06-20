import numpy as np
import matplotlib.pyplot as plt
import module.graph as graph
from numba import njit, float64, int32
from matplotlib import animation
import module.graph as graph

Nx = 300
Ny = Nx
Nt = 3000

L = 2 * np.pi
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
dt = 2e-9
dx = x[1] - x[0]
print("dx ", dx)
dy = y[1] - y[0]
INTV = 20

D = 0.5
mu, sigma = 1 / 2, 1 / 10
Cx = D * dt / dx ** 2
Cy = D * dt / dy ** 2
print(Cx)

psi02D = np.zeros((Nx, Ny))
V2D = np.zeros((Nx, Ny))

for i in range(Nx - 1):
    for j in range(Ny - 1):
        # psi02D[i, j] = np.sqrt(2) * np.sin(5 * np.pi * x[i] * np.pi * y[j])
        psi02D[i, j] = np.exp(-((x[i] - mu) ** 2 + (y[j] - mu) ** 2) / (2 * sigma ** 2))


@njit()
def compute_psi2D(psi):
    B_COND = "periodic"
    psi_tot = np.zeros((int(Nt / INTV), Nx, Ny))

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
            temp = np.zeros(Nx)
            for j in range(1, Ny - 1):
                # psi_bc = boundary_billiard2(x[i], y[j], psi[i, j])
                # psi[i, j] = psi_bc
                # if B_COND == "Dirichlet":
                #     iplus = i + 1
                #     if iplus == Nx - 1:
                #         iplus = Nx - 2
                #     iminus = i - 1
                #     if iminus == 0:
                #         iminus = 1
                #     jplus = j + 1
                #     if jplus == Ny - 1:
                #         jplus = Ny - 2
                #     jminus = j - 1
                #     if jminus == 0:
                #         jminus = 1
                # if B_COND == "periodic":

                #     iplus = (i + 1) % Nx
                #     iminus = (i - 1) % Nx
                #     if iminus < 0:
                #         iminus += Nx
                #     jplus = (j + 1) % Ny
                #     jminus = (j - 1) % Ny
                #     if jminus < 0:
                #         jminus += Ny
                iplus, jplus, iminus, jminus = i + 1, j + 1, i - 1, j - 1
                temp[j] = (
                    psi[i, j]
                    - Cx * (psi[iplus, j] - 2 * psi[i, j] + psi[iminus, j])
                    - Cy * (psi[i, jplus] - 2 * psi[i, j] + psi[i, jminus])
                    # - dt * psi[i, j]
                )
                temp[j] = temp[j] * np.cos(x[i] + y[j])  # np.cos(phase)
            for j in range(Ny - 1):
                psi_new[i, j] = temp[j]
                psi[i, j] = psi_new[i, j]

        # psi[0, :] = psi[1, :]
        # psi[:, 0] = psi[:, 1]
        # psi[Nx - 1, :] = psi[Nx - 2, :]
        # psi[:, Ny - 1] = psi[:, Ny - 2]

        if t % INTV == 0:
            psi_tot[c] = psi_new
            c += 1

        # psi = psi_new

    return psi_tot


psi_total = compute_psi2D(psi02D)
graph.animate_matplotlib(x, y, psi_total)

