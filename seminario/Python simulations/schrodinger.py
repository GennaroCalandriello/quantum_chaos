import numpy as np
import matplotlib.pyplot as plt
import module.graph as graph
from numba import njit, float64, int32
from matplotlib import animation
import module.graph as graph

Nx = 200
Ny = Nx
Nt = 10000
Nt2D = 1000
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
dt = 1e-6
dx = x[1] - x[0]
dy = y[1] - y[0]
INTV = 50

psi0 = np.sqrt(2) * np.sin(np.pi * x)


mu, sigma = 1 / 2, 1 / 5
V = -1e4 * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


Cx = dt / dx ** 2
Cy = dt / dy ** 2

psi = np.zeros([Nt, Nx])
psi[0] = psi0


@njit()
def compute_psi1D(psi):
    for t in range(0, Nt - 1):
        if t % 1000 == 0:
            print("time step: ", t)
        for i in range(1, Nx - 1):
            psi[t + 1][i] = (
                psi[t][i]
                + 1j / 2 * Cx * (psi[t][i + 1] - 2 * psi[t][i] + psi[t][i - 1])
                - 1j * dt * V[i] * psi[t][i]
            )

        normalization = np.sum(np.absolute(psi[t + 1] ** 2)) * dx
        for i in range(1, Nx - 1):
            psi[t + 1][i] = psi[t + 1][i] / normalization
    return psi


onedimension = False
twodimensions = True

if onedimension:
    psi_sol = compute_psi1D(psi.astype(complex))

    def animate(i):
        ln1.set_data(x, np.absolute(psi_sol[INTV * i]) ** 2)
        time_text.set_text("$(10^4 mL^2)^{-1}t=$" + "{:.1f}".format(100 * i * dt * 1e4))

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # ax.grid()
    (ln1,) = plt.plot([], [], "r-", lw=2, markersize=8, label="Method 1")
    time_text = ax.text(
        0.65, 16, "", fontsize=15, bbox=dict(facecolor="white", edgecolor="black")
    )
    ax.set_ylim(-1, 20)
    ax.set_xlim(0, 1)
    ax.set_ylabel("$|\psi(x)|^2$", fontsize=20)
    ax.set_xlabel("$x/L$", fontsize=20)
    ax.legend(loc="upper left")
    ax.set_title("$(mL^2)V(x) = -10^4 \cdot n(x, \mu=L/2, \sigma=L/20)$")
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
    ani.save("pen.gif", writer="pillow", fps=50, dpi=100)

if twodimensions:
    psi02D = np.zeros((Nx, Ny))
    V2D = np.zeros((Nx, Ny))

    for i in range(Nx - 1):
        for j in range(Ny - 1):
            # psi02D[i, j] = np.sqrt(2) * np.sin(5 * np.pi * x[i] * np.pi * y[j])
            psi02D[i, j] = -1e1 * np.exp(
                -((x[i] - mu) ** 2 + (y[j] - mu) ** 2) / (2 * sigma ** 2)
            )

    @njit()
    def compute_psi2D(psi, psi_tot):
        c = 0
        psi_new = psi.copy()

        for t in range(Nt - 1):
            if t % INTV == 0:
                print("time step: ", t)
            for i in range(Nx - 1):
                for j in range(Ny - 1):
                    psi_new[i, j] = (
                        psi[i, j]
                        + 1j / 2 * Cx * (psi[i + 1, j] - 2 * psi[i, j] + psi[i - 1, j])
                        + 1j / 2 * Cy * (psi[i, j + 1] - 2 * psi[i, j] + psi[i, j - 1])
                        - 1j * dt * psi[i, j]
                    )
                    psi_new[i, j] = psi_new[i, j] * np.cos(x[i] + y[j])

            # print(psi_new)

            # for i in range(Nx - 1):
            #     normalizex = np.sum(np.absolute(psi_new[i, :]) ** 2) * dx
            # print(normalizex)
            # for j in range(Ny - 1):
            #     normalizey = np.sum(np.absolute(psi_new[:, j]) ** 2) * dy

            # for i in range(Nx - 1):
            #     psi_new[i, :] = psi_new[i, :] / normalizex
            # for j in range(Ny - 1):
            #     psi_new[:, j] = psi_new[:, j] / normalizey

            if t % INTV == 0:
                psi_tot[c] = psi_new
                c += 1

            psi = psi_new

        return psi_tot

    psi_tot = np.zeros((int(Nt / INTV), Nx, Ny)).astype(complex)
    psi_total = compute_psi2D(psi02D.astype(complex), psi_tot)
    graph.animate_matplotlib(x, y, np.absolute(psi_total) ** 2)
    # graph.static_plot(x, y, np.absolute(psi_total[0] ** 2))
    # graph.static_plot(x, y, np.absolute(V2D) ** 2)

