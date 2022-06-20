import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import sparse
import scipy.linalg as SLA
from scipy.sparse.linalg import isolve
import time
from numba import njit

# Main program PARAMS
Nx = 200
Ny = Nx  # TODO: Only tried for Nx=Ny
tmax = 200  # Number of frames
xmax = 1.0
ymax = 1.0
# dx = 1.0 / (Nx - 1)
x = np.linspace(0, xmax, Nx)
y = np.linspace(0, ymax, Ny)
dx = x[1] - x[0]
print("icccccccccsssssssssssssss", x)
a = 1j / 2 / dx ** 2  # Time term: Adds to diagonal of FD Hamiltonian
b = 1 / 4 / dx ** 2.0  # off-diagonal elements of FD Hamiltonian
TOL = 1e-6  # tolerance in iterative linear solver

# Initial Gaussian and its momentum
sig = 0.07  # std dev of Gaussian
cwp = [Nx / 6, Ny / 2]  # start position
pwp = [-4 * Nx / 10, 6 * Nx / 10]  # Momentum, 2pi/lambda

# Potential
V = np.zeros((Nx, Ny))
# Define whatever potential you wish here

psi = np.zeros((Nx, Ny)).astype(complex)

# Gaussian wave packet
for ii in range(Nx):
    for jj in range(Ny):

        psi[ii, jj] = np.exp(
            1j * (pwp[0] * (ii - cwp[0]) + pwp[1] * (jj - cwp[1])) * dx
            - ((ii - cwp[0]) ** 2 + (jj - cwp[1]) ** 2) * dx ** 2 / (2 * sig ** 2)
        )
# return psi


# time.sleep(737)
# psi = np.array(psi).reshape(Nx, Ny)
# psi = initial_coherent_state()
print("questo è psiii: ", psi)
psi = psi / SLA.norm(psi)

# Create FD Hamiltonian for one slice of the domain
lhs = np.zeros((3, Nx))
lhs[0, :] = b * np.ones(Nx)  # Upper off-diagonal
lhs[2, :] = lhs[0, ::-1]  # Lower off-diagonal

# Returns FD Ham in sparse csr format to be used in an iterative solver below
# Only the diagonal is updated in LHS/LHS2


def LHS(idx):
    mtx = sparse.spdiags(
        [lhs[0], 2 * a - 2 * b - V[:, idx] / 2, lhs[2]], [1, 0, -1], Nx, Nx
    )
    return sparse.csr_matrix(mtx)


def LHS2(idx):
    mtx = sparse.spdiags(
        [lhs[0], 2 * a - 2 * b - V[idx, :] / 2, lhs[2]], [1, 0, -1], Nx, Nx
    )
    return sparse.csr_matrix(mtx)


def RHS(idx):
    return np.array(
        (2 * a + 2 * b + V[:, idx] / 2) * psi[:, idx]
        - b
        * np.array(
            [
                psi[ii, min(Nx - 1, idx + 1)] + psi[ii, max(0, idx - 1)]
                for ii in range(Nx)
            ]
        )
    )


def RHS2(idx):
    return np.array(
        (2 * a + 2 * b + V[idx, :] / 2) * psi[:, idx]
        - b
        * np.array(
            [
                psi[jj, min(Nx - 1, idx + 1)] + psi[jj, max(0, idx - 1)]
                for jj in range(Nx)
            ]
        )
    )


def plotandsave(tt, psi):
    plt.imshow(np.abs(psi) ** 2)
    # Add possible line plots, etc. here
    plt.savefig(
        "gif_animation/fig" + f"{tt:04d}.png"
    )  # Manually create your framefolder/ directory


t1 = time.process_time()
for tt in range(tmax):
    plotandsave(tt, psi)
    print(tt, "saved")
    psi = np.array([isolve.gmres(LHS(jj), RHS(jj), tol=TOL)[0] for jj in range(Nx)])
    psi = np.array([isolve.gmres(LHS2(ii), RHS2(ii), tol=TOL)[0] for ii in range(Nx)])
    psi = psi / SLA.norm(psi)
tm = time.process_time() - t1
print("\nCALCULATION DONE, FRAMES SAVED. Total time:", tm, "s =", tm / 60, "min")

