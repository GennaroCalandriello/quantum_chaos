import os
import sys
import PIL
import numpy as np
from PIL import Image

import scipy.linalg
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csr_matrix

from time import time, sleep
import matplotlib.pyplot as plt
from matplotlib import animation
from colorsys import hls_to_rgb
from numba import njit, prange
from IPython.display import display, clear_output
from mpl_toolkits.mplot3d import Axes3D
import graph

c16 = np.complex128
plt.rc("font", family="serif")


class Field:
    def __init__(self):
        self.potential_expr = None
        self.obstacle_expr = None

    def setPotential(self, expr):
        self.potential_expr = expr
        self.test_pot_expr()

    def setObstacle(self, expr):
        self.obstacle_expr = expr
        self.test_obs_expr()

    def test_pot_expr(self):
        # required for eval()
        x = 0
        y = 0
        try:
            a = eval(self.potential_expr)
        except:
            print(self.potential_expr)
            print("Potential calculation error: set to 0 by default")
            self.potential_expr = "0"

    def test_obs_expr(self):
        # required for eval()
        x = 0
        y = 0
        try:
            a = eval(self.obstacle_expr)
        except:
            print("Error setting obstacle: Set to False by default")
            self.obstacle_expr = "False"

    def isObstacle(self, x, y):
        a = False
        try:
            a = eval(self.obstacle_expr)
        except:
            print(f"Invalid obstacle: {self.obstacle_expr}")
        return a

    def getPotential(self, x, y):
        a = 0 + 0j
        try:
            a = eval(self.potential_expr)
        except:
            print(f"Invalid potential: {self.potential_expr}")
        return a


def apply_obstacle(MM, N, meshX, meshY):
    for i in range(N):
        for j in range(N):
            if Field.isObstacle(meshX[i][j], meshY[i][j]):
                MM[i][j] = 0 + 0j
    return MM


def getAdjPos(x, y, N):
    res = []
    res.append((x - 1, y))
    res.append((x + 1, y))
    res.append((x, y - 1))
    res.append((x, y + 1))
    res.append((x - 1, y + 1))
    res.append((x - 1, y - 1))
    res.append((x + 1, y + 1))
    res.append((x + 1, y + 1))
    return res


def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 - 1.0 / (1.0 + 2 * r ** 1.2)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    c = c.swapaxes(0, 1)
    return c


do_parralel = False


@njit(cache=True, parallel=do_parralel)
def x_concatenate(MM, N):
    result = np.zeros((N * N), dtype=c16)
    for j in prange(N):
        for i in prange(N):
            index = i + N * j
            result[index] = MM[i][j]
    return result


@njit(cache=True, parallel=do_parralel)
def x_deconcatenate(vector, N):
    result = np.zeros((N, N), dtype=c16)
    for j in prange(N):
        for i in prange(N):
            result[i][j] = vector[N * j + i]
    return result


@njit(cache=True, parallel=do_parralel)
def y_concatenate(MM, N):
    result = np.zeros((N * N), dtype=c16)
    for i in prange(N):
        for j in prange(N):
            index = j + N * i
            result[index] = MM[i][j]
    return result


@njit(cache=True, parallel=do_parralel)
def y_deconcatenate(vector, N):
    result = np.zeros((N, N), dtype=c16)
    for i in prange(N):
        for j in prange(N):
            result[i][j] = vector[N * i + j]
    return result


@njit(cache=True, parallel=do_parralel)
def dx_square(MM, N, Δx):
    result = np.zeros((N, N), dtype=c16)
    for j in prange(N):
        result[0][j] = MM[1][j] - 2 * MM[0][j]
        for i in prange(1, N - 1):
            result[i][j] = MM[i + 1][j] + MM[i - 1][j] - 2 * MM[i][j]
        result[N - 1][j] = MM[N - 2][j] - 2 * MM[N - 1][j]
    return result / (Δx ** 2)


@njit(cache=True, parallel=do_parralel)
def dy_square(MM, N, Δx):
    result = np.zeros((N, N), dtype=c16)
    for j in prange(N):
        result[j][0] = MM[j][1] - 2 * MM[j][0]
        for i in prange(1, N - 1):
            result[j][i] = MM[j][i + 1] + MM[j][i - 1] - 2 * MM[j][i]
        result[j][N - 1] = MM[j][N - 2] - 2 * MM[j][N - 1]
    return result / (Δx ** 2)


@njit(cache=True)
def integrate(MM, N, Δx):
    S = 0
    air = Δx * Δx / 2
    for i in prange(N - 1):
        for j in range(N - 1):
            AA, AB, BA, BB = MM[i][j], MM[i][j + 1], MM[i + 1][j], MM[i + 1][j + 1]
            S += air * (AA + AB + BA) / 3
            S += air * (BB + AB + BA) / 3
    return S


N = 512
SIZE = 10
Δt = 0.001
Δx = SIZE / N
# V = "5*x**2 + 5*y**2" # 2D harmonic oscillator potential
V = "False"


x0 = [-2.5]
y0 = [0.0]
k_x = [9.0]
k_y = [0.0]
a_x = [0.8]
a_y = [0.8]


# Potential as a function of x and y
field = Field()
field.setPotential(V)  # Ex: x**2+y**2"
potential_boudnary = []


# Obstacle: boolean expression in fct of x and y (set to False if you do not want an obstacle)
obstacles = (
    "(x > 0.5 and x < 1 and not ((y > 0.1 and y < 0.6) or (y < -0.1 and y > -0.6)))"
)
# obstacles = "False"
field.setObstacle(obstacles)
wall_potential = 1e10

######## Create points at all xy coordinates in meshgrid ########
x_axis = np.linspace(-SIZE / 2, SIZE / 2, N)
y_axis = np.linspace(-SIZE / 2, SIZE / 2, N)
X, Y = np.meshgrid(x_axis, y_axis)


######## Initialize Wavepackets ########
n = 0
phase = np.exp(1j * (X * k_x[n] + Y * k_y[n]))
px = np.exp(-((x0[n] - X) ** 2) / (4 * a_x[n] ** 2))
py = np.exp(-((y0[n] - Y) ** 2) / (4 * a_y[n] ** 2))

Ψ = phase * px * py
norm = np.sqrt(integrate(np.abs(Ψ) ** 2, N, Δx))
Ψ = Ψ / norm

for n in range(1, len(x0)):
    phase = np.exp(1j * (X * k_x[n] + Y * k_y[n]))
    px = np.exp(-((x0[n] - X) ** 2) / (4 * a_x[n] ** 2))
    py = np.exp(-((y0[n] - Y) ** 2) / (4 * a_y[n] ** 2))

    Ψn = phase * px * py
    norm = np.sqrt(integrate(np.abs(Ψn) ** 2, N, Δx))

    Ψ += Ψn / norm

NORM = np.sqrt(integrate(np.abs(Ψ) ** 2, N, Δx))
Ψ = Ψ / NORM


######## Create Potential ########
V_x = np.zeros(N * N, dtype="c16")
for j in range(N):
    for i in range(N):
        xx = i
        yy = N * j
        if field.isObstacle(x_axis[j], y_axis[i]):
            V_x[xx + yy] = wall_potential
        else:
            V_x[xx + yy] = field.getPotential(x_axis[j], y_axis[i])

V_y = np.zeros(N * N, dtype="c16")
for j in range(N):
    for i in range(N):
        xx = j * N
        yy = i
        if field.isObstacle(x_axis[i], y_axis[j]):
            V_y[xx + yy] = wall_potential
        else:
            V_y[xx + yy] = field.getPotential(x_axis[i], y_axis[j])

V_x_matrix = sp.sparse.diags([V_x], [0])
V_y_matrix = sp.sparse.diags([V_y], [0])
print("sparessssss, ", V_x_matrix)

######## Create Hamiltonian ########
LAPLACE_MATRIX = sp.sparse.lil_matrix(-2 * sp.sparse.identity(N * N))
for i in range(N):
    for j in range(N - 1):
        k = i * N + j
        LAPLACE_MATRIX[k, k + 1] = 1
        LAPLACE_MATRIX[k + 1, k] = 1

LAPLACE_MATRIX = LAPLACE_MATRIX / (Δx ** 2)

HX = 1 * sp.sparse.identity(N * N) - 1j * (Δt / 2) * (LAPLACE_MATRIX - V_x_matrix)
HX = csr_matrix(HX)

HY = 1 * sp.sparse.identity(N * N) - 1j * (Δt / 2) * (LAPLACE_MATRIX - V_y_matrix)
HY = csr_matrix(HY)


######## Place Obstacles ########
for i in range(0, N):
    for j in range(0, N):
        if field.isObstacle(x_axis[j], y_axis[i]):
            adj = getAdjPos(i, j, N)
            for xx, yy in adj:
                coord_check = xx >= 0 and yy >= 0 and xx < N and yy < N
                if coord_check and not field.isObstacle(x_axis[yy], y_axis[xx]):
                    potential_boudnary.append((i, j))


def evolve(Ψ):

    vector_wrt_x = x_concatenate(Ψ, N)
    vector_deriv_y_wrt_x = x_concatenate(dy_square(Ψ, N, Δx), N)
    U_wrt_x = vector_wrt_x + (1j * Δt / 2) * (vector_deriv_y_wrt_x - V_x * vector_wrt_x)
    U_wrt_x_plus = scipy.sparse.linalg.spsolve(HX, U_wrt_x)
    Ψ = x_deconcatenate(U_wrt_x_plus, N)

    vector_wrt_y = y_concatenate(Ψ, N)
    vector_deriv_x_wrt_y = y_concatenate(dx_square(Ψ, N, Δx), N)
    U_wrt_y = vector_wrt_y + (1j * Δt / 2) * (vector_deriv_x_wrt_y - V_y * vector_wrt_y)
    U_wrt_y_plus = scipy.sparse.linalg.spsolve(HY, U_wrt_y)
    Ψ = y_deconcatenate(U_wrt_y_plus, N)

    return Ψ


N_iter = 100
Ψ_arr = np.zeros((N_iter, N, N), dtype="c16")

start = time()
for i in range(N_iter):
    Ψ = evolve(Ψ)
    Ψ_arr[i] = Ψ
    NORM = np.sqrt(integrate(np.abs(Ψ) ** 2, N, Δx))
    clear_output(wait=True)
    print("Iteration {} / {}".format(i, N_iter))
    print("Function norm : {0:.9f} ".format(NORM))

graph.animate_matplotlib(x_axis, y_axis, np.absolute(Ψ_arr) ** 2)
end = time()
print("{:.1f} minutes".format((end - start) / 60))

