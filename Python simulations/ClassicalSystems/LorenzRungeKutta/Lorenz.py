from cProfile import label
import numpy as np
import pylab
from numba import njit
import matplotlib.pyplot as plt


# @njit()


def Lorenz(u, t):

    sigma, rho, beta = 10, 28, 8 / 3

    x, y, z = u

    fx = sigma * (y - x)
    fy = x * (rho - z) - y
    fz = x * y - beta * z

    return np.array([fx, fy, fz], float)


# @njit()
def RungeKutta4th(f, u0, t0, tf, n):

    t = np.linspace(t0, tf, n + 1)
    u = np.array((n + 1) * [u0])
    h = t[1] - t[0]

    for i in range(n):
        k1 = h * f(u[i], t[i])
        k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5 * h)
        k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5 * h)
        k4 = h * f(u[i] + k3, t[i] + h)
        u[i + 1] = u[i] + (k1 + 2 * (k2 + k3) + k4) / 6

    return u, t


if __name__ == "__main__":

    # Compare two different set of initial conditions:

    # 1
    initial_conditions1 = np.array([0.1, 0.2, 0.1])
    uLorenz, t = RungeKutta4th(Lorenz, initial_conditions1, 0.0, 40, 10000)
    fx, fy, fz = uLorenz.T

    # 2
    initial_conditions2 = np.array([0.1, 0.2, 0.1001])
    uLorenz2, t2 = RungeKutta4th(Lorenz, initial_conditions2, 0.0, 40, 10000)
    fx2, fy2, fz2 = uLorenz2.T

    pylab.figure()
    ax = pylab.axes(projection="3d")
    ax.plot3D(
        fx,
        fy,
        fz,
        "blue",
        label=f"x={initial_conditions1[0]}, y={initial_conditions1[1]}, z={initial_conditions1[2]}",
    )  # 1
    ax.plot3D(
        fx2,
        fy2,
        fz2,
        "orange",
        label=f" x={initial_conditions2[0]}, y={initial_conditions2[1]}, z={initial_conditions2[2]}",
    )  # 2
    pylab.title("Lorenz system for 2 different sets of initial conditions")
    pylab.legend()
    pylab.show()

