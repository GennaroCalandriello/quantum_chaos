import numpy as np
import pylab
from numba import njit


# @njit()


def func(u, t):

    sigma, rho, beta = 10, 28, 8 / 3

    x, y, z = u

    fx = sigma * (y - x)
    fy = x * (rho - z) - y
    fz = x * y - beta * z

    return np.array([fx, fy, fz], float)


# @njit()
def RungeKutta4th(f, u0, t0, tf, n):

    print("i arrii")
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

    u, t = RungeKutta4th(func, np.array([0.0, 1.0, 10.0]), 0.0, 10, 10000)
    fx, fy, fz = u.T

    fig = pylab.figure()
    ax = pylab.axes(projection="3d")
    ax.plot3D(fx, fy, fz, "blue")
    pylab.show()

