import numpy as np
import matplotlib.pyplot as plt


def logisticMap(mu, x):
    return mu * x * (1 - x)


def PlotLogisticMap(mu, x0, n, ax=None):
    t = np.linspace(0, 1)
    ax.plot(t, logisticMap(mu, t), "k", lw=2)
    ax.plot([0, 1], [0, 1], "k", lw=2)
    x = x0

    for i in range(n):
        y = logisticMap(mu, x)
        ax.plot([x, x], [x, y], "k", lw=1)
        ax.plot([x, y], [y, y], "k", lw=1)
        ax.plot([x], [y], "ok", ms=10, alpha=(i + 1) / n)

        x = y

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"$ /mu={mu: .1f}, \, x_0={x0: .1f}$")


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# PlotLogisticMap(1.4, 0.01, 10, ax=ax1)
# PlotLogisticMap(3.5, 0.01, 10, ax=ax2)
# plt.show()


def BifurcationAndLyapunov(iterations=10000, n=10000):
    last = 200

    lyapunov = 0
    muvec = np.linspace(
        2.5, 7.0, n
    )  # vector of increment of the control parameter of the map
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    x = 1e-4 * np.random.rand(n)  # initial conditions

    for i in range(iterations):
        x = logisticMap(muvec, x)
        # partial sum of Lyapunov exponent:
        lyapunov += np.log(
            abs(muvec - 2 * muvec * x)
        )  # lim(N->infty SUM(i) log(f'(x_i))) pag 149 Tabor

        if i >= (iterations - last):  # plotting the last 'last' iterations
            ax1.plot(muvec, x, ",k", alpha=0.5)

    ax1.set_xlim(2.5, 7)
    ax1.set_title("Bifurcation diagram")

    # displaying Lyapunov exponents
    ax2.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax2.plot(
        muvec, lyapunov / iterations, ".k", alpha=0.5, ms=0.5,
    )

    ax2.set_xlim(2.5, 4)
    ax2.set_ylim(-2, 1)
    ax2.set_title("Lyapunov exponent")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    BifurcationAndLyapunov()
