import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as SLA
from scipy.integrate import quad


plot = True
eig = np.loadtxt("eigenvalues_Sinai.txt", dtype=complex)

n = len(eig)
eig = eig.real
# eig = eig / SLA.norm(eig)
len_analysis = 50


##----------------------Staircase function working: unfolding spectra--------------------------
def staircase_and_unfolding():
    lin_list = []
    n_list = []
    idx = 1
    eig1 = eig[:len_analysis]
    eig1 = np.sort(eig1)

    for e in range(1, len(eig1)):
        lin = 0
        lin = np.linspace(eig1[e], eig1[e - 1], 1)
        lin_list.append(lin)
        for _ in lin:
            n_list.append(idx)
        idx += 1

    lin_list = np.reshape(lin_list, len(n_list))

    poly = np.polyfit(lin_list, n_list, 10)
    poly_y = np.poly1d(poly)(lin_list)
    plt.step(lin_list, n_list, c="blue", label="Staircase")
    plt.plot(lin_list, poly_y, c="red", linestyle="--", label="Unfolding")
    plt.xlabel("E")
    plt.ylabel("N(E)")
    plt.legend()
    plt.show()


##--------------------------------------------------------------------------
##----------------Spectral rigidity-----------------------------------------

"""tutte le formule da pagina 110 della buonanima di Stockmann (Quantum Chaos An Introduction)"""


def spectral_rigidity_theoretical(L):
    delta3 = (np.log(L) - 0.068) / np.pi ** 2
    return delta3


def spectral_rigidity_try(e):
    gamma = 0.577216
    delta3 = (1 / np.pi ** 2) * (np.log(2 * np.pi * e) + gamma - 5 / 4 - np.pi ** 2 / 8)
    return delta3


def Y2(E):
    sin = lambda x: np.sin(x) / x
    Si, _ = quad(sin, 0, E * np.pi)
    part = (np.sin(np.pi * E) / (np.pi * E)) ** 2
    y2 = part + (np.pi / 2 * np.sign(E) - Si) * (
        (np.cos(np.pi * E)) / (np.pi * E) - (np.sin(np.pi * E)) / (np.pi * E) ** 2
    )
    return y2


def integrand_delta3(E, L):
    integrand = (L - E) ** 3 * (2 * L ** 2 - 9 * L * E - 3 * E ** 2) * Y2(E)
    return integrand


def delta3_integrata():
    # eig = eig.real
    delta3_list = []
    # l = np.linspace(min(eig), max(eig), len_analysis)
    # l = np.linspace(1, 20, len_analysis)
    # l = max(eig)
    c = 0
    for e in eig[:len_analysis]:
        delta3, _ = quad(integrand_delta3, 0, e, args=(e,))
        delta3_list.append(e / 15 - (1 / (15 * e ** 4)) * delta3)
        c += 1

    L = np.linspace(min(eig[:len_analysis]), max(eig[:len_analysis]), 500)
    delta3teorica = spectral_rigidity_try(L)

    plt.scatter(
        eig[:len_analysis], delta3_list, s=25, c="blue", label=r"$\Delta_3$ integrata"
    )
    plt.plot(L, delta3teorica, c="g", linestyle="-.", label=r"$\Delta_3$ teorica")
    plt.xlabel("E")
    plt.ylabel(r"$\Delta_3(E)$")
    plt.legend()
    plt.yscale("log")
    plt.xlim((-0.5, max(eig[:len_analysis]) + 1))
    plt.show()

    # plt.show()


##--------------------------------------------------------------------------
##-----------density of states----------------------------------------------
def rho():

    ff = np.fft.fft(eig)  # trasformata di Fourier dello spettro di autovalori
    ff = ff / SLA.norm(ff)
    t = np.arange(n)
    freq = np.fft.fftfreq(t.shape[-1])  # frequenze

    if plot:
        plt.plot(freq[0 : len(ff) // 2], np.abs(ff)[0 : len(ff) // 2] ** 2, c="green")
        plt.xlabel("f")
        plt.ylabel(r"$|\rho|^2$")
        plt.legend()
        plt.show()


##---------------------------------------------------------------------------
if __name__ == "__main__":

    staircase_and_unfolding()
    delta3_integrata()
    rho()

