import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as SLA
from scipy.integrate import quad
from multiprocessing import Pool
from scipy.interpolate import UnivariateSpline

potentials_list = [
    "square",  # 0
    "Sinai",  # 1
    "Bunimovich",  # 2
    "mickey",  # 3
    "Ehrenfest",  # 4
    "Anderson",  # 5
    "Henon",  # 6
    "cardioid",  # 7
    "harm_osc",  # 8
]


potential = potentials_list[2]

plot = True
save = True
univariate = True  # if True use scipy.interpolate to unfold
unfolded = "n"

"""Select if you want the original spectrum or the unfolded one"""

if unfolded == "y":
    eig = np.loadtxt(
        f"unfolded_spectra/unfolded_spectrum_{potential}.txt", dtype=complex
    )
else:
    eig = np.loadtxt(f"eigenvalues_{potential}.txt", dtype=complex)


n = len(eig)
eig = eig.real
# eig = eig / SLA.norm(eig)
len_analysis = 78
gamma = 0.577216


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

    plt.figure()

    if univariate:
        poly_y = UnivariateSpline(lin_list, n_list)
        poly_y.set_smoothing_factor(5)

        if save:
            np.savetxt(f"unfolded_spectrum_{potential}.txt", poly_y(lin_list))

        plt.plot(lin_list, poly_y(lin_list), c="red", linestyle="--", label="Unfolding")

    else:
        poly = np.polyfit(lin_list, n_list, 60)
        poly_y = np.poly1d(poly)(lin_list)

        if save:
            np.savetxt(f"unfolded_spectrum_{potential}.txt", poly_y)

        plt.plot(lin_list, poly_y, c="red", linestyle="--", label="Unfolding")

    plt.step(lin_list, n_list, c="blue", label="Staircase")
    plt.xlabel("E", fontsize=15)
    plt.ylabel("N(E)", fontsize=15)
    plt.title(
        f"Staircase & Unfolding for {potential}, first {len_analysis} e.v.", fontsize=22
    )
    plt.legend()
    plt.show()


##--------------------------------------------------------------------------

"""Da qui in poi analisi statistiche dello spettro per correlazioni a lungo raggio"""

##----------------Spectral rigidity-----------------------------------------

"""tutte le formule da pagina 110 della buonanima di Stockmann (Quantum Chaos, An Introduction)"""

##------------- Rigidità spettrale e number variance teoriche per l'ensemble GOE-------------
def spectral_rigidity_GOE(e):

    """Spectral rigidity theoretical, for N -> infty"""

    delta3 = (1 / np.pi ** 2) * (np.log(2 * np.pi * e) + gamma - 5 / 4 - np.pi ** 2 / 8)
    return delta3


def number_variance_GOE(e):

    """Number Variance theoretical for N -> infty"""

    sigma2_teorica = (2 / np.pi ** 2) * (
        np.log(2 * np.pi * e) + gamma + 1 - np.pi ** 2 / 8
    )
    return sigma2_teorica


##--------------------------------------------------------------------------------------------
##-----------Integrandi per Sigma2 e Delta3----------------------------------------


def Y2(E):
    sin = lambda x: np.sin(x) / x
    Si, _ = quad(sin, 0, E * np.pi)
    partial = (np.sin(np.pi * E) / (np.pi * E)) ** 2
    y2 = partial + (np.pi / 2 * np.sign(E) - Si) * (
        (np.cos(np.pi * E)) / (np.pi * E) - (np.sin(np.pi * E)) / (np.pi * E) ** 2
    )
    return y2


def integrand_delta3(E, L):
    integrand = (L - E) ** 3 * (2 * L ** 2 - 9 * L * E - 3 * E ** 2) * Y2(E)
    return integrand


def integrand_sigma2(E, L):
    integrand_s2 = (L - E) * Y2(E)
    return integrand_s2


##---------------------------------------------------------------------------------

##----------------------Integrazione e plot delle statistiche Delta3 e Sigma2----------------

"""Integrazione delle statistiche e confronto con plot teorico atteso dalle previsioni delle RMT"""


def delta3_integrata():

    print(f"Calcolo la rigidità spettrale per {potential}")
    delta3_list = []
    c = 0
    for e in eig[:len_analysis]:
        delta3, _ = quad(integrand_delta3, 0, e, args=(e,))
        delta3_list.append(e / 15 - (1 / (15 * e ** 4)) * delta3)
        c += 1

    L = np.linspace(0, 500, 500)
    delta3teorica = spectral_rigidity_GOE(L)

    plt.scatter(
        eig[:len_analysis], delta3_list, s=25, c="blue", label=r"$\Delta_3$ integrata"
    )
    plt.plot(L, delta3teorica, c="g", linestyle="-.", label=r"$\Delta_3$ teorica")
    plt.title(f"Rigidità spettrale {potential}", fontsize=22)
    plt.xlabel("E", fontsize=15)
    plt.ylabel(r"$\Delta_3(E)$", fontsize=15)
    plt.legend()
    # plt.yscale("log")
    plt.xlim((-50, max(eig[:len_analysis]) + 50))
    plt.show()


def sigma2_integrata():

    """Misura del Number Variance sullo spettro selezionato"""

    print(f"Calcolo number variance per {potential}")

    sigma2_list = []
    for e in eig[:len_analysis]:
        sigma2, _ = quad(integrand_sigma2, 0, e, args=(e,))
        sigma2_list.append(e - 2 * sigma2)

    L = np.linspace(min(eig[:len_analysis]), max(eig[:len_analysis]), 500)
    sigma2teorica = number_variance_GOE(L)

    plt.figure()
    plt.scatter(
        eig[:len_analysis], sigma2_list, s=25, c="red", label=r"$\Sigma_2$ integrata"
    )
    plt.plot(L, sigma2teorica, c="m", linestyle="-.", label=r"$\Sigma_2$ teorica")
    plt.title(f"Number Variance {potential}", fontsize=22)
    plt.xlabel("E", fontsize=15)
    plt.ylabel(r"$\Sigma_2 (E)$", fontsize=15)
    plt.legend()
    # plt.yscale("log")
    plt.xlim((-50, max(eig[:len_analysis]) + 50))
    plt.show()


##---------------------------------------------------------------------------------

##-----------density of states----------------------------------------------
def rho():

    """Trasformata di Fourier dello spettro"""
    eigf = eig
    ff = np.fft.fft(eigf)
    # ff = ff / SLA.norm(ff)
    t = np.arange(n)
    freq = np.fft.fftfreq(t.shape[-1])  # frequenze

    if plot:
        plt.plot(freq[0 : len(ff) // 2], np.abs((ff)[0 : len(ff) // 2]) ** 2, c="green")
        plt.xlabel("f")
        plt.ylabel(r"$|\rho|^2$")
        plt.legend()
        plt.show()


def fluctuations():

    """Level energies must be unfolded"""

    eigfl = eig[:len_analysis]
    delta_n = []
    for i in range(len(eigfl)):
        delta_n.append(i - 0.5 - eigfl[i])

    plt.plot(eigfl, delta_n)
    plt.show()


##---------------------------------------------------------------------------
if __name__ == "__main__":

    parallel_exe = False

    """Executing in parallel the 2 statistics"""
    from multiprocessing import Process

    if parallel_exe:
        p1 = Process(target=delta3_integrata)
        p2 = Process(target=sigma2_integrata)
        p1.start()
        p2.start()
        p1.join()
        p2.join()

    staircase_and_unfolding()
    fluctuations()

