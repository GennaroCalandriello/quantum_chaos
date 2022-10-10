import numpy as np
import module.graph as graph
import matplotlib.pyplot as plt
import shutil
import os
import multiprocessing
import model.par as par


"""Evolution of Gaussian Wave Packet in phas space through Husimi distribution"""

M = 600
kicks = 40
x0 = 0.15  # questo lo devi cambiare anche sotto sennò era un casotto
p0 = 0.03  # questo lo devi cambiare anche sotto sennò era un casotto
hbar = 2 * np.pi / M

multiproc = False  # execution in multiprocessing
executor = True  # execution of mean energy

processes = 8  # define the number of different K
maxK = 6.66
minK = 0.6


# def waveFunc(M, x0):
#     n = np.arange(M)
#     x = 2 * np.pi * n / M
#     psi0 = np.zeros(M, dtype=complex)
#     psi0[:] = np.exp(-((x - x0) ** 2) / 2 / hbar)
#     psi0 = psi0 / np.linalg.norm(psi0)
#     return x, psi0


# psi0, x = waveFunc(M, np.pi)
# plt.plot(x, np.abs(psi0)**2)
# plt.show()


def gaussianWavePacket(M, x0, p0):
    n = np.arange(M)
    x = 2 * np.pi * n / M
    p = 2 * np.pi * n / M

    n0 = x0 * M / 2 / np.pi
    m0 = p0 * M / 2 / np.pi

    psi_exp = 1j * 2 * np.pi * m0 * n / M
    psil = np.exp(psi_exp)

    nt, l = np.meshgrid(n, np.arange(-4, 5))
    psi_exp = -np.pi / M * (nt - n0 + l * M) ** 2
    psir = np.sum(np.exp(psi_exp), axis=0)
    N = np.linalg.norm(psil * psir)

    psi_x = psil * psir / N
    psi_p = 1 / np.sqrt(M) * np.fft.fft(psi_x)
    return x, p, psi_x, psi_p


def phaseVectors(x, p, K):

    """unitary evolution operator U=U(kick)*U(free)"""

    Ukick = np.exp(-1j / hbar * K * np.cos(x))
    Ufree = np.exp(-1j / (2 * hbar) * p ** 2)
    return Ukick, Ufree


def evolveOneStep(Ukick, Ufree, psi_x):

    """The keyblade"""

    fm = Ukick * psi_x
    sm = Ufree * np.fft.fft(fm)
    s = np.fft.ifft(sm)

    return s


def StdMap(n, K, x0=1.0, p0=1.0):

    x, p = np.zeros(n), np.zeros(n)
    x[0], p[0] = x0, p0
    for i in np.arange(n - 1):
        p[i + 1] = np.mod(p[i] + K * np.sin(x[i]), 2 * np.pi)
        x[i + 1] = np.mod(x[i] + p[i + 1], 2 * np.pi)
    return x, p


def MeanEnergy(Qdist, result=True):
    n = np.arange(M)
    x = 2 * np.pi * n / M
    p = 2 * np.pi * n / M
    dx = x[1] - x[0]
    meanP = []
    length = len(Qdist)
    # p2 = numpy.sum((abs(wave_function) * tab_momentum) ** 2)
    Uk, Uf = phaseVectors(x, p, 9)
    p2 = p ** 2

    for k in range(length - 1):

        meanP.append(np.mean(np.abs(Qdist[k])))

    meanP = np.array(meanP)

    if not result:
        plt.figure()
        plt.title("Mean Energy", fontsize=20)
        plt.plot(range(len(meanP)), meanP, c="blue")
        plt.xlabel("kicks", fontsize=15)
        plt.ylabel(r"$<E>$", fontsize=15)
        plt.axvline(x=35, linestyle="dotted", label="range of quantum break time")
        plt.axvline(x=47, linestyle="dotted")
        plt.legend()
        plt.show()

    deltaP = []
    meanP = np.array(meanP)
    for d in range(len(meanP) - 1):
        deltaP.append(meanP[d + 1] - meanP[d])

    localization_length = np.mean(deltaP) / hbar ** 2
    print("This is the approx localization length", np.sqrt(localization_length))

    if result:
        return meanP, localization_length


def meanenergyretry():
    n = np.arange(M)
    x = 2 * np.pi * n / M
    p = 2 * np.pi * n / M

    meanP = []
    p2 = p ** 2
    for k in range(kicks):
        Ukick, Ufree = phaseVectors(x, p, k)
        p2 = np.conj(Ukick * Ufree) * p2 * (Ukick * Ufree)
        meanP.append(np.mean(p2))

    plt.plot(range(kicks), meanP)
    plt.show()


# meanenergyretry()


def KRmain(Kick):

    """Multiprocessing execution"""

    x0 = 0.15
    p0 = 0.03
    # M = 400
    # kicks = 50

    print(f"Process for K = {Kick}")
    graphic = False
    x, p, psi_x, psi_p = gaussianWavePacket(M, x0, p0)
    V, P = phaseVectors(x, p, Kick)

    Qdist = np.zeros((M, M), dtype=complex)
    Qdist_tot = np.zeros((kicks, M, M), dtype=complex)

    for k in np.arange(kicks):
        print(f"Kick number {k}")
        psi_x = evolveOneStep(V, P, psi_x)

        for i in np.arange(M):
            x0 = 2 * np.pi * i / M
            x, p, psiG_n0, psi_p = gaussianWavePacket(M, x0, 0)
            Qdist[:, i] = 1 / np.sqrt(M) * np.fft.fft(np.conj(psiG_n0) * psi_x)

        n = np.arange(M)
        Qdist_tot[k] = Qdist

    n = np.arange(M)
    X = 2 * np.pi * n / M
    Y = X
    # graph.animate_matplotlib(X, Y, np.abs(Qdist_tot) ** 2)

    if graphic:
        path = f"strength_{Kick}"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        for k in range(kicks):
            graph.writeVtk(k, np.abs(Qdist_tot[k]) ** 2, M, X[1] - X[0], path)

    return Qdist_tot


if __name__ == "__main__":

    K_array = np.around(np.linspace(minK, maxK, processes), 1)
    print(K_array)
    # K_array = np.around(K_array, 1)

    plt.figure()
    plt.title("Mean Energy", fontsize=20)
    plt.xlabel("kicks", fontsize=15)
    plt.ylabel(r"$<E>$", fontsize=15)
    plt.axvline(x=2.5, linestyle="dotted", label="range of quantum break times")
    plt.axvline(x=4, linestyle="dotted")

    if multiproc:

        with multiprocessing.Pool(processes=len(K_array)) as pool:

            Qmp = np.array(pool.map(KRmain, K_array), dtype="object")
            pool.close()
            pool.join()

        for k in range(len(K_array)):

            E, _ = MeanEnergy(Qmp[k])
            plt.scatter(range(len(E)), E, s=5, label=f"K = {round(K_array[k], 1)}")

    else:

        path = "data"

        if executor:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            c = 0
            for k in K_array:

                QTOT = KRmain(k)
                QTOT = np.array(np.reshape(QTOT, (kicks, M ** 2)))
                np.savetxt(f"{path}/K{c}.txt", QTOT)
                c += 1

        # for k in K_array:
        c = 0
        for files in os.listdir(f"{path}"):

            fileso = os.path.join(path, files)
            # QTOT = open(f"{path}/K.{k}.txt")
            print(fileso)
            QTOT = np.loadtxt(f"{fileso}", dtype=complex)
            QTOT = np.reshape(QTOT, (kicks, M, M))
            E, _ = MeanEnergy(QTOT)
            plt.plot(range(len(E)), E, label=f"K = {round(K_array[c], 1)}")
            c += 1

    plt.legend()
    plt.show()

