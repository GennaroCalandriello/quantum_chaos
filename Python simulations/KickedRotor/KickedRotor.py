import numpy as np
import module.graph as graph
import matplotlib.pyplot as plt
import shutil
import os

M = 300
kicks = 2
K = 0.8
x0 = 1.5
p0 = 0.3
hbar = 2 * np.pi / M


def waveFunc(M, x0):
    n = np.arange(M)
    x = 2 * np.pi * n / M
    psi0 = np.zeros(M, dtype=complex)
    psi0[:] = np.exp(-((x - x0) ** 2) / 2 / hbar)
    psi0 = psi0 / np.linalg.norm(psi0)
    return x, psi0


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
    """#unitary evolution operator U=U(kick)*U(free)"""
    Ukick = np.exp(-1j / hbar * K * np.cos(x))
    Ufree = np.exp(-1j / (2 * hbar) * p ** 2)
    return Ukick, Ufree


def evolveOneStep(V, P, psi_x):
    M = psi_x.size
    fm = V * psi_x
    sm = P * np.fft.fft(fm)
    s = np.fft.ifft(sm)
    return s


def StdMap(n, K, x0=1.0, p0=1.0):
    x, p = np.zeros(n), np.zeros(n)
    x[0], p[0] = x0, p0
    for i in np.arange(n - 1):
        p[i + 1] = np.mod(p[i] + K * np.sin(x[i]), 2 * np.pi)
        x[i + 1] = np.mod(x[i] + p[i + 1], 2 * np.pi)
    return x, p


# std_x, std_p = StdMap(kicks + 1, K, x0, p0)
x, p, psi_x, psi_p = gaussianWavePacket(M, x0, p0)
V, P = phaseVectors(x, p, K)

Qdist = np.zeros((M, M), dtype=complex)
Qdist_tot = np.zeros((kicks, M, M), dtype=complex)

for k in np.arange(kicks):
    psi_x = evolveOneStep(V, P, psi_x)

    for i in np.arange(M):
        x0 = 2 * np.pi * i / M
        x, p, psiG_n0, psi_p = gaussianWavePacket(M, x0, 0)
        Qdist[:, i] = 1 / np.sqrt(M) * np.fft.fft(np.conj(psiG_n0) * psi_x)

    n = np.arange(M)
    Qdist_tot[k] = Qdist
X = 2 * np.pi * n / M
Y = X
# graph.animate_matplotlib(X, Y, np.abs(Qdist_tot) ** 2)
path = f"strength_{K}"
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)

for k in range(kicks):
    graph.writeVtk(k, np.abs(Qdist_tot[k]) ** 2, M, X[1] - X[0], path)

