import numpy as np
import matplotlib.pyplot as plt


def FFT(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    """
    N = len(x)

    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)

        X = np.concatenate(
            [
                X_even + factor[: int(N / 2)] * X_odd,
                X_even + factor[int(N / 2) :] * X_odd,
            ]
        )
        return X


def multiple2(x):

    """from given array return the array elements near the highest possible power of 2"""

    n = len(x)
    exp = 0

    while (n) > 2:
        exp += 1
        n = n / 2
    print("Power of 2:", exp)

    return x[: 2 ** exp]


eigen = np.real(np.loadtxt("eigenvalues_mod_Bunimovich.txt", dtype=complex))
eigen = multiple2(eigen)
print(len(eigen))
sample_rate = 200
# calculate the frequencies:
n = len(eigen)
N = np.arange(n)
T = n / sample_rate
freq = N / T

e = FFT(eigen)
efft = abs(np.fft.fft(eigen))
plt.plot(freq, np.abs(e))
plt.plot(freq, efft)
plt.show()

