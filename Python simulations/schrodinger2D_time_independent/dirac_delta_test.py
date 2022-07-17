from turtle import color
import numpy as np
import matplotlib.pyplot as plt

plot = True
pot = "cardioid"
eigen = np.loadtxt(f"unfolded_spectrum_{pot}.txt", dtype=complex)
print(len(eigen))
analysis = 400
eigen_comb = eigen[:analysis]
spacing = []
for i in range(1, analysis - 1):
    spacing.append(
        max(eigen_comb[i + 1] - eigen_comb[i], eigen_comb[i] - eigen_comb[i - 1])
    )

if plot:
    uni = np.ones(analysis)  # avevo detto che erano 2 righe... mannaggialamaronna
    plt.stem(eigen_comb, uni, linefmt="blue")
    plt.ylabel(r"$ \rho(E) $", fontsize=15)
    plt.xlabel("E", fontsize=15)
    plt.title(f"Densità degli stati per {pot} (unfolded)", fontsize=22)
    plt.legend()
    plt.show()

    print(max(spacing))
    unispacing = np.ones(len(spacing))
    plt.stem(spacing, unispacing)
    plt.show()

number_neg = 0
number_pos = 0

for e in eigen_comb:
    if e < 0:
        number_neg += 1
    else:
        number_pos += 1

print(number_neg, number_pos)
