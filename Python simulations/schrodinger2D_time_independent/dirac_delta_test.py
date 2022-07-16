import numpy as np
import matplotlib.pyplot as plt

plot = True

eigen = np.loadtxt("eigenvalues_Sinai.txt", dtype=complex).real
analysis = 50  # number of eigenvalues for the analysis
eigen_comb = eigen[:analysis]
spacing = []
for i in range(1, analysis - 1):
    spacing.append(
        max(eigen_comb[i + 1] - eigen_comb[i], eigen_comb[i] - eigen_comb[i - 1])
    )

if (
    plot
):  # returns the dirac comb (pettine di Dirac) for the number of eigenvalues selected
    uni = np.ones(analysis)
    plt.stem(eigen_comb, uni)
    plt.show()

    print(
        max(spacing)
    )  # analyze delta dirac on the spacing... in the max concentration of the spacing one can visualize the distribution of the RMT predictions
    unispacing = np.ones(len(spacing))
    plt.stem(spacing, unispacing)
    plt.show()
