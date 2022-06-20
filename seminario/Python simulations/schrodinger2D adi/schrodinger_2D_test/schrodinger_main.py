import numpy as np
from numba import njit, float64, int32

Nx=400
Ny=400
Nt=10000
dt=2e-3
x=np.linspace(0, 1, Nx)
y=np.linspace(0, 1, Ny)

dx=x[1]-x[0]
dy=y[1]-y[0]

mu, sigma=1, 1/10
D=0.5

alphax=D*dt/(2*dx**2)
alphay=D*dt/(2*dy**2)

def init_coherent_state(psi, complex_part=False):
    if not complex_part:
        for i in range(Nx):
            for j in range(Ny):
                psi[i, j]=np.exp(-(x-mu)**2/(2*sigma)-(y-mu)**2/(2*sigma))
    
    return psi

def V(x, y):
    return V

@njit(
    float64[:](int32, float64[:], float64[:], float64[:], float64[:]),
    fastmath=True,
    cache=True,
)  # questa funzione risolve il sistema di equazioni invece di invertire matrici
def solve_matrix(n, lower_diagonal, main_diagonal, upper_diagonal, solution_vector):

    """Solve systems of equations through Thomas Algorithm instead of inverting matrices. It returns
       the same solution of np.linalg.solve"""

    w = np.zeros(n - 1)
    g = np.zeros(n)
    result = np.zeros(n)

    w[0] = upper_diagonal[0] / main_diagonal[0]
    g[0] = solution_vector[0] / main_diagonal[0]

    for i in range(1, n - 1):
        w[i] = upper_diagonal[i] / (main_diagonal[i] - lower_diagonal[i - 1] * w[i - 1])
    for i in range(1, n):
        g[i] = (solution_vector[i] - lower_diagonal[i - 1] * g[i - 1]) / (
            main_diagonal[i] - lower_diagonal[i - 1] * w[i - 1]
        )
    result[n - 1] = g[n - 1]
    for i in range(n - 1, 0, -1):
        result[i - 1] = g[i - 1] - w[i - 1] * result[i]
    return result  # restituisce la stessa soluzione che con linalg.solve

def implicit_x_explicit_y(psi, psi_new):
    main=np.ones(Nx)
    up_diag=np.ones(Nx)*(-alphax)
    low_diag=np.ones(Nx)*(-alphax)
    step=np.zeros(Nx)
    temp=np.zeros(Nx)
    for j in range(1, Ny-1):
        for i in range(Nx):
            main[i]=1+2*alphax-dt/2*V(x[i], y[j])
        for i in range(1, Nx-1):
            step[i]=psi[i, j]+alphay*(psi[i, j+1]-2*psi[i, j]+psi[i, j-1])+V(x[i], y[j])*psi[i, j]
        temp=solve_matrix(Nx, low_diag[1:], main, up_diag[:Nx-1], step)
        for i in range(Nx):
            psi_new[i,j]=temp[i]
    return psi_new

def implicit_y_explicit_x(psi, psi_new):
    main=np.ones(Ny)
    up_diag=np.ones(Ny)
    low_diag=np.ones(Ny)
    step=np.zeros(Ny)
    temp=np.zeros(Ny)
    for i in range(1, Nx-1):
        for j in range


        

