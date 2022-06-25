import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
import shutil
import os

def check_path(path): 
    '''crea il path per i file .vtk'''
    if os.path.exists(path)==True: #stringa da scrivere nel rispettivo programma
        shutil.rmtree(path)
    os.mkdir(path)

def writeVtk(count, u, N, h, cammino):
    fp = open(f"{cammino}/data_" + str(count) + ".vtk", "w")
    fp.write("# vtk DataFile Version 4.1 \n")
    fp.write("COMMENT\n")
    fp.write("ASCII\n")
    fp.write("DATASET STRUCTURED_POINTS \n")
    fp.write("DIMENSIONS " + str(N) + " " + str(N) + " 1 \n")
    fp.write("ORIGIN 0 0 0\n")
    fp.write("SPACING " + str(h) + " " + str(h) + " 0 \n")
    fp.write("POINT_DATA " + str(N*N) + "\n")
    fp.write("SCALARS U double 1\n")
    fp.write("LOOKUP_TABLE default\n")
    for i in range(N):
        for j in range(N):
            fp.write(str(u[i, j]) + "\n")
    fp.close()

def animate_matplotlib(x, y, u_della_vita):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    print(u_della_vita)
    X, Y = np.meshgrid(x, y)
    def update_plot(frame_number, u_della_vita, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X, Y, u_della_vita[frame_number,:,:], cmap="terrain")
        
    plot = [ax.plot_surface(X,Y, u_della_vita[0,:,:], color='0.75', rstride=1, cstride=1)] #first frame
    # ax.set_zlim(-1,1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ani= animation.FuncAnimation(fig, update_plot, 50, fargs=(u_della_vita, plot), interval=200)
    plt.show()

def static_plot(x, y, u):
    fig = plt.figure(figsize=(11, 8), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, u[:, :], cmap=cm.viridis, rstride=1, cstride=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()