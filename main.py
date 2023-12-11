from interfaces import Wire, MagField, create_example_path, create_grid
from symbolic import SymWire, SymMagField
from fem import FemWire, FemMagField
from time import time

import numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.quiver import Quiver
#import Axes3D for 3d plotting
from mpl_toolkits.mplot3d import Axes3D

#set numpy print options
np.set_printoptions(precision=2, suppress=True, linewidth=200)

def test_streamplot():
    #streamplot magnetic field
    fig = plt.figure(figsize=(80,80))  # big figure just to makeit full screen
    g3, b3 = grid.reshape((ix,iy,iz,3)), mf.B.reshape((ix,iy,iz,3))
    # gp3x, gp3y, gp3z = g3[0,:,:,0], g3[0,:,:,1], g3[0,:,:,2]
    # bp3x, bp3y, bp3z = b3[0,:,:,0], b3[0,:,:,1], b3[0,:,:,2]
    gp3x, gp3y, gp3z = g3[:,:,10,0], g3[:,:,10,1], g3[:,:,10,2]
    bp3x, bp3y, bp3z = b3[:,:,10,0], b3[:,:,10,1], b3[:,:,10,2]
    print(f'gp3x: \n{gp3x}')
    print(f'gp3y: \n{gp3y}')
    print(f'gp3z: \n{gp3z}')
    # plt.streamplot(gp3y.T, gp3z.T, bp3y.T, bp3z.T)
    plt.streamplot(gp3x.T, gp3y.T, bp3x.T, bp3y.T, density=2.5)
    plt.axis('equal')
    plt.show()


def test_magfield_plot():
    #plot magnetic field and wire
    fig = plt.figure(figsize=(80,80))  # big figure just to makeit full screen
    ax = fig.add_subplot(111, projection='3d')
    dec = max(1, ix*iy*iz//4999)
    cmap = np.log(1000*np.clip(mf.normB[::dec], 0, 0.01))
    print(f'dec={dec}')
    mf.quiver(ax, grid, dec=dec, length=0.4, normalize=True, 
              color=plt.cm.viridis(cmap), arrow_length_ratio=0.0)
    for w in wires: w.plot(ax, color='r') #plot wires
    # ax.scatter(grid[:,0], grid[:,1], grid[:,2], s=1, color='k') #plot grid points
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(-3,3)
    plt.tight_layout()
    plt.show()


def test_magfield_animation():
    pass

if __name__ == '__main__':
    ix, iy, iz = 20,20,20
    grid = create_grid((-3,3), (-3,3), (-3,3), n=(ix,iy,iz)) #create a grid
    wp1 = create_example_path(n=3, r=2.0, z=0.0) #create a wire path
    wp2 = create_example_path(n=4, r=2.0, z=1.5) #create a wire path
    wp3 = create_example_path(n=5, r=2.0, z=-1.5) #create a wire path

    ## FEM
    w1 = FemWire(wp1, V=50,  seg_len=0.1) #create a wire
    w2 = FemWire(wp2, V=40,  seg_len=0.1) #create a wire
    w3 = FemWire(wp3, V=-40, seg_len=0.1) #create a wire
    wires = [w1, w2, w3] 
    # wires = [w1] 

    mf = FemMagField(wires) #create a magnetic field
    calcs = time()
    mf.calc(grid) #calculate magnetic field
    calce = time()
    print(f'calc0: {calce-calcs:.3f} s')
    print(f'mean of mf norm: {np.mean(mf.normB):.5f} T')

    # TESTS

    # test_magfield_plot()

    # test_streamplot()

    # test_magfield_animation()
    fig = plt.figure(figsize=(80,80))  # big figure just to makeit full screen
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=(-3,3), ylim=(-3,3), zlim=(-3,3), xlabel='x', ylabel='y', zlabel='z')
    ax.quiver(grid[0,0], grid[0,1], grid[0,2], mf.B[0,0], mf.B[0,1], mf.B[0,2],
                        length=0.4, normalize=True, color=plt.cm.viridis(0), arrow_length_ratio=0.0)
    for w in wires: w.plot(ax, color='r') #plot wires

    def update(id):
        #randomly select elements of grid and quiver them
        ni = 50
        idx = np.random.randint(0, ix*iy*iz, ni)
        gridi = grid[idx]
        Bi = mf.B[idx]
        cmap = np.log(1000*np.clip(mf.normB, 0, 0.01))
        ax.clear()
        ax.set(xlim=(-3,3), ylim=(-3,3), zlim=(-3,3), xlabel='x', ylabel='y', zlabel='z')
        ax.quiver(gridi[:,0], gridi[:,1], gridi[:,2], Bi[:,0], Bi[:,1], Bi[:,2],
                        length=0.4, normalize=True, color=plt.cm.viridis(cmap[idx]), arrow_length_ratio=0.0)
        for w in wires: w.plot(ax, color='r') #plot wires
        print(f'idx: {id}')
        return ax

    ani = animation.FuncAnimation(fig=fig, func=update, frames=10, interval=20, blit=False, repeat=False)
    plt.show()




