from interfaces import Wire, MagField, create_example_path, create_grid
from symbolic import SymWire, SymMagField
from fem import FemWire, FemMagField
from time import time

import numpy as np, matplotlib.pyplot as plt

if __name__ == '__main__':
   
    grid = create_grid((-3,3), (-3,3), (-3,3), n=(15,15,5)) #create a grid
    wp1 = create_example_path(n=3, r=2.0, z=0.0) #create a wire path
    wp2 = create_example_path(n=4, r=2.0, z=1.5) #create a wire path
    wp3 = create_example_path(n=5, r=2.0, z=-1.5) #create a wire path

    ## FEM
    w1 = FemWire(wp1, V=50, seg_len=0.1) #create a wire
    w2 = FemWire(wp2, V=40, seg_len=0.1) #create a wire
    w3 = FemWire(wp3, V=-40, seg_len=0.1) #create a wire
    wires = [w1, w2, w3] 
    # wires = [w1] 

    mf = FemMagField(wires) #create a magnetic field
    calcs = time()
    mf.calc(grid) #calculate magnetic field
    calce = time()
    print(f'calc0: {calce-calcs:.3f} s')

    #plot magnetic field and wire
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = np.log(1000*np.clip(mf.normB, 0, 0.01))
    mf.quiver(ax, grid, length=0.4, normalize=True, 
              color=plt.cm.viridis(cmap), arrow_length_ratio=0.0)
    for w in wires: w.plot(ax, color='r') #plot wires
    ax.scatter(grid[:,0], grid[:,1], grid[:,2], s=1, color='k') #plot grid points

    # ## SYMBOLIC
    # #create a wire
    # w = SymWire(wp=wp, V=50)
    # #create a magnetic field
    # mf = SymMagField([w])
    # #calculate magnetic field
    # mf.calc(grid)
    # #plot magnetic field
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # mf.quiver(ax, grid, length=0.8, normalize=True, 
    #           color=plt.cm.viridis(mf.normB), arrow_length_ratio=0.0)
    
    plt.tight_layout()
    plt.show()

