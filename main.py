from interfaces import Wire, Grid, MagField, create_example_path
from symbolic import SymWire, SymMagField
from fem import FemWire, FemMagField

import numpy as np, matplotlib.pyplot as plt

if __name__ == '__main__':
    #create a wire path
    wp = create_example_path()
    print(f'wp.shape={wp.shape}')
    #create a grid
    grid = Grid((-3,3), (-3,3), (-3,3), n=(15,15,5))

    ## FEM
    #create a wire
    w = FemWire(wp, seg_len=0.1)
    #create a magnetic field
    mf = FemMagField([w])
    #calculate magnetic field
    mf.calc(grid)
    #plot magnetic field
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mf.quiver(ax, grid, length=0.8, normalize=True, 
              color=plt.cm.viridis(mf.normB), arrow_length_ratio=0.0)
    
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
    
    plt.show()

