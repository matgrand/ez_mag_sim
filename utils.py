import numpy as np

def create_grid(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1), n=(10,10,10)):
    #create a grid of points
    xl = np.linspace(*xlim, n[0])
    yl = np.linspace(*ylim, n[1])
    zl = np.linspace(*zlim, n[2])
    grid = np.zeros((n[0],n[1],n[2],3), dtype=np.float32)
    for xi, x in enumerate(xl):
        for yi, y in enumerate(yl):
            for zi, z in enumerate(zl):
                grid[xi,yi,zi] = np.array([x,y,z])
    return grid.reshape((-1,3))

def create_example_path(n=8, r=2.0, z=1.0):
    #create a wire path
    t = np.linspace(0, 2*np.pi, n+1)
    x = r*np.cos(t)
    y = r*np.sin(t)
    z = np.ones_like(x)*z
    wp = np.array([x,y,z]).T
    return wp
