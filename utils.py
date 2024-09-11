import numpy as np

def create_grid(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1), n=(10,10,10)):
    #create a grid of points
    xl = np.linspace(*xlim, n[0])
    yl = np.linspace(*ylim, n[1])
    zl = np.linspace(*zlim, n[2])
    grid = np.zeros((n[0]*n[1]*n[2],3), dtype=np.float32)
    for xi, x in enumerate(xl):
        for yi, y in enumerate(yl):
            for zi, z in enumerate(zl):
                grid[xi*n[1]*n[2] + yi*n[2] + zi] = np.array([x,y,z])
    return grid

def create_horiz_circular_path(n=8, r=2.0, z=1.0):
    #create a wire path
    t = np.linspace(0, 2*np.pi, n+1)
    x = r*np.cos(t)
    y = r*np.sin(t)
    z = np.ones_like(x)*z
    wp = np.array([x,y,z]).T
    return wp

def create_toroidal_coils_paths(R=6.2, N=18, samples=100):
    # b2d_pts = np.array([[.8,0],[.8,1],[.9,1.3],[1.2,0]])
    bcp = np.array([[.8,1],[.8,1.3],[1.2,1.3],[1.2,0]])
    #create a bezier curve with bcp as control points
    def bernstein_poly(i, n, t):
        from scipy.special import comb
        """The Bernstein polynomial of n, i as a function of t"""
        return comb(n,i) * (t**(n-i)) * (1-t)**i

    def bezier_curve(pts, nt=100):
        n = len(pts)
        t = np.linspace(0.0, 1.0, nt)
        polynomial_array = np.array([ bernstein_poly(i, n-1, t) for i in range(n)])
        xvals = np.dot(pts[:,0], polynomial_array)
        yvals = np.dot(pts[:,1], polynomial_array)
        return np.array([xvals, yvals, np.zeros_like(xvals)]).T

    b2d = bezier_curve(bcp, samples)
    return [b2d]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from tests import FIGSIZE
    from fem import FemWire
    tor_wires_paths = create_toroidal_coils_paths()
    tor_wires = [FemWire(path, V=50, seg_len=0.1) for path in tor_wires_paths]

    #create a figure and 3d plot all wires
    fig = plt.figure(figsize=FIGSIZE)  # big figure just to makeit full screen
    ax = fig.add_subplot(projection='3d')
    for w in tor_wires: w.plot(ax, color='r') #plot wires
    plt.tight_layout()
    plt.show()

