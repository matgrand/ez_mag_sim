import numpy as np, matplotlib.pyplot as plt
from numpy import ndarray
from tqdm import tqdm
from time import time

#define wire interface
class Wire():
    def __init__(self, wp, V=0.0, ρ=1.77e-8, section=1e-4):
        self._wp, self._V, self._ρ, self._s = wp, V, ρ, section
        self._R, self._L, self._I = None, None, None

    def plot(self, ax, **kwargs):
        raise NotImplementedError

    def __str__(self) -> str:
        return f'Wire: V={self.V:.2f} V, ρ={self.ρ:.2e} Ωm, s={self.s:.2e} m^2, L={self.L:.2f} m, R={self.R:.2e} Ω, I={self.I:.2e} A'

    @property #current
    def I(self): return self._I
    @property #resistance
    def R(self): return self._R
    @property #length
    def L(self): return self._L
    @property #wire path
    def wp(self): return self._wp 
    @property #voltage
    def V(self): return self._V
    @property #resistivity
    def ρ(self): return self._ρ 
    @property #section
    def s(self): return self._s 

class MagField():
    def __init__(self, wires):
        self._wires = wires
        self._B = None
        self._normB = None

    def calc(self, grid:ndarray):
        raise NotImplementedError
    
    def quiver(self, ax:plt.Axes, grid:ndarray, **kwargs):
        assert isinstance(ax, plt.Axes), 'ax must be a matplotlib Axes object'
        assert isinstance(grid, ndarray) and grid.shape[1] == 3, 'grid must be a (n,3) array'
        assert self.B is not None, 'magnetic field must be calculated first'
        ax.quiver(grid.grid[:,0], grid.grid[:,1], grid.grid[:,2], self.B[:,0], self.B[:,1], self.B[:,2], **kwargs)
        return ax
    
    def plot_field_lines(self, ax:plt.Axes, something, **kwargs):
        raise NotImplementedError

    def __str__(self) -> str:
        return f'Magnetic Field: {self.B} T'

    @property #magnetic field
    def B(self): return self._B
    @property #norm of magnetic field
    def normB(self): return self._normB
    @property #wires
    def wires(self): return self._wires

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



