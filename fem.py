from interfaces import Wire, Grid, MagField
import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

class FemWire(Wire):
    def __init__(self, wp, V=0.0, ρ=1.77e-8, section=1e-4, seg_len=5e-2):
        super().__init__(wp, V, ρ, section)
        self._seg_len = seg_len
        assert self._wp.shape[1] == 3, f'wire_path must be a (n,3) array, not {self.wp.shape}'

        #calculate legnth of wire
        diff = self.wp - np.roll(self.wp, 1, axis=0) #difference between points
        self._L = np.sum(np.sqrt(np.sum(diff**2, axis=1)))
        print(f'L={self.L:.2f} m')

        #resample wire path with segments of length similar to seg_len
        w = []
        for i in range(len(self.wp)):
            p1, p2 = self.wp[i], self.wp[(i+1)%len(self.wp)]
            l = np.linalg.norm(p2-p1)
            n = int(l/self.seg_len)
            for j in range(n):
                w.append(p1 + (p2-p1)*j/n)
        self._wp = np.array(w)

        self._R = self._ρ*self._L/self._s # resistance R = ρ*L/A
        self._I = self._V/self._R # current I = V/R

    def plot(self, ax:plt.Axes, **kwargs):
        for i in range(len(self.wp)):
            p1, p2 = self.wp[i], self.wp[(i+1)%len(self.wp)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], **kwargs)
            # ax.scatter(p1[0], p1[1], p1[2], **kwargs)
        return ax

    @property #segment length
    def seg_len(self): return self._seg_len
    
class FemMagField(MagField):
    def __init__(self, wires:[FemWire]):
        super().__init__(wires)
        
    def calc(self, grid_obj:Grid):
        grid = grid_obj.grid
        # calculate B field on a grid
        assert grid.shape[1] == 3, f'grid must be a (n,3) array, not {grid.shape}'
        self._B = np.zeros_like(grid) #initialize B field
        #calculate B field
        for wi, w in enumerate(self._wires):
            μ0 = 4*np.pi*1e-7 #vacuum permeability
            for i, p in enumerate(tqdm(grid, desc=f'B field {wi+1}/{len(self._wires)}', leave=True, ncols=80)):
                wp1, wp2 = w.wp, np.roll(w.wp, -1, axis=0)
                dl = wp2 - wp1  # dl|
                r = p - (wp1 + wp2) / 2  # r
                rnorm = np.linalg.norm(r, axis=1).reshape(-1,1)  # |r|
                r̂ = r / rnorm  # unit vector r
                self._B[i] += np.sum(
                    μ0 * w.I * np.cross(dl, r̂) / (4 * np.pi * rnorm ** 2),
                    axis=0,
                )  # Biot-Savart law
            return self._B
        
    def quiver(self, ax:plt.Axes, grid:Grid, **kwargs):
        assert isinstance(ax, plt.Axes), 'ax must be a matplotlib Axes object'
        assert isinstance(grid, Grid), 'grid must be a Grid object'
        assert self._B is not None, 'B field must be calculated first'
        self._normB = np.linalg.norm(self._B, axis=1)
        ax.quiver(grid.grid[:,0], grid.grid[:,1], grid.grid[:,2], 
                  self._B[:,0], self._B[:,1], self._B[:,2], **kwargs)
        return ax