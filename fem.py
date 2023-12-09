from interfaces import Wire, MagField
import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from numpy import ndarray

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
    def __init__(self, wires:list):
        super().__init__(wires)
        
    def calc(self, grid:ndarray):
        # calculate B field on a grid
        assert grid.shape[1] == 3, f'grid must be a (n,3) array, not {grid.shape}'
        self._B = np.zeros_like(grid) #initialize B field
        #calculate B field
        μ0 = 4*np.pi*1e-7 #vacuum permeability
        for w in self.wires: #for each wire
            n, m = grid.shape[0], w.wp.shape[0] # n=grid points, m=wire points
            wp1, wp2 = w.wp, np.roll(w.wp, -1, axis=0)  # wire points (m,3)
            dl = wp2 - wp1  # dl (m,3)
            wm = (wp1 + wp2) / 2  # wire middle (m,3)
            r = np.zeros((n,m,3), dtype=np.float32) # r (n,m,3)
            #print ram used by r
            print(f'ram used by r: {r.nbytes/1e6:.2f} MB')
            ps = np.repeat(grid.reshape(n,1,3), m, axis=1) # grid points (n,1,3) -> (n,m,3)
            r = ps - wm # r (n,m,3)
            rnorm = np.linalg.norm(r, axis=-1).reshape(n,m,1) # |r| (n,m,1)
            r̂ = r / rnorm # unit vector r (n,m,3)
            #calculate B field with Biot-Savart law
            self._B += np.sum(μ0*w.I*np.cross(dl,r̂)/(4*np.pi*rnorm**2), axis=1)

        self._normB = np.linalg.norm(self.B, axis=1)
        return self._B
    
    def calc_slower(self, grid:ndarray):
        # this is slower than calc, but uses less ram
        # calculate B field on a grid
        assert grid.shape[1] == 3, f'grid must be a (n,3) array, not {grid.shape}'
        self._B = np.zeros_like(grid) #initialize B field
        #calculate B field, n=grid points, m=wire points
        μ0 = 4*np.pi*1e-7 #vacuum permeability
        for wi, w in enumerate(self.wires): #for each wire
            wp1, wp2 = w.wp, np.roll(w.wp, -1, axis=0)  # wire points (m,3)
            dl = wp2 - wp1  # dl (m,3)
            wm = (wp1 + wp2) / 2  # wire middle (m,3)
            for i, p in enumerate(tqdm(grid, desc=f'{wi+1}/{len(self.wires)}')): # n times
                r = p - wm  # r (m,3)
                rnorm = np.linalg.norm(r, axis=1).reshape(-1,1)  # |r| (m,1)
                r̂ = r / rnorm  # unit vector r (m,3)
                self._B[i] += np.sum(
                    μ0 * w.I * np.cross(dl, r̂) / (4*np.pi*rnorm**2),
                    axis=0,
                )  # Biot-Savart law
        self._normB = np.linalg.norm(self.B, axis=1)
        return self._B
    
    def calc_slowest(self, grid:ndarray):
        # this is extremely slow, but may be useful to understand the algorithm
        # calculate B field on a grid
        assert grid.shape[1] == 3, f'grid must be a (n,3) array, not {grid.shape}'
        self._B = np.zeros_like(grid) #initialize B field
        #calculate B field
        μ0 = 4*np.pi*1e-7 #vacuum permeability
        for wi, w in enumerate(self.wires): #for each wire
            for i, p in enumerate(tqdm(grid, desc=f'{wi+1}/{len(self.wires)}')):
                for j in range(len(w.wp)):
                    wp1, wp2 = w.wp[j], w.wp[(j+1)%len(w.wp)]
                    dl = wp2 - wp1 #dl
                    r = p - (wp1+wp2)/2 #r
                    rnorm = np.linalg.norm(r) #|r|
                    r̂ = r/rnorm # unit vector r
                    dlnorm = np.linalg.norm(dl) #|dl|
                    dl̂ = dl/dlnorm # unit vector dl
                    self._B[i] += dlnorm*μ0*w.I*np.cross(dl̂, r̂)/(4*np.pi*rnorm**2) #Biot-Savart law
        self._normB = np.linalg.norm(self.B, axis=1)
        return self._B

    
        
    def quiver(self, ax:plt.Axes, grid:ndarray, **kwargs):
        assert isinstance(ax, plt.Axes), 'ax must be a matplotlib Axes object'
        assert isinstance(grid, ndarray) and grid.shape[1] == 3, 'grid must be a (n,3) array'
        assert self.B is not None, 'B field must be calculated first'
        self._normB = np.linalg.norm(self.B, axis=1)
        ax.quiver(grid[:,0], grid[:,1], grid[:,2], 
                  self.B[:,0], self.B[:,1], self.B[:,2], **kwargs)
        return ax