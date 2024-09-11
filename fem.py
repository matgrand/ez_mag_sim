import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from numpy import ndarray

CHUNCK_SIZE = 1e8
# T = np.float32
T = np.float64

class FemWire():
    def __init__(self, wp, V=0.0, ρ=1.77e-8, section=1e-4, seg_len=5e-2):
        self._wp, self._V, self._ρ, self._s = wp, V, ρ, section
        self._R, self._L, self._I = None, None, None
        self._seg_len = seg_len
        assert self._wp.shape[1] == 3, f'wire_path must be a (n,3) array, not {self.wp.shape}'
        self._in_wp = self._wp.copy() #initial wire path (before resampling)

        #calculate legnth of wire
        diff = self.wp - np.roll(self.wp, 1, axis=0) #difference between points
        self._L = np.sum(np.sqrt(np.sum(diff**2, axis=1)))

        #resample wire path with segments of length similar to seg_len
        w = []
        for i in range(len(self.wp)):
            p1, p2 = self.wp[i], self.wp[(i+1)%len(self.wp)]
            l = np.linalg.norm(p2-p1)
            n = int(l/self.seg_len)
            for ii in range(n):
                w.append(p1 + (p2-p1)*ii/n)
        self._wp = np.array(w)

        self._R = self._ρ*self._L/self._s # resistance R = ρ*L/A
        self._I = self._V/self._R # current I = V/R

    def __str__(self) -> str:
        return f'Wire: V={self.V:.2f} V, ρ={self.ρ:.2e} Ωm, s={self.s:.2e} m^2, L={self.L:.2f} m, R={self.R:.2e} Ω, I={self.I:.2e} A'

    def plot(self, ax:plt.Axes, **kwargs):
        ax.plot(self._in_wp[:,0], self._in_wp[:,1], self._in_wp[:,2], **kwargs)
        return ax

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
    @property #segment length
    def seg_len(self): return self._seg_len
    
class FemMagField():
    def __init__(self, wires:list):
        self._wires = wires
        self._B = None
        self._normB = None
        
    def calc(self, grid:ndarray, normalized=False):
        # calculate B field on a grid
        assert grid.shape[1] == 3, f'grid must be a (n,3) array, not {grid.shape}'
        self._B = np.zeros_like(grid, dtype=T) #initialize B field
        #calculate B field
        μ0 = 4*np.pi*1e-7 #vacuum permeability
        for iw, w in enumerate(self.wires): #for each wire 
            n, m = grid.shape[0], w.wp.shape[0] # n=grid points, m=wire points
            #split grid into chuncks, to limit ram usage
            n_chuncks = int(np.ceil(n*m/CHUNCK_SIZE))
            grid_chuncks = np.array_split(grid.astype(T), n_chuncks)

            #calculate B field for each chunck
            B_idx = 0
            iterator = tqdm(grid_chuncks, desc=f'mf {iw}', ncols=80, leave=False) if len(grid_chuncks) > 1 else grid_chuncks
            for gc in iterator:
                nc = len(gc) # n=grid points
                wp1, wp2 = w.wp.astype(T), np.roll(w.wp, -1, axis=0).astype(T)  # wire points (m,3)
                dl = wp2 - wp1  # dl (m,3)
                wm = (wp1 + wp2) / 2  # wire middle (m,3)
                r = np.zeros((nc,m,3), dtype=T) # r (n,m,3)
                #print ram used by r
                ps = np.repeat(gc.reshape(nc,1,3), m, axis=1) # grid points (n,1,3) -> (n,m,3)
                r = ps - wm # r (n,m,3)
                rnorm = np.linalg.norm(r, axis=-1).reshape(nc,m,1) # |r| (n,m,1)
                r̂ = r / rnorm # unit vector r (n,m,3)
                assert wp1.dtype == T and wp2.dtype == T and dl.dtype == T and wm.dtype == T and r.dtype == T and rnorm.dtype == T and r̂.dtype == T, 'dtype error'
                #calculate B field with Biot-Savart law
                Bc = np.sum(μ0*w.I*np.cross(dl,r̂)/(4*np.pi*rnorm**2), axis=1)
                #add B field to total B field
                self._B[B_idx:B_idx+nc] += Bc
                B_idx += nc #update B_idx

        self._normB = np.linalg.norm(self.B, axis=-1)
        return self._B if not normalized else self._B / self._normB.reshape(-1,1)
    
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
                for ii in range(len(w.wp)):
                    wp1, wp2 = w.wp[ii], w.wp[(ii+1)%len(w.wp)]
                    dl = wp2 - wp1 #dl
                    r = p - (wp1+wp2)/2 #r
                    rnorm = np.linalg.norm(r) #|r|
                    r̂ = r/rnorm # unit vector r
                    dlnorm = np.linalg.norm(dl) #|dl|
                    dl̂ = dl/dlnorm # unit vector dl
                    self._B[i] += dlnorm*μ0*w.I*np.cross(dl̂, r̂)/(4*np.pi*rnorm**2) #Biot-Savart law
        self._normB = np.linalg.norm(self.B, axis=1)
        return self._B
    
    @property #magnetic field
    def B(self): return self._B
    @property #norm of magnetic field
    def normB(self): return self._normB
    @property #wires
    def wires(self): return self._wires