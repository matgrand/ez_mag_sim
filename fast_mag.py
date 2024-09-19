import numpy as np
from time import time

# def calc_mag_field(w:np.ndarray, I, grpts:np.ndarray): #calculate magnetic field, wpaths: wire paths, wIs: wire currents, grpts: grid points
#     B = np.zeros_like(grpts) #initialize B field (n,3)
#     μ0 = 4*np.pi*1e-7 #vacuum permeability
#     wa, wb = w, np.roll(w, -1, axis=0)  # wire points (m,3)
#     dl = wb - wa  # dl (m,3)
#     wm = (wa + wb) / 2  # wire middle (m,3)
#     for i, p in enumerate(grpts): # n times
#         r = p - wm  # r (m,3)
#         rnorm = np.linalg.norm(r, axis=1).reshape(-1,1)  # |r| (m,1)
#         B[i] += μ0 * I * np.sum(np.cross(dl, r) / rnorm**3, axis=0) / 4*np.pi # Biot-Savart law
#     return B

def calc_mag_field(w:np.ndarray, I, grpts:np.ndarray): #calculate magnetic field, wpaths: wire paths, wIs: wire currents, grpts: grid points
    B = np.zeros_like(grpts) #initialize B field (n,3)
    μ0 = 4*np.pi*1e-7 #vacuum permeability
    wa, wb = w, np.roll(w, -1, axis=0)  # wire points (m,3)
    dl = wb - wa  # dl (m,3)
    wm = (wa + wb) / 2  # wire middle (m,3)
    for i, p in enumerate(grpts): # n times
        r = p - wm  # r (m,3)
        B[i] += μ0 * I * np.sum(np.cross(dl, r) / (np.linalg.norm(r, axis=1).reshape(-1,1))**3, axis=0) / 4*np.pi # Biot-Savart law
    return B

# create a wire 
angles = np.linspace(0, 2*np.pi, 1000)
w = np.array([np.cos(angles), np.sin(angles), np.ones_like(angles)]).T
print(f'wire: {w.shape}')

# create a grid
grpts = np.random.rand(10000,3)
print(f'grid: {grpts.shape}')

# calculate magnetic field
times = []
for i in range(4):
    start = time()
    B = calc_mag_field(w, 100.0, grpts)
    times.append(time()-start)

print(f'avg time: {np.mean(times):.4f} s')