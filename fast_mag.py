import numpy as np
from time import time

# print with x decimal places
np.set_printoptions(precision=4)

def calc_mag_field(w:np.ndarray, I, grpts:np.ndarray): #calculate magnetic field, wpaths: wire paths, wIs: wire currents, grpts: grid points
    B = np.zeros_like(grpts) #initialize B field (n,3)
    μ0 = 4*np.pi*1e-7 #vacuum permeability
    wa, wb = w, np.roll(w, -1, axis=0)  # wire points (m,3)
    dl = wb - wa  # dl (m,3)
    wm = (wa + wb) / 2  # wire middle (m,3)
    for i, p in enumerate(grpts): # n times
        r = p - wm  # r (m,3)
        B[i] += μ0 * I * np.sum(np.cross(dl, r) / (np.linalg.norm(r, axis=1).reshape(-1,1))**3, axis=0) / (4*np.pi) # Biot-Savart law
    return B

# create a wire 
M = 1000
angles = np.linspace(0, 2*np.pi, M)
w = np.array([np.cos(angles), np.sin(angles), np.ones_like(angles)]).T
print(f'wire: {w.shape}')

# create a grid
N = 10000
# grpts = np.random.rand(N,3)
grpts = np.array([[i/N,i/N,i/N] for i in range(1,N+1)])
print(f'grid: {grpts.shape}')

# calculate magnetic field
times = []
for i in range(1):
    start = time()
    B = calc_mag_field(w, 1_000_000.0, grpts)
    times.append(time()-start)

print(f'avg time: {np.mean(times):.4f} s')

#print the 10 first grid points and the corresponding B field
for i in range(5):
    print(f'grid: {grpts[i]}, B: {B[i]}')