import numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from numpy import ndarray

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
np.__config__.show()

# T = np.float32
T = np.float64

def create_wire(wp, V=0.0, ρ=1.77e-8, s=1e-4, seg_len=5e-2): #create a wire, V: voltage, ρ: resistivity, s: section, seg_len: segment length
    #calculate legnth of wire
    diff = wp - np.roll(wp, 1, axis=0) #difference between points
    L = np.sum(np.sqrt(np.sum(diff**2, axis=1))) #length of wire
    #resample wire path with segments of length similar to seg_len
    w = []
    for i in range(len(wp)):
        p1, p2 = wp[i], wp[(i+1)%len(wp)]
        l = np.linalg.norm(p2-p1)
        n = int(l/seg_len)
        for ii in range(n): w.append(p1 + (p2-p1)*ii/n)
    wp = np.array(w)
    R = ρ*L/s # resistance R = ρ*L/A
    I = V/R # current I = V/R
    return wp, I

# faster version of calc_mag_field
# CHUNCK_SIZE = 1e4
# def calc_mag_field(wpaths:list[ndarray], wIs:list[ndarray], gpoints:ndarray): #calculate magnetic field, wpaths: wire paths, wIs: wire currents, gpoints: gpoints points
#     # calculate B field on a gpoints
#     assert gpoints.shape[1] == 3, f'gpoints must be a (n,3) array, not {gpoints.shape}'
#     assert all([wp.shape[1] == 3 for wp in wpaths]), 'wire paths must be a (m,3) array'
#     B = np.zeros_like(gpoints, dtype=T) #initialize B field
#     μ0 = 4*np.pi*1e-7 #vacuum permeability
#     for iw, (w, I) in enumerate(zip(wpaths, wIs)): #for each wire 
#         n, m = gpoints.shape[0], w.shape[0] # n=gpoints points, m=wire points
#         #split gpoints into chuncks, to limit ram usage
#         n_chuncks = int(np.ceil(n*m/CHUNCK_SIZE))
#         grid_chuncks = np.array_split(gpoints.astype(T), n_chuncks)
#         #calculate B field for each chunck
#         B_idx = 0 # idx to update B at each chunck
#         iterator = tqdm(grid_chuncks, desc=f'{iw+1}/{len(wIs)}', ncols=80, leave=False) if len(grid_chuncks) > 1 else grid_chuncks
#         for gc in iterator:
#             nc = len(gc) # n=gpoints points
#             wp1, wp2 = w.astype(T), np.roll(w, -1, axis=0).astype(T)  # wire points (m,3)
#             dl = wp2 - wp1  # dl (m,3)
#             wm = (wp1 + wp2) / 2  # wire middle (m,3)
#             r = np.zeros((nc,m,3), dtype=T) # r (n,m,3)
#             ps = np.repeat(gc.reshape(nc,1,3), m, axis=1) # gpoints points (n,1,3) -> (n,m,3)
#             r = ps - wm # r (n,m,3)
#             rnorm = np.linalg.norm(r, axis=-1).reshape(nc,m,1) # |r| (n,m,1)
#             r̂ = r / rnorm # unit vector r (n,m,3)
#             assert wp1.dtype == T and wp2.dtype == T and dl.dtype == T and wm.dtype == T and r.dtype == T and rnorm.dtype == T and r̂.dtype == T, 'dtype error'
#             #calculate B field with Biot-Savart law
#             Bc = np.sum(μ0*I*np.cross(dl,r̂)/(4*np.pi*rnorm**2), axis=1)
#             #add B field to total B field
#             B[B_idx:B_idx+nc] += Bc
#             B_idx += nc #update B_idx
#     return B 

def calc_mag_field(wpaths:list[ndarray], wIs:list[ndarray], gpoints:ndarray): #calculate magnetic field, wpaths: wire paths, wIs: wire currents, gpoints: grid points
    # this is slower than calc, but uses less ram
    # calculate B field on a gpoints
    B = np.zeros_like(gpoints) #initialize B field (n,3)
    μ0 = 4*np.pi*1e-7 #vacuum permeability
    for iw, (w, I) in enumerate(zip(wpaths, wIs)): #for each wire
        wp1, wp2 = w, np.roll(w, -1, axis=0)  # wire points (m,3)
        dl = wp2 - wp1  # dl (m,3)
        wm = (wp1 + wp2) / 2  # wire middle (m,3)
        for i, p in enumerate(tqdm(gpoints, desc=f'{iw+1}/{len(wIs)}', leave=False)): # n times
            r = p - wm  # r (m,3)
            rnorm = np.linalg.norm(r, axis=1).reshape(-1,1)  # |r| (m,1)
            r̂ = r / rnorm  # unit vector r (m,3)
            B[i] += μ0 * I * np.sum(np.cross(dl, r̂) / rnorm**2, axis=0) / 4*np.pi # Biot-Savart law
    return B
