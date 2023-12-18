from interfaces import Wire, MagField, create_example_path, create_grid
from symbolic import SymWire, SymMagField
from fem import FemWire, FemMagField
from time import time
from tqdm import tqdm

import numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

FIGSIZE = (12,12)
GRID_LIM = (-4,4)
ARROW_LEN = 0.3

#set numpy print options
np.set_printoptions(precision=2, suppress=True, linewidth=200)

def test_streamplot():
    #streamplot magnetic field
    fig = plt.figure(figsize=FIGSIZE)  # big figure just to makeit full screen
    g3, b3 = grid.reshape((ix,iy,iz,3)), mf.B.reshape((ix,iy,iz,3))
    # gp3x, gp3y, gp3z = g3[0,:,:,0], g3[0,:,:,1], g3[0,:,:,2]
    # bp3x, bp3y, bp3z = b3[0,:,:,0], b3[0,:,:,1], b3[0,:,:,2]
    gp3x, gp3y, gp3z = g3[:,:,10,0], g3[:,:,10,1], g3[:,:,10,2]
    bp3x, bp3y, bp3z = b3[:,:,10,0], b3[:,:,10,1], b3[:,:,10,2]
    # plt.streamplot(gp3y.T, gp3z.T, bp3y.T, bp3z.T)
    plt.streamplot(gp3x.T, gp3y.T, bp3x.T, bp3y.T, density=2.5)
    plt.axis('equal')
    plt.show()

def update_ax(ax:plt.Axes, points:np.ndarray, vecs:np.ndarray, linewidths=1.0, colors='k'):
    assert points.shape == vecs.shape, f'points and vecs must have the same shape, not {points.shape} and {vecs.shape}'
    assert points.shape[1] == 3, f'must be a (n,3) array, not {points.shape}'
    assert points.ndim == 2, f'must be a (n,3) array, not {points.shape}'
    p1s, p2s = points, points + vecs*ARROW_LEN
    s12 = np.hstack([p1s,p2s]).copy().reshape((-1,2,3)) #create segments
    lc = Line3DCollection(s12, linewidths=linewidths, colors=colors)
    ax.add_collection(lc) #add line collection to plot
    return ax

def test_magfield_plot():
    #plot magnetic field and wire
    fig = plt.figure(figsize=FIGSIZE)  # big figure just to makeit full screen
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=GRID_LIM, ylim=GRID_LIM, zlim=GRID_LIM, xlabel='x', ylabel='y', zlabel='z')
    dec = max(1, ix*iy*iz//4999)
    cmap = np.log(1000*np.clip(mf.normB[::dec], 0, 0.01))
    #plot magnetic field
    ax = update_ax(ax, grid[::dec], mf.B[::dec]/mf.normB[::dec,np.newaxis], linewidths=0.8, colors=plt.cm.inferno(cmap))
    for w in wires: w.plot(ax, color='r') #plot wires
    ax.scatter(grid[::dec,0], grid[::dec,1], grid[::dec,2], s=1, color='k') #plot grid points
    plt.tight_layout()
    plt.show()

def test_magfield_animation():
    NIDXS = 3000 #number of idxs to plot
    STEP_SIZE = 0.12 #step size for each iteration
    N_ITER = 2500 #number of iterations
    FPS = 30.0 #frames per second
    SKIP_FRAMES = 1 #skip frames to reduce animation size

    #colors
    import colorsys
    def rainbow_c(idx, n=10): # return a rainbow color
        c_float = colorsys.hsv_to_rgb(idx/n, 1.0, 1.0)
        return tuple([int(round(255*x)) for x in c_float])
    
    rcol = [rainbow_c(i, NIDXS) for i in range(NIDXS)]

    fig = plt.figure(figsize=FIGSIZE)  # big figure just to makeit full screen
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=GRID_LIM, ylim=GRID_LIM, zlim=GRID_LIM, xlabel='x', ylabel='y', zlabel='z', aspect='equal')
    plt.tight_layout()

    for w in wires: w.plot(ax, color='r') #plot wires

    curr_poss = np.random.uniform(GRID_LIM[0], GRID_LIM[1], (NIDXS,3)) #random positions
    all_poss = np.zeros((N_ITER, NIDXS, 3), dtype=np.float32)
    all_mfs = np.zeros((N_ITER, NIDXS, 3), dtype=np.float32)
    for i in tqdm(range(N_ITER), desc='prep anim', ncols=80, leave=False):
        curr_mf = mf.calc(curr_poss, normalized=True)
        all_poss[i], all_mfs[i] = curr_poss, curr_mf
        next_pos = curr_poss + STEP_SIZE*curr_mf
        #remove positions outside grid
        valid_mask = np.logical_and(np.min(next_pos, axis=-1) >= GRID_LIM[0], np.max(next_pos, axis=-1) <= GRID_LIM[1])
        curr_poss = next_pos #update curr_poss
        curr_poss[~valid_mask] = np.random.uniform(GRID_LIM[0], GRID_LIM[1], (np.sum(~valid_mask),3)) #random positions

    def update(id):
        ii = (id*SKIP_FRAMES) % N_ITER # trick to skip frames and get a faster animation
        ax.clear() # clear axis, comment to leave trails
        ax.set(xlim=GRID_LIM, ylim=GRID_LIM, zlim=GRID_LIM, xlabel='x', ylabel='y', zlabel='z', aspect='equal', title=f'iteration {id}/{N_ITER}')
        plt.tight_layout()
        update_ax(ax, all_poss[ii], all_mfs[ii], linewidths=0.8, colors=rcol)
        for w in wires: w.plot(ax, color='r') #plot wires
        ax.view_init(elev=30, azim=360*id/N_ITER) #rotate view, comment to be able to rotate view yourself
        return ax

    ani = animation.FuncAnimation(fig=fig, func=update, frames=N_ITER, interval=1000/FPS, blit=False, repeat=True)
    # ani.save('anim.gif', writer='imagemagick') # save animation as gif
    plt.show()

if __name__ == '__main__':
    # ix, iy, iz = 10,10,10 #number of points in each dimension
    # ix, iy, iz = 20,20,20 #number of points in each dimension
    ix, iy, iz = 37,37,37 #number of points in each dimension
    # ix, iy, iz = 50,50,50 #number of points in each dimension
    # ix, iy, iz = 80,80,80 #number of points in each dimension
    grid = create_grid(GRID_LIM, GRID_LIM, GRID_LIM, n=(ix,iy,iz)) #create a grid
    wp1 = create_example_path(n=3, r=2.0, z=-1.0) #create a wire path
    wp2 = create_example_path(n=5, r=2.0, z=1.5) #create a wire path
    # wp3 = create_example_path(n=6, r=2.0, z=-1.5) #create a wire path
    wp3 = np.array([[[1.3*np.sin(t)],[3],[1.3*np.cos(t)]] for t in np.linspace(0,2*np.pi,5+1)]).reshape((-1,3))
    wp4 = np.array([[[3],[1.5*np.sin(t)],[1.5*np.cos(t)]] for t in np.linspace(0,2*np.pi,7+1)]).reshape((-1,3))

    ## FEM
    w1 = FemWire(wp1, V=50,  seg_len=0.1) #create a wire
    w2 = FemWire(wp2, V=40,  seg_len=0.1) #create a wire
    w3 = FemWire(wp3, V=-40, seg_len=0.1) #create a wire
    w4 = FemWire(wp4, V=-40, seg_len=0.1) #create a wire
    # wires = [w1, w2, w3] 
    wires = [w1, w2, w3, w4]
    # wires = [w1] 

    mf = FemMagField(wires) #create a magnetic field
    calcs = time()
    mf.calc(grid) #calculate magnetic field
    calce = time()
    print(f'calc0: {calce-calcs:.3f} s')
    print(f'mean of mf norm: {np.mean(mf.normB):.5f} T')

    # TESTS

    # test_magfield_plot()

    # test_streamplot()

    test_magfield_animation()






