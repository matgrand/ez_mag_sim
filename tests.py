from fem import FemWire, FemMagField
from utils import create_grid, create_horiz_circular_path
from time import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt, os, shutil
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

FIGSIZE = (10,10)
GRID_LIM = (-6,6)
ARROW_LEN = 0.3
#animation
NPART = 4000 #number of particles to plot, reduce for a faster animation
STEP_SIZE = 0.08 #step size for each iteration
N_ITER = 3000 #number of iterations to animate 3000
FPS = 30.0 #frames per second
ANIM_SPEED = 2.5 #speed of animation (can also be used to speed up slow animations)
SKIP_FRAMES = 1 #skip frames to reduce animation size
SAVE_MP4 = False  # use saved pics to create mp4 video, for big animations
AUTO_ROTATE_VIEW = SAVE_MP4 or False # rotate view in 3d plot automatically

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
    return fig

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
    return fig

def test_magfield_animation():

    #colors
    import colorsys
    def rainbow_c(idx, n=10): # return a rainbow color
        c_float = colorsys.hsv_to_rgb(idx/n, 1.0, 1.0)
        return tuple([int(round(255*x)) for x in c_float])
    
    rcol = [rainbow_c(i, NPART) for i in range(NPART)]

    fig = plt.figure(figsize=FIGSIZE)  # big figure just to makeit full screen
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=GRID_LIM, ylim=GRID_LIM, zlim=GRID_LIM, xlabel='x', ylabel='y', zlabel='z', aspect='equal')
    for w in wires: w.plot(ax, color='r') #plot wires

    curr_poss = np.random.uniform(GRID_LIM[0], GRID_LIM[1], (NPART,3)) #random positions
    all_poss = np.zeros((N_ITER, NPART, 3), dtype=np.float32)
    all_mfs = np.zeros((N_ITER, NPART, 3), dtype=np.float32)
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
        update_ax(ax, all_poss[ii], all_mfs[ii], linewidths=0.8, colors=rcol)
        for w in wires: w.plot(ax, color='r') #plot wires
        if AUTO_ROTATE_VIEW: ax.view_init(elev=30, azim=2*360*id/N_ITER) #rotate view
        plt.tight_layout()
        print(f'iteration {id}/{N_ITER}         ', end='\r')
        #save image
        if SAVE_MP4: plt.savefig(f'anim/{id:06d}.png', dpi=100) #dpi=300

    if SAVE_MP4: # create images
        if os.path.exists('anim'): shutil.rmtree('anim')#remove old images
        os.mkdir('anim') #create folder for images
        for i in tqdm(range(N_ITER), desc='anim', ncols=80, leave=False): update(i) #create images
        # create mp4 using ffmpeg
        os.system(f'ffmpeg -y -r {FPS} -i anim/%06d.png -c:v libx264 -vf fps={FPS} -pix_fmt yuv420p anim.mp4')
    else:
        # # matplotlib animation
        print('creating animation... this may take a while...')
        ani = animation.FuncAnimation(fig=fig, func=update, frames=N_ITER, interval=1000/FPS/ANIM_SPEED, blit=False, repeat=True)
    return None if SAVE_MP4 else ani


if __name__ == '__main__':
    # ix, iy, iz = 10,10,10 #number of points in each dimension
    # ix, iy, iz = 20,20,20 #number of points in each dimension
    # ix, iy, iz = 37,37,37 #number of points in each dimension
    ix, iy, iz = 53,53,53 #number of points in each dimension
    # ix, iy, iz = 80,80,80 #number of points in each dimension
    grid = create_grid(GRID_LIM, GRID_LIM, GRID_LIM, n=(ix,iy,iz)) #create a grid
    wp1 = create_horiz_circular_path(n=3, r=2.0, z=-1.0) #create a wire path
    wp2 = create_horiz_circular_path(n=5, r=2.0, z=1.5) #create a wire path
    # wp3 = create_horiz_circular_path(n=6, r=2.0, z=-1.5) #create a wire path
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

    f1 = test_magfield_plot()

    f2 = test_streamplot()

    f3 = test_magfield_animation()

    plt.show()






