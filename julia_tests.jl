# utils 
using LinearAlgebra
using ProgressMeter
using InteractiveUtils
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%.7f", f) # show only 2 decimals

include("utils.jl")
include("julia_fem.jl")

function create_grid(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1), n=(10,10,10))
    x = range(xlim[1], stop=xlim[2], length=n[1])
    y = range(ylim[1], stop=ylim[2], length=n[2])
    z = range(zlim[1], stop=zlim[2], length=n[3])
    grid = [V3(0.0, 0.0, 0.0) for _ in 1:n[1]*n[2]*n[3]]
    for i in 1:n[1], j in 1:n[2], k in 1:n[3]
        grid[(i-1)*n[2]*n[3] + (j-1)*n[3] + k] = V3(x[i], y[j], z[k])
    end
    return grid
end

function create_horiz_circular_path(n=8, r=2.0, z=1.0)
    # create a wire path
    ts = range(0, stop=2π, length=n+1)
    wp = [V3(r*cos(t), r*sin(t), z) for t in ts]
    return wp
end;

##

# test all the functions
const N_GRID = 67
const GRID_LIM = (-3.5,3.5)

grpts = create_grid(GRID_LIM,GRID_LIM,GRID_LIM, (N_GRID, N_GRID, N_GRID)) # grid points
println("grid points: $(size(grpts,1))")

wp1 = create_horiz_circular_path(3, 2.0, -1.0)
wp2 = create_horiz_circular_path(5, 2.0, 1.0)

# # # plot wire paths
# plot() # create a plot
# plot!(wp1[:,1], wp1[:,2], wp1[:,3], label="wp1", line=(:red, 3))
# plot!(wp2[:,1], wp2[:,2], wp2[:,3], label="wp2", line=(:blue, 3))
# display(plot!()) # display the plot

w1, I1 = create_wire(wp1, 50.0, 1.77e-8, 1e-4, 0.01)
w2, I2 = create_wire(wp2, 40.0, 1.77e-8, 1e-4, 0.01)

wpaths = [w1, w2]
wIs = [I1, I2]

# # plot wires
# plot() # create a plot
# x1, y1, z1 = [w[1] for w in w1], [w[2] for w in w1], [w[3] for w in w1]
# x2, y2, z2 = [w[1] for w in w2], [w[2] for w in w2], [w[3] for w in w2] 
# plot!(x1, y1, z1, label="wp1", line=(:red, 3))
# plot!(x2, y2, z2, label="wp2", line=(:blue, 3))
# display(plot!()) # display the plot
# println("wires plotted")


# _ = @code_warntype calc_mag_field(wpaths, wIs, grpts)
B = calc_mag_field(wpaths, wIs, grpts)
@time B = calc_mag_field(wpaths, wIs, grpts)

# println("B field calculated")

# calculate a vector of norms using list comprehension
normB = [norm(B[i,:]) for i in axes(B, 1)]
#print normB line for line
for i in axes(normB, 1)[1:5]
    println("grid: $(grpts[i]) B: $(B[i]) normB: $(normB[i])")
end

x, y, z = [g[1] for g in grpts], [g[2] for g in grpts], [g[3] for g in grpts]
B̂ = 0.2 .* B ./ normB
u, v, w = [b[1] for b in B̂], [b[2] for b in B̂], [b[3] for b in B̂]
x1, y1, z1 = [w[1] for w in w1], [w[2] for w in w1], [w[3] for w in w1]
x2, y2, z2 = [w[1] for w in w2], [w[2] for w in w2], [w[3] for w in w2]


using Plots
# plotlyjs() # use the plotlyjs backend

function draw_segments!(x,)
    
end

#magnetic field plot
plot(size=(1000, 1000)) # create a full screen plot
quiver!(x, y, z, quiver=(u, v, w), color=:orange, colorbar_title="B", lw=2)
# arrow3d!(x, y, z, u, v, w, as=0.2, lc=:black, la=0.5, lw=2)
# plot wire paths
plot!(x1, y1, z1, label="wp1", line=(:red, 3))
plot!(x2, y2, z2, label="wp2", line=(:blue, 3))
display(plot!()) # display the plot

# 3d animation
#animation
const NPART = 50 #number of particles to plot, reduce for a faster animation
const STEP_SIZE = 0.08 #step size for each iteration
const N_ITER = 500 #number of iterations to animate 3000
const FPS = 30.0 #frames per second
const ANIM_SPEED = 2.5 #speed of animation (can also be used to speed up slow animations)
const SKIP_FRAMES = 1 #skip frames to reduce animation size
const SAVE_MP4 = false  # use saved pics to create mp4 video, for big animations

