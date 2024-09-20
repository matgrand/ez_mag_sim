# utils 
using LinearAlgebra
using Plots
using ProgressMeter
using InteractiveUtils
# plotlyjs()
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%.4f", f) # show only 2 decimals

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
const N_GRID = 20
const GRID_RANGE = (-1.5,1.5)

grpts = create_grid(GRID_RANGE,GRID_RANGE,GRID_RANGE, (N_GRID, N_GRID, N_GRID)) # grid points
println("grid points: $(size(grpts,1))")

wp2 = create_horiz_circular_path(5, 2.0, 1.5)
wp1 = create_horiz_circular_path(3, 2.0, -1.0)

# # # plot wire paths
# plot() # create a plot
# plot!(wp1[:,1], wp1[:,2], wp1[:,3], label="wp1", line=(:red, 3))
# plot!(wp2[:,1], wp2[:,2], wp2[:,3], label="wp2", line=(:blue, 3))
# display(plot!()) # display the plot

w2, I2 = create_wire(wp2, 40.0, 1.77e-8, 1e-4, 0.01)
w1, I1 = create_wire(wp1, 50.0, 1.77e-8, 1e-4, 0.01)

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
for i in axes(normB, 1)[1:10]
    println("grid: $(grpts[i]) B: $(B[i]) normB: $(normB[i])")
end

# plot() # create a plot
# u, v, w = normB[:,1], normB[:,2], normB[:,3]
# # quiver!(grpts[:,1], grpts[:,2], grpts[:,3], quiver=(u, v, w),color=:blues, colorbar_title="B", normalize=true)
# arrow3d!(grpts[:,1], grpts[:,2], grpts[:,3], u, v, w, as=0.2, lc=:black, la=0.5, lw=2)
# # plot wire paths
# plot!(wp1[:,1], wp1[:,2], wp1[:,3], label="wp1", line=(:red, 3))
# plot!(wp2[:,1], wp2[:,2], wp2[:,3], label="wp2", line=(:blue, 3))
# display(plot!()) # display the plot


# # plotting matplotlib
# using PyPlot
# fig = figure(figsize=(10,10))
# ax = fig.add_subplot(111, projection="3d")
# ax.quiver(grpts[:,1], grpts[:,2], grpts[:,3], B[:,1], B[:,2], B[:,3], length=0.1, normalize=true)
# ax.plot(wp1[:,1], wp1[:,2], wp1[:,3], label="wp1", color="red")
# ax.plot(wp2[:,1], wp2[:,2], wp2[:,3], label="wp2", color="blue")
# legend()
# show()

