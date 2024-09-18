# utils 
using LinearAlgebra
using Plots
using ProgressMeter
# plotlyjs()
include("utils.jl")

function create_grid(xlim=(-1,1), ylim=(-1,1), zlim=(-1,1), n=(10,10,10))
    x = range(xlim[1], stop=xlim[2], length=n[1])
    y = range(ylim[1], stop=ylim[2], length=n[2])
    z = range(zlim[1], stop=zlim[2], length=n[3])
    grid = zeros(n[1]*n[2]*n[3], 3)
    for i in 1:n[1], j in 1:n[2], k in 1:n[3]
        grid[(i-1)*n[2]*n[3] + (j-1)*n[3] + k, 1] = x[i]
        grid[(i-1)*n[2]*n[3] + (j-1)*n[3] + k, 2] = y[j]
        grid[(i-1)*n[2]*n[3] + (j-1)*n[3] + k, 3] = z[k]
    end
    return grid
end

function create_horiz_circular_path(n=8, r=2.0, z=1.0)
    # create a wire path
    t = range(0, stop=2π, length=n+1)
    wp = [r * cos.(t), r * sin.(t), fill(z, n+1)]  # Create a wire path
    println("wp type: $(typeof(wp))")
    println("wp size: $(size(wp))")
    println("wp: $wp")
    return wp
end;

## 

# wire and magnetic field
const myT = Float64 # type of the wire

mutable struct FemWire
    wp::Vector{Vector{myT}} # wire path (m,3)
    V::myT
    ρ::myT
    section::myT
    seg_len::myT
    R::myT
    L::myT
    I::myT
    
    # constructor with calculated parameters R L I
    function FemWire(wp::Vector{Vector{myT}}, V::myT, ρ::myT, section::myT, seg_len::myT)
        # calculate length of the wire L, and resample the wire path with seg_len
        L = path_length(wp) # length L = path_length(wp) 
        R = ρ * L / section # resistance R = ρ * L / A
        I = V / R # current I = V / R
        # resample the wire path with seg_len
        wp = upresample(wp, seg_len) # resample the wire path
        return new(wp, V, ρ, section, seg_len, R, L, I)
    end
end

function calc_mf(ws::Vector{FemWire}, fem_grid::Vector{Vector{myT}}) 
    # calculate B field on a fem_grid
    @assert size(fem_grid, 2) == 3 "fem_grid must be a (n,3) array, not $(size(fem_grid))"
    B = zeros(size(fem_grid,1), 3) # B field
    # calculate B field, n=fem_grid points, m=wire points
    μ0 = 4 * π * 1e-7 # vacuum permeability
    for w in wires # loop over wires 
        wp1, wp2 = w.wp, circshift(w.wp, -1) # wire points (m,3)
        @assert size(wp1, 1) == size(wp2, 1) "wp1 and wp2 must have the same number of points"
        @assert size(wp1, 2) == 3 "wp1 must be a (m,3) array, not $(size(wp1))"
        for i in axes(fem_grid, 1) # loop over fem_grid points
            for j in axes(w.wp, 1) # loop over wire points
                dl = wp2 - wp1 # dl 
                wm = (wp1 + wp2) / 2 # wire midpoint
                r = wm - fem_grid[i,:] # r 
                rnorm = norm(r) # r norm 
                r̂ = r / rnorm # unit vector r 
                B[i,:] += μ0 * w.I * cross(dl, r̂) / (4 * π * rnorm^2) # Biot-Savart law
            end
        end
	# map(x ->  f(x[1], x[2]), Iterators.product(axes1, axes2))
    end
    return B
end;

##

# test all the functions
const N_GRID = 15
const GRID_RANGE = (-1.5,1.5)

gr = create_grid(GRID_RANGE,GRID_RANGE,GRID_RANGE, (N_GRID, N_GRID, N_GRID))

wp1 = create_horiz_circular_path(3, 2.0, -1.0)
wp2 = create_horiz_circular_path(5, 2.0, 1.5)

# # plot wire paths
plot() # create a plot
plot!(wp1[:,1], wp1[:,2], wp1[:,3], label="wp1", line=(:red, 3))
plot!(wp2[:,1], wp2[:,2], wp2[:,3], label="wp2", line=(:blue, 3))
display(plot!()) # display the plot

w1 = FemWire(wp1, 50.0, 1.77e-8, 1e-4, 0.01)
w2 = FemWire(wp2, 40.0, 1.77e-8, 1e-4, 0.01) 

wires = [w1, w2]
# wires = [w1]
println("wires created")

# plot wires
plot() # create a plot
plot!(w1.wp[:,1], w1.wp[:,2], w1.wp[:,3], label="w1", line=(:red, 3))
plot!(w2.wp[:,1], w2.wp[:,2], w2.wp[:,3], label="w2", line=(:blue, 3))
display(plot!()) # display the plot
println("wires plotted")
readline()


@time B = calc_mf(wires, gr)
println("B field calculated")

# calculate a vector of norms using list comprehension
normB = [norm(B[i,:]) for i in axes(B, 1)]
#print normB line for line
for i in axes(normB, 1)[1:10]
    println("grid: ($(round(gr[i,1], digits=3)),$(round(gr[i,2], digits=3)),$(round(gr[i,3], digits=3))) normB: $(round(normB[i], digits=5))")
end

# plot() # create a plot
# u, v, w = normB[:,1], normB[:,2], normB[:,3]
# # quiver!(gr[:,1], gr[:,2], gr[:,3], quiver=(u, v, w),color=:blues, colorbar_title="B", normalize=true)
# arrow3d!(gr[:,1], gr[:,2], gr[:,3], u, v, w, as=0.2, lc=:black, la=0.5, lw=2)
# # plot wire paths
# plot!(wp1[:,1], wp1[:,2], wp1[:,3], label="wp1", line=(:red, 3))
# plot!(wp2[:,1], wp2[:,2], wp2[:,3], label="wp2", line=(:blue, 3))
# display(plot!()) # display the plot


# # plotting matplotlib
# using PyPlot
# fig = figure(figsize=(10,10))
# ax = fig.add_subplot(111, projection="3d")
# ax.quiver(gr[:,1], gr[:,2], gr[:,3], B[:,1], B[:,2], B[:,3], length=0.1, normalize=true)
# ax.plot(wp1[:,1], wp1[:,2], wp1[:,3], label="wp1", color="red")
# ax.plot(wp2[:,1], wp2[:,2], wp2[:,3], label="wp2", color="blue")
# legend()
# show()

