include("utils.jl")
clc()
include("mag_field.jl")
using LinearAlgebra
using Distributions
using GeometryBasics
using Printf
using GLMakie
Base.show(io::IO, f::Float64) = @printf(io, "%.2f", f) # show only x decimals

const GL = 6.0 # grid limits

# create wires
w1, I1 = create_wire([V3(2*cos(t), 2*sin(t), -1) for t in range(0, stop=2π, length=4)], 50.0, 1.77e-8, 1e-4, 0.01)
w2, I2 = create_wire([V3(2*cos(t), 2*sin(t), 1) for t in range(0, stop=2π, length=6)], 40.0, 1.77e-8, 1e-4, 0.01)
w3, I3 = create_wire([V3(1.3*sin(t), 3, 1.3*cos(t)) for t in range(0, stop=2π, length=5)], -40.0, 1.77e-8, 1e-4, 0.01)
w4, I4 = create_wire([V3(3, 1.5*sin(t), 1.5*cos(t)) for t in range(0, stop=2π, length=9)], -40.0, 1.77e-8, 1e-4, 0.01)
wires, currs = [w1, w2, w3, w4], [I1, I2, I3, I4]

#create the animation first
const NP = 100 # number of particles
const NT = 500 # number of time steps
const STEP = 0.1 # step size
const TL = 30 # tail length

function create_animation_vectors()
    pos = [V3(rand(Uniform(-GL,GL), 3)) for _ in 1:NP] # current positions
    all_pos = fill(fill(V3(0.0, 0.0, 0.0), NP), NT) # all positions
    all_mfs = fill(fill(V3(0.0, 0.0, 0.0), NP), NT) # all magnetic fields
    all_norms = fill(fill(0.0, NP), NT) # all magnetic field norms
    @showprogress for t in 1:NT
        mf = calc_mag_field(wires, currs, pos) # magnetic field at current positions
        all_pos[t] = pos # save positions
        all_mfs[t] = mf # save magnetic fields
        nmf = norm.(mf) # magnetic field norms
        all_norms[t] = nmf # save magnetic field norms
        pos += STEP .* mf ./ nmf # update positions
        #remove posiitions outside the grid and replace them with random positions
        pos = [any(p .< -GL) || any(p .> GL) ? V3(rand(Uniform(-GL,GL), 3)) : p for p in pos]
    end
    return all_pos, all_mfs, all_norms
end

all_pos, all_mfs, all_norms = create_animation_vectors()
max_mf = mean(maximum.([all_norms[t] for t in 1:NT]))
min_mf = mean(minimum.([all_norms[t] for t in 1:NT]))
println("max_mf: $(max_mf*1000), min_mf: $(min_mf*1000)")

all_norms = [0. .+ (nt .- min_mf) ./ ((max_mf - min_mf)) for nt in all_norms] # map to [0,1]
all_cmap = [cgrad.(:inferno, scale=:exp)[all_norms[i]] for i in 1:NT] # color map

# animation
using Colors
using DataStructures: CircularBuffer

fig = Figure(size=(1000,1000), theme=theme_black())
title_mf = Observable("Magnetic Field 1/$(NT)") # title
ax = Axis3(fig[1,1], aspect = :equal, xlabel="x", ylabel="y", zlabel="z", title=title_mf)
limits!(ax, -GL, GL, -GL, GL, -GL, GL)
#fading colors
colors = distinguishable_colors(NP)
αs = range(0, 1.0, length=TL) |> collect
fading_colors = [[RGBA(c.r, c.g, c.b, α) for α in αs] for c in colors]
obs_colors = [Observable(c) for c in fading_colors]

# tails for the particles
tails = [CircularBuffer{Point3f}(TL) for _ in 1:NP] # tails
tailsc = [CircularBuffer{RGBA}(TL) for _ in 1:NP] # tail colors
tails = [Observable(t) for t in tails]
for (i, tail) in enumerate(tails) fill!(tail[], Point3f(all_pos[1][i])) end   # initialize the tails
for (i, tn) in enumerate(tailsc) fill!(tn, all_cmap[1][i]) end # initialize the tail norms

# plot 
for w in wires lines!(Point3f.(w), color=RGB(1-0.2*rand(), 1-0.2*rand(), 1-0.2*rand()), linewidth=3, transparency=true) end # plot the wires
for (i, tail) in enumerate(tails) lines!(ax, tail, linewidth=2, color=obs_colors[i]) end # plot the tails
display(fig)

# Function to update the plot
function update_plot(frame)
    ps = Point3f.(all_pos[frame]) # current positions
    # println(norms)
    for i in 1:NP push!(tailsc[i], all_cmap[frame][i]) end

    title_mf[] = "Magnetic Field $(frame)/$(NT)"
    #update tails
    for (i, tail) in enumerate(tails)
        push!(tail[], ps[i])
        tail[] = tail[]
    end
    #filter out new particles "zapping" across the 3d space
    for (i, tail) in enumerate(tails) 
        zaps = norm.(diff(tail[]))/STEP; push!(zaps, 0.0) # calculate the zaps
        tmp_α = copy(αs);
        for j in 1:TL if zaps[j] > 1.2 tmp_α[j:min(j+2, TL)] .= 0.0 end end
        k = 0.5 # mixing color factor
        obs_colors[i][] = [RGBA(colors[i].r*k+tailsc[i][j].r*(1-k), colors[i].g*k+tailsc[i][j].g*(1-k), colors[i].b*k+tailsc[i][j].b*(1-k), tmp_α[j]) for j in 1:TL] 
    end
    #update camera angle
    ax.azimuth[] = 0.5 * sin(2 * 2π * frame / NT ) + 5π/4
end

for i in 1:NT # create live animation
    update_plot(i%NT+1)
    sleep(0.001)
end

# record(fig, "magnetic_field.mp4", 1:NT, framerate=60) do i update_plot(i%NT+1) end # save mp4