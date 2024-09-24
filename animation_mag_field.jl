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
const NP = 400 # number of particles
const NT = 3000 # number of time steps
const STEP = 0.1 # step size
const TAIL_LENGTH = 30

function create_animation_vectors()
    pos = [V3(rand(Uniform(-GL,GL), 3)) for _ in 1:NP] # current positions
    all_pos = fill(fill(V3(0.0, 0.0, 0.0), NP), NT) # all positions
    all_mfs = fill(fill(V3(0.0, 0.0, 0.0), NP), NT) # all magnetic fields
    @showprogress for t in 1:NT
        mf = normalize.(calc_mag_field(wires, currs, pos)) # magnetic field at current positions
        all_pos[t] = pos # save positions
        all_mfs[t] = mf # save magnetic fields
        pos += STEP .* mf # update positions
        #remove posiitions outside the grid and replace them with random positions
        pos = [any(p .< -GL) || any(p .> GL) ? V3(rand(Uniform(-GL,GL), 3)) : p for p in pos]
    end
    return all_pos, all_mfs
end

all_pos, all_mfs = create_animation_vectors()

# animation
using Colors
using DataStructures: CircularBuffer

fig = Figure(size=(1000,1000), theme=theme_black())
title_mf = Observable("Magnetic Field 1/$(NT)") # title
cam_angle = Observable(5π/4) # camera angle
ax = Axis3(fig[1,1], aspect = :equal, xlabel="x", ylabel="y", zlabel="z", title=title_mf, azimuth=cam_angle)
limits!(ax, -GL, GL, -GL, GL, -GL, GL)
#fading colors
colors = distinguishable_colors(NP)
αs = range(0, 1.0, length=TAIL_LENGTH) |> collect
fading_colors = [[RGBA(c.r, c.g, c.b, α) for α in αs] for c in colors]
obs_colors = [Observable(c) for c in fading_colors]

for w in wires lines!(Point3f.(w), color=RGB(1-0.2*rand(), 1-0.2*rand(), 1-0.2*rand()), linewidth=3, transparency=true) end

# tails for the particles
tails = [CircularBuffer{Point3f}(TAIL_LENGTH) for _ in 1:NP]
tails = [Observable(t) for t in tails]
for (i, tail) in enumerate(tails) fill!(tail[], Point3f(all_pos[1][i])) end   # initialize the tails
for (i, tail) in enumerate(tails) lines!(ax, tail, linewidth=1, color=obs_colors[i]) end # plot the tails

display(fig)

# Function to update the plot
function update_plot(frame)
    ps = Point3f.(all_pos[frame])
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
        for j in 1:TAIL_LENGTH if zaps[j] > 1.2 tmp_α[j:min(j+2, TAIL_LENGTH)] .= 0.0 end end
        obs_colors[i][] = [RGBA(colors[i].r, colors[i].g, colors[i].b, α) for α in tmp_α]
    end
    #update camera angle
    cam_angle[] = 0.5 * sin(2 * 2π * frame / NT ) + 5π/4
end
for i in 1:NT # create live animation
    update_plot(i%NT+1)
    sleep(0.001)
end

# record(fig, "magnetic_field.mp4", 1:NT, framerate=60) do i 
#     update_plot(i%NT+1) 
# end # save mp4