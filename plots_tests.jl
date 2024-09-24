using LinearAlgebra
include("utils.jl")
clc()
include("mag_field.jl")
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%.2f", f) # show only x decimals
using Distributions
using GeometryBasics

const NG = 10 # nmber of grid points    
const GL = 5.0 # grid limits

# grid points
grpts = [V3(i, j, k) for i in range(-GL,GL,NG) for j in range(-GL,GL,NG) for k in range(-GL,GL,NG)]

# create wires
w1, I1 = create_wire([V3(2*cos(t), 2*sin(t), -1) for t in range(0, stop=2π, length=4)], 50.0, 1.77e-8, 1e-4, 0.01)
w2, I2 = create_wire([V3(2*cos(t), 2*sin(t), 1) for t in range(0, stop=2π, length=6)], 40.0, 1.77e-8, 1e-4, 0.01)
w3, I3 = create_wire([V3(-2.3, 1.3*sin(t), 1.6*cos(t)) for t in range(0, stop=2π, length=5)], 40.0, 1.77e-8, 1e-4, 0.01)
w4, I4 = create_wire([V3(1.8*sin(t), 2.4, 1.5*cos(t)) for t in range(0, stop=2π, length=9)], 40.0, 1.77e-8, 1e-4, 0.01)

wires, currs = [w1, w2, w3, w4], [I1, I2, I3, I4]

#calculate the magnetic field
B = calc_mag_field(wires, currs, grpts)


using GLMakie
# create an animation of the magnetic field

#create the animation first
const NP = 5 # number of particles
const NT = 300 # number of time steps
const STEP = 0.1 # step size

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
# println("all_pos: $all_pos, \n\nall_mfs: $all_mfs")
# println("all_pos: $(typeof(all_pos)), all_mfs: $(typeof(all_mfs))")


# animation
using Colors
using DataStructures: CircularBuffer
const TAIL_LENGTH = 150

# # fig = Figure(size=(800,800), theme=theme_dark())
# fig = Figure(size=(800,800))
# ax = Axis3(fig[1,1], aspect = :equal)
# ps, ms = Point3f.(all_pos[1]), Vec3d.(all_mfs[1])
# ls = norm.(ms) # lengths
# ms = 0.2 .* ms ./ ls # normalize

# colors = distinguishable_colors(NP, colorant"blue")

# arrows!(ax, ps, ms, color=colors, arrowsize=Vec3f(0.1, 0.1, 0.1))
# fig

# Initialize the points and magnetic field vectors
SEG_LEN = 0.3
ps = Point3f.(all_pos[1])
ms = Vec3f.(all_mfs[1])
ls = norm.(ms) # lengths
ms = SEG_LEN .* ms ./ ls # normalize

# create the animation
fig = Figure(size=(800,800))
title_mf = Observable("Magnetic Field 1/$(NT)") # title
ax = Axis3(fig[1,1], aspect = :equal, xlabel="x", ylabel="y", zlabel="z", title=title_mf)
limits!(ax, -GL, GL, -GL, GL, -GL, GL)
#fading colors
colors = distinguishable_colors(NP)
αs = range(0, 1.0, length=TAIL_LENGTH) |> collect
fading_colors = [[RGBA(c.r, c.g, c.b, α) for α in αs] for c in colors]
obs_colors = [Observable(c) for c in fading_colors]

p1s, p2s = copy(ps), ps + ms
#create a single vector taking an element from p1s and p2s one after the other [p1s[1], p2s[1], p1s[2], p2s[2], ...]
segments = Iterators.flatmap((p1, p2) -> [p1, p2], p1s, p2s) |> collect
segments = Observable(segments) # make it observable
println("p1s: $(typeof(p1s)), p2s: $(typeof(p2s))")
println("segments: $(typeof(segments))")
# linesegments!(ax, segments, color=colors1, transparency=true)
for w in wires lines!(Point3f.(w), color=:black, linewidth=3) end # plot the wires

# tails for the particles
tails = [CircularBuffer{Point3f}(TAIL_LENGTH) for _ in 1:NP]
tails = [Observable(t) for t in tails]
for (i, tail) in enumerate(tails)
    tail_pts = Point3f.([all_pos[a][i] for a in 1:TAIL_LENGTH])
    for p in tail_pts push!(tail[], p) end
end   
println("tails: $(typeof(tails))")
for (i, tail) in enumerate(tails) 
    # put transparency very low to the "jumps" in the tail so they are not visible
    jumps = norm.(diff(tail[]))/STEP; pushfirst!(jumps, 0.0) # calculate the jumps
    tmp_α = copy(αs);
    # tmp_α[jumps .> 1.2] .= 0.0 # set the elements of αs to 0.0 where jumps > 1.2
    for j in 1:TAIL_LENGTH
        if jumps[j] > 1.01
            tmp_α[j] = 0.0 
            if j > 1 tmp_α[j-1] = 0.0 
            elseif j < TAIL_LENGTH tmp_α[j+1] = 0.0 end
        end
    end
    obs_colors[i][] = [RGBA(colors[i].r, colors[i].g, colors[i].b, α) for α in tmp_α]
    lines!(ax, tail, linewidth=2, color=obs_colors[i])
end

display(fig)

# Function to update the arrows
function update_arrows!(frame)
    ps = Point3f.(all_pos[frame])
    ms = Vec3f.(all_mfs[frame])
    ls = norm.(ms) # lengths
    ms = SEG_LEN .* ms ./ ls # normalize
    p1s, p2s = copy(ps), ps + ms
    segments[] = Iterators.flatmap((p1, p2) -> [p1, p2], p1s, p2s) |> collect
    title_mf[] = "Magnetic Field $(frame)/$(NT)"

    #update tails
    for (i, tail) in enumerate(tails)
        push!(tail[], ps[i])
        tail[] = tail[]
    end
end

for i in 1:2*NT
    update_arrows!(i%NT+1)
    sleep(0.02)
end



