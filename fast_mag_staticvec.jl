using LinearAlgebra
using ProgressMeter
using InteractiveUtils
using StaticArrays
using BenchmarkTools
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%.4f", f) # show only 2 decimals


include("utils.jl")

function calc_mag_field(w, I, grpts) 
    B = fill(V3(0.,0.,0.), length(grpts)) # B field
    μ0 = 4π * 1e-7 # permeability of vacuum
    dl = [w[(i%length(w))+1]-w[i] for i∈1:length(w)] # wire segment vectors
    wm = [(w[(i%length(w))+1]+w[i])/2 for i∈1:length(w)] # wire segment midpoints
    for (ig, p) in enumerate(grpts) # loop over grid points
        r = p - wm # vector from wire segment midpoint to grid point (m,3)
        B[ig] += μ0 * I * sum(@.(dl × r / norm(r)^3)) / 4π # Biot-Savart law
    end
    return B
end


# create a wire
M = 1000
angles = range(0.0, stop=2π, length=M)
w = [V3(cos(t), sin(t), 1.0) for t in angles]
println("wire: $(size(w))")

# create a grid
N = 10000
# grpts = [V3(rand(), rand(), rand()) for _ in 1:10000]
grpts = [V3(i/N,i/N,i/N) for i in 1:N]
println("grid: $(size(grpts))")

# B0 = @code_warntype calc_mag_field(w, 100.0, grpts) 

# calculate the magnetic field
# B = @code_warntype calc_mag_field(w, 100.0, grpts[1:10])
B = calc_mag_field(w, 1_000_000.0, grpts[1:10])

@time B = calc_mag_field(w, 1_000_000.0, grpts)

#print the 10 first grid points and the corresponding B field
for i in 1:5 println("grid: $(grpts[i]) B: $(B[i])") end

