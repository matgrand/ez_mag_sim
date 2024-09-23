using LinearAlgebra
using ProgressMeter
include("utils.jl")

function create_wire(wp, V, ρ, section, seg_len)
    L = path_length(wp) # length L = path_length(wp) 
    R = ρ * L / section # resistance R = ρ * L / A
    I = V / R # current I = V / R
    wp = upresample(wp, seg_len) # resample the wire path
    return wp, I
end

function calc_mag_field(wpaths::Vector{Vector{V3{T}}}, wIs::Vector{T}, grpts::Vector{V3{T}}) where T
    B = fill(V3(), length(grpts)) # B field
    for (w, I) in zip(wpaths, wIs) calc_mag_field!(w, I, grpts, B) end
    return B
end

function calc_mag_field!(w::Vector{V3{T}}, I::T, grpts::Vector{V3{T}}, B::Vector{V3{T}}) where T
    μ0 = 4π * 1e-7 # permeability of vacuum
    dl = [w[(i%length(w))+1]-w[i] for i∈1:length(w)] # wire segment vectors
    wm = [(w[(i%length(w))+1]+w[i])/2 for i∈1:length(w)] # wire segment midpoints
    @showprogress "Magnetic Field:" for (ig, p) in enumerate(grpts) # loop over grid points
        r = p - wm # vector from wire segment midpoint to grid point (m,3)
        B[ig] += μ0 * I * sum(@.(dl × r / norm(r)^3)) / 4π # Biot-Savart law
    end
end
