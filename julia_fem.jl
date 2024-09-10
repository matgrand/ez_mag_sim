using LinearAlgebra
using PyPlot
using ProgressMeter

const T = Float32 # Float32 or Float64

struct FemWire
    wp::Array{T, 3} # wire path (m,3)
    V::T # voltage
    ρ::T # resistivity
    section::T # section area
    seg_len::T # segment length
    R::T # resistance
    L::T # length
    I::T # current
    _in_wp::Array{T, 3} # input wire path
    
    function FemWire(wp::Array{T, 3}, V::T=0.0, ρ::T=1.77e-8, section::T=1e-4, seg_len::T=5e-2)
        fw = new{T}()
        fw.wp = wp
        fw.V = V
        fw.ρ = ρ
        fw.section = section
        fw.seg_len = seg_len
        fw.R = zero(T)
        fw.L = zero(T)
        fw.I = zero(T)
        fw._in_wp = copy(fw.wp)
        
        # calculate length of wire
        diff = fw.wp - circshift(fw.wp, (1, 0)) # difference between points
        fw.L = sum(sqrt.(sum(diff.^2, dims=2)))
        
        # resample wire path with segments of length similar to seg_len
        w = []
        for i in 1:size(fw.wp, 1)
            p1, p2 = fw.wp[i, :], fw.wp[mod1(i+1, size(fw.wp, 1)), :]
            l = norm(p2 - p1)
            n = Int(l / fw.seg_len)
            for ii in 0:n-1
                push!(w, p1 + (p2 - p1) * ii / n)
            end
        end
        fw.wp = convert(Array{T}, w)
        
        fw.R = fw.ρ * fw.L / fw.section # resistance R = ρ * L / A
        fw.I = fw.V / fw.R # current I = V / R
        
        return fw
    end
end

function Base.show(io::IO, w::FemWire)
    println(io, "Wire: V=$(w.V) V, ρ=$(w.ρ) Ωm, s=$(w.section) m^2, L=$(w.L) m, R=$(w.R) Ω, I=$(w.I) A")
end

function plot(w::FemWire, ax::PyPlot.Axes; kwargs...)
    ax.plot(w._in_wp[:, 1], w._in_wp[:, 2], w._in_wp[:, 3]; kwargs...)
    return ax
end

struct FemMagField
    wires::Vector{FemWire}
    B::Array{T, 3}
    normB::Array{T, 1}
    
    function FemMagField(wires::Vector{FemWire})
        mg = new{T}()
        mg.wires = wires
        mg.B = zeros(T, size(wires[1].wp, 1), 3)
        mg.normB = zeros(T, size(wires[1].wp, 1))
        return mg
    end
end

function calc(mf::FemMagField, grid::Array{T, 3}) where T
    # calculate B field on a grid
    @assert size(grid, 2) == 3 "grid must be a (n,3) array, not $(size(grid))"
    mf.B .= zero(T) # initialize B field
    
    # calculate B field, n=grid points, m=wire points
    μ0 = 4 * π * 1e-7 # vacuum permeability
    for (wi, w) in enumerate(mf.wires) # for each wire
        wp1, wp2 = w.wp, circshift(w.wp, (-1, 0)) # wire points (m,3)
        dl = wp2 - wp1 # dl (m,3)
        wm = (wp1 + wp2) / 2 # wire middle (m,3)
        @progress for i in 1:size(grid, 1) # n times
            r = grid[i, :] - wm # r (m,3)
            rnorm = vecnorm(r, dims=2) # |r| (m,1)
            r̂ = r ./ rnorm # unit vector r (m,3)
            mf.B[i, :] .+= sum(μ0 * w.I .* cross(dl, r̂) ./ (4 * π * rnorm.^2), dims=1) # Biot-Savart law
        end
    end
    
    mf.normB = vecnorm(mf.B, dims=2)
    return mf.B
end