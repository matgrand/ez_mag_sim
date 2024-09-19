using LinearAlgebra
using ProgressMeter
using InteractiveUtils
using StaticArrays

struct V3{T} <: FieldVector{3,T} 
    x::T
    y::T
    z::T
end

# define + - for Vector{V3} and V3
Base.:-(a::Vector{V3{T}}, b::V3{T}) where {T<:Number} = a .- Ref(b) # [ai - b for ai in a]
Base.:-(a::V3{T}, b::Vector{V3{T}}) where {T<:Number} = Ref(a) .- b
Base.:+(a::Vector{V3{T}}, b::V3{T}) where {T<:Number} = a .+ Ref(b)
Base.:+(a::V3{T}, b::Vector{V3{T}}) where {T<:Number} = Ref(a) .+ b

function calc_mag_field(w, I, grpts) 
    B = [V3(0.,0.,0.) for _ in 1:size(grpts,1)] # B field
    μ0 = 4π * 1e-7
    ws = circshift(w, -1) # wire points
    dl = ws - w # wire segment (m,3)
    wm = (w + ws) / 2  # wire segment midpoint 
    for (ig, p) in enumerate(grpts) # loop over grid points
        r = p - wm # vector from wire segment midpoint to grid point (m,3)
        rnorm = norm.(r) # distance from wire segment midpoint to grid point (m)
        r̂ = r ./ rnorm # unit vector from wire segment midpoint to grid point (m,3)
        B[ig] += μ0 * I * sum(cross.(dl, r̂) ./ rnorm.^2) / 4π # Biot-Savart law
    end
    return B
end

# create a wire
angles = range(0.0, stop=2π, length=1000)
w = [V3(cos(t), sin(t), 1.0) for t in angles]
println("wire: $(size(w))")

# create a grid
grpts = [V3(rand(), rand(), rand()) for _ in 1:10000]
println("grid: $(size(grpts))")

# B0 = @code_warntype calc_mag_field(w, 100.0, grpts) 

# calculate the magnetic field
B = calc_mag_field(w, 100.0, grpts)
@time B = calc_mag_field(w, 100.0, grpts)
