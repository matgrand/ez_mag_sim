using LinearAlgebra
using ProgressMeter
using InteractiveUtils
using StaticArrays

struct V3{T} <: FieldVector{3,T} # 3D vector
    x::T
    y::T
    z::T
end

function calc_mag_field(wa, I, grpts) 
    B = [V3(0.,0.,0.) for _ in 1:size(grpts,1)] # B field
    μ0 = 4π * 1e-7
    wb = circshift(wa, -1) # wire points
    dl = wb - wa # wire segment (m,3)
    wm = (wa + wb) / 2  # wire segment midpoint 
    for (ig, p) in enumerate(grpts) # loop over grid points
        r = [p - wm[i] for i in 1:size(dl,1)] # vector from wire segment midpoint to grid point (m,3)
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
@time B = calc_mag_field(w, 100.0, grpts)
@time B = calc_mag_field(w, 100.0, grpts)
