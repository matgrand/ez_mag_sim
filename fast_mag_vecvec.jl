using LinearAlgebra
using ProgressMeter

function calc_mag_field(w, I, grpts) 
    B = [[0.0, 0.0, 0.0] for _ in 1:size(grpts,1)] # B field
    μ0 = 4π * 1e-7
    wa, wb = w, circshift(w, -1) # wire points
    dl = wb .- wa # wire segment (m,3)
    wm = (wa + wb) / 2  # wire segment midpoint 
    for (ig, p) in enumerate(grpts) # loop over grid points
        r = (p,) .- wm # vector from wire segment midpoint to grid point (m,3)
        rnorm = norm.(r) # distance from wire segment midpoint to grid point (m)
        r̂ = r ./ rnorm # unit vector from wire segment midpoint to grid point (m,3)
        B[ig] += μ0 * I * sum(cross.(dl, r̂) ./ rnorm.^2) / 4π # Biot-Savart law
    end
    return B
end

# create a wire
angles = range(0, stop=2π, length=1000)
w = [[cos(t), sin(t), 1.0] for t in angles]
println("wire: $(size(w)), $(size(w[1]))")

# create a grid
grpts = [rand(3) for _ in 1:10000]
println("grid: $(size(grpts)), $(size(grpts[1]))")

# calculate the magnetic field
@time B = calc_mag_field(w, 100.0, grpts)
