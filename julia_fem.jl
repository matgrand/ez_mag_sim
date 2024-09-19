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

function calc_mag_field(wpaths, wIs, grpts) 
    B = [V3(0.,0.,0.) for _ in 1:size(grpts,1)] # B field
    μ0 = 4π * 1e-7
    for (w, I) in zip(wpaths, wIs)
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
    end
    return B
end


