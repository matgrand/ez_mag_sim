using LinearAlgebra
using ProgressMeter

function create_wire(wp, V, ρ, section, seg_len)
    L = path_length(wp) # length L = path_length(wp) 
    R = ρ * L / section # resistance R = ρ * L / A
    I = V / R # current I = V / R
    wp = upresample(wp, seg_len) # resample the wire path
    return wp, I
end

function calc_mag_field(wpaths, wIs, gpoints) 
    B = [[0.0, 0.0, 0.0] for _ in 1:size(gpoints,1)] # B field
    μ0 = 4π * 1e-7
    for (w, I) in zip(wpaths, wIs)
        wa, wb = w, circshift(w, -1) # wire points
        dl = wb - wa # wire segment (m,3)
        wm = (wa + wb) / 2  # wire segment midpoint 
        for (ig, p) in enumerate(gpoints) # loop over grid points
            for iw in 1:size(dl,1) # loop over wire segments
                r = p - wm[iw] # vector from wire segment midpoint to grid point (3)
                rnorm = norm(r) # distance from wire segment midpoint to grid point
                r /= rnorm # unit vector from wire segment midpoint to grid point (3)
                B[ig] += μ0 * I * cross(dl[iw], r) / rnorm^2 / 4π # Biot-Savart law
            end
        end
    end
    return B
end

# function calc_mag_field(wpaths, wIs, gpoints) 
#     B = [[0.0, 0.0, 0.0] for _ in 1:size(gpoints,1)] # B field
#     μ0 = 4π * 1e-7
#     for (w, I) in zip(wpaths, wIs)
#         wa, wb = w, circshift(w, -1) # wire points
#         dl::Vector{Vector{Float64}} = wb .- wa # wire segment (m,3)
#         wm = (wa + wb) / 2  # wire segment midpoint 

#         cr = [[0.0, 0.0, 0.0] for _ in 1:size(dl,1)] # cross product

#         for (ig, p) in enumerate(gpoints) # loop over grid points
#             r = (p,) .- wm # vector from wire segment midpoint to grid point (m,3)
#             rnorm = norm.(r) # distance from wire segment midpoint to grid point (m)
#             normalize!.(r) # unit vector from wire segment midpoint to grid point (m,3)
#             cross!.(cr, dl, r)
#             B[ig] += μ0 * I * sum(cr ./ rnorm.^2) / 4π # Biot-Savart law
#         end
#     end
#     return B
# end

