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
    for i in 1:size(wpaths,1) # loop over wires
        w = wpaths[i]
        I = wIs[i]
        wa, wb = w, circshift(w, -1) # wire points
        dl = wb - wa # wire segment (m,3)
        wm = (wa + wb) / 2  # wire segment midpoint 
        for (ig, p) in enumerate(gpoints) # loop over grid points
            r = (p,) .- wm # vector from wire segment midpoint to grid point (m,3)
            rnorm = norm.(r) # distance from wire segment midpoint to grid point (m)
            r̂ = r ./ rnorm # unit vector from wire segment midpoint to grid point (m,3)
            B[ig] += μ0 * I * sum(cross.(dl, r̂) ./ rnorm.^2) / 4π # Biot-Savart law
        end
    end
    return B
end

# function calc_mag_field(wpaths, wIs, gpoints) 
#     # println("wpaths: $(typeof(wpaths)), $(size(wpaths)), $(size(wpaths[1])), $(size(wpaths[1][1]))")
#     # println("wIs: $(typeof(wIs)), $(size(wIs))")
#     # println("gpoints: $(typeof(gpoints)), $(size(gpoints)), $(size(gpoints[1]))")
#     B = [[0.0, 0.0, 0.0] for _ in 1:size(gpoints,1)] # B field
#     μ0 = 4π * 1e-7
#     for (w, I) in zip(wpaths, wIs)
#         wa, wb = w, circshift(w, -1) # wire points 
#         dl = wb - wa # wire segment (m,3)
#         wm = (wa + wb) / 2  # wire segment midpoint 
#         @showprogress for (ig, p) in enumerate(gpoints)
#             # println("p: $(p), wm: $(wm)")
#             # println("p size: $(size(p)), wm size: $(size(wm)) $(size(wm[1]))")
#             r = (p,) .- wm # vector from wire segment midpoint to grid point (m,3)
#             # println("r: $(r)")
#             rnorm = norm.(r) # distance from wire segment midpoint to grid point (m)
#             # println("rnorm: $(rnorm)")
#             r̂ = r ./ rnorm # unit vector from wire segment midpoint to grid point (m,3)
#             # cr = cross.(dl, r̂) # cross product of dl and r̂ (m,3)
#             # println("cr: $(cr)")
#             # println("sum: $(sum(μ0 * cross.(dl, r̂) * I ./ (4π * rnorm.^2)))")
#             B[ig] += μ0 * I * sum(cross.(dl, r̂) ./ rnorm.^2) / 4π # Biot-Savart law
#         end
#     end
#     return B
# end


# function calc_mf(ws::Vector{FemWire}, fem_grid::Vector{Vector{myT}}) 
#     # calculate B field on a fem_grid
#     @assert size(fem_grid, 2) == 3 "fem_grid must be a (n,3) array, not $(size(fem_grid))"
#     B = zeros(size(fem_grid,1), 3) # B field
#     # calculate B field, n=fem_grid points, m=wire points
#     μ0 = 4 * π * 1e-7 # vacuum permeability
#     for w in wires # loop over wires 
#         wp1, wp2 = w.wp, circshift(w.wp, -1) # wire points (m,3)
#         @assert size(wp1, 1) == size(wp2, 1) "wp1 and wp2 must have the same number of points"
#         @assert size(wp1, 2) == 3 "wp1 must be a (m,3) array, not $(size(wp1))"
#         for i in axes(fem_grid, 1) # loop over fem_grid points
#             for j in axes(w.wp, 1) # loop over wire points
#                 dl = wp2 - wp1 # dl 
#                 wm = (wp1 + wp2) / 2 # wire midpoint
#                 r = wm - fem_grid[i,:] # r 
#                 rnorm = norm(r) # r norm 
#                 r̂ = r / rnorm # unit vector r 
#                 B[i,:] += μ0 * w.I * cross(dl, r̂) / (4 * π * rnorm^2) # Biot-Savart law
#             end
#         end
# 	# map(x ->  f(x[1], x[2]), Iterators.product(axes1, axes2))
#     end
#     return B
# end;