# functions for julia
using LinearAlgebra
using StaticArrays
# using Plots

#############################################################################################
###### 3D vector stuff ######################################################################
#############################################################################################
#struct for 3D vectors
struct V3{T<:Number} <: FieldVector{3,T}
    x::T
    y::T
    z::T
end

# empty constructor
V3() = V3(0.0, 0.0, 0.0)

#redefine the print function for V3
Base.show(io::IO, v::V3) = print(io,"[$(v.x),$(v.y),$(v.z)]")

# define + - for Vector{V3} and V3
Base.:-(a::Vector{V3{T}}, b::V3{T}) where {T<:Number} = [a - b for a in a]
Base.:-(a::V3{T}, b::Vector{V3{T}}) where {T<:Number} = [a - b for b in b]
Base.:+(a::Vector{V3{T}}, b::V3{T}) where {T<:Number} = [a + b for a in a]
Base.:+(a::V3{T}, b::Vector{V3{T}}) where {T<:Number} = [a + b for b in b]

#############################################################################################
## Random stuff #############################################################################
#############################################################################################
const MACSCREEN = (1728, 1050) # mac screen size for plotting

#clear the console
clc() = print("\033[2J\033[H") # clear the console
# cl() = clc(); closeall() # clear the console and close all plots (if using Plots)

#############################################################################################
## Misc #####################################################################################
#############################################################################################

function path_length(x::Array)
    diff = x - circshift(x, 1)
    norms = [norm(diff[i,:]) for i in axes(diff, 1)]
    return sum(norms)
end

function upresample(x, s)
    diff = x - circshift(x, 1)
    norms = [norm(diff[i,:]) for i in axes(diff, 1)]
    L = sum(norms)
    n_segments = Int(ceil(L/s))
    s_new = L/n_segments # update s
    xnew = [V3(0.0, 0.0, 0.0) for _ in 1:n_segments]          
    # interpolate
    for i in 1:n_segments
        t = (i-1)*s_new
        idx = findfirst(x -> x > t, cumsum(norms))
        if idx == 1
            xnew[i] = x[1]
        else
            t0 = cumsum(norms)[idx-1]
            t1 = cumsum(norms)[idx]
            alpha = (t - t0)/(t1 - t0)
            xnew[i] = x[idx-1] + alpha*(x[idx] - x[idx-1])
        end
    end
    #add the first point back to the end
    xnew[end,:] = x[1,:]
    return xnew
end

# 3D arrow plot
# as: arrow head size 0-1 (fraction of arrow length)
# la: arrow alpha transparency 0-1
function arrow3d!(x, y, z,  u, v, w; as=0.1, lc=:black, la=1, lw=0.4, scale=:identity)
    (as < 0) && (nv0 = -maximum(norm.(eachrow([u v w]))))
    for (x,y,z, u,v,w) in zip(x,y,z, u,v,w)
        nv = sqrt(u^2 + v^2 + w^2)
        v1, v2 = -[u,v,w]/nv, nullspace(adjoint([u,v,w]))[:,1]
        v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
        v5 = v4 - 2*(v4'*v2)*v2
        (as < 0) && (nv =   nv0) 
        v4, v5 = -as*nv*v4, -as*nv*v5
        plot!([x,x+u], [y,y+v], [z,z+w], lc=lc, la=la, lw=lw, scale=scale, label=false)
        plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], [z+w,z+w-v5[3]], lc=lc, la=la, lw=lw, label=false)
        plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], [z+w,z+w-v4[3]], lc=lc, la=la, lw=lw, label=false)
    end
end

;