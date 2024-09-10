# functions for julia
using LinearAlgebra

function path_length(x::Array)
    diff = x - circshift(x, 1)
    norms = [norm(diff[i,:]) for i in axes(diff, 1)]
    return sum(norms)
end

function upresample(x::Array, s)
    diff = x - circshift(x, 1)
    norms = [norm(diff[i,:]) for i in axes(diff, 1)]
    L = sum(norms)
    n_segments = Int(ceil(L/s))
    s_new = L/n_segments # update s
    xnew = zeros(Float32, n_segments+1, size(x, 2))
    # interpolate
    for i in 1:n_segments
        t = (i-1)*s_new
        idx = findfirst(x -> x > t, cumsum(norms))
        if idx == 1
            xnew[i,:] = x[1,:]
        else
            t0 = cumsum(norms)[idx-1]
            t1 = cumsum(norms)[idx]
            alpha = (t - t0)/(t1 - t0)
            xnew[i,:] = x[idx-1,:] + alpha*(x[idx,:] - x[idx-1,:])
        end
    end
    #add the first point back to the end
    xnew[end,:] = x[1,:]

    # # 3d plot x and xnew
    # plot()
    # plot!(x[:,1], x[:,2], x[:,3], label="x", line=(:red, 2))
    # scatter!(x[:,1], x[:,2], x[:,3], label="x", markercolor=:red)
    # # plot!(xnew[:,1], xnew[:,2], xnew[:,3], label="xnew", line=(:blue, 2))
    # scatter!(xnew[:,1], xnew[:,2], xnew[:,3], label="xnew", markercolor=:blue)
    # display(plot!())
    return xnew
end

# x = [2.0 0.0 -1.0; -0.9999999999999996 1.7320508075688774 -1.0; -1.0000000000000009 -1.732050807568877 -1.0; 2.0 -4.898587196589413e-16 -1.0]
# upresample(x, 0.001);


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