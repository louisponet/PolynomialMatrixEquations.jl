using LinearAlgebra.LAPACK: gges!

function zero_cols(a)
    n = size(a,2)
    sums = zeros(n)
    Threads.@threads for i in axes(a, 2)
        sums[i] = sum(view(a, :, i))
    end
    return findall(y -> y ≈ 0, sums)
end

"""
    GSSolverWs

Workspace for solving with the GeneralizedSchur solver.
Can be constructed using a matrix and the number of stable solutions, i.e.
`GSSolverWs(A, n)` with `A` an example `Matrix` and `n` the number of solutions. 
""" 
mutable struct GSSolverWs{T<:AbstractFloat}
    tmp1::Matrix{T}
    tmp2::Matrix{T}
    g1::Matrix{T}
    g2::Matrix{T}
    d::Matrix{T}
    e::Matrix{T}
    luws1::LUWs
    luws2::LUWs
    schurws::GeneralizedSchurWs{T}
    
end

function GSSolverWs(a0::AbstractMatrix)
    n_stable = size(a0, 2) - length(zero_cols(a0)) 
    sums = zeros(size(a0, 2))
    n = size(a0, 1)
    n2 = n - n_stable
    
    tmp1 = similar(a0, n_stable, n_stable)
    tmp2 = similar(a0, n_stable, n_stable)
    g1   = similar(a0, n_stable, n_stable)
    g2   = similar(a0, max(0, n2), n_stable)
    luws1 = LUWs(n_stable)
    luws2 = LUWs(max(0, n2))
    schurws = GeneralizedSchurWs(a0)
    GSSolverWs(tmp1,tmp2,g1,g2, similar(a0), similar(a0), luws1, luws2, schurws)
end

n_stable(ws::GSSolverWs) = size(ws.g1, 1)

"""
    solve!([ws::GSSolverWs,] d::Matrix, e::Matrix, n1::Int64, qz_criterium)

The solution is returned in `ws.g1` and `ws.g2`.
`d` and `e` are mutated during the solving.
`n1` determines how many stable solutions should be found.
"""
function solve!(ws::GSSolverWs{T}, d::Matrix{T}, e::Matrix{T}, args...) where {T<:AbstractFloat}
    copy!(ws.d, d)
    copy!(ws.e, e)
    solve!(ws, args...)
    copy!(d, ws.d)
    copy!(e, ws.e)
end

function solve!(ws::GSSolverWs{T}, a0::Matrix{T}, a1::Matrix{T}, a2::Matrix{T}, args...) where {T<:AbstractFloat}
    n = size(a1,2)
    nstable = n_stable(ws)
    @views begin
        ws.e[:, 1:nstable]   .= .-a0[:, 1:nstable]
        ws.e[:, nstable+1:n] .= .-a1[:, nstable+1:n]
        ws.d[:, 1:nstable]   .=  a1[:, 1:nstable]
        ws.d[:, nstable+1:n] .=  a2[:, nstable+1:n]
    end
    solve!(ws, args...)
end

function solve!(ws::GSSolverWs, qz_criterium::Number = 1 + 1e-6) 
    gges!(ws.schurws, 'N', 'V', ws.e, ws.d; select = (αr, αi, β) -> αr^2 + αi^2 < qz_criterium * β^2)
    
    nstable = ws.schurws.sdim[]::Int

    n = size(ws.d, 1)

    if nstable < n_stable(ws)
        throw(UnstableSystemException())
    elseif nstable > n_stable(ws)
        throw(UndeterminateSystemException())
    end
    
    transpose!(ws.g2, view(ws.schurws.vsr, 1:nstable, nstable+1:n))
    lu_t = LU(factorize!(ws.luws2, view(ws.schurws.vsr,nstable+1:n, nstable+1:n))...)
    ldiv!(lu_t', ws.g2)
    lmul!(-1.0,ws.g2)
    
    transpose!(ws.tmp1, view(ws.schurws.vsr, 1:nstable, 1:nstable))
    lu_t = LU(factorize!(ws.luws1, view(ws.d, 1:nstable,1:nstable))...)
    ldiv!(lu_t', ws.tmp1)

    transpose!(ws.tmp2, view(ws.e, 1:nstable,1:nstable))
    lu_t = LU(factorize!(ws.luws1, view(ws.schurws.vsr,1:nstable, 1:nstable))...)
    ldiv!(lu_t', ws.tmp2)
    mul!(ws.g1, ws.tmp1', ws.tmp2', 1.0, 0.0)
    
    return ws.g1
end

solve!(d::Matrix, e::Matrix, n1::Int, args...) =
    solve!(GSSolverWs(d, n1), d, e, n1, args...)

solve(d::Matrix, e::Matrix, n1::Int, args...) =
    solve!(similar(d), similar(e), n1, args...)
