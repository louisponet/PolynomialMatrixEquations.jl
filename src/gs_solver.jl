using LinearAlgebra.LAPACK: gges!

struct GSSolverWs{T<:AbstractFloat}
    tmp1::Matrix{T}
    tmp2::Matrix{T}
    x1::Matrix{T}
    x2::Matrix{T}
    luws1::LUWs
    luws2::LUWs
    schurws::GeneralizedSchurWs{T}
    
end
function GSSolverWs(d::AbstractMatrix, n1::Int)
    n = size(d, 1)
    n2 = n - n1
    tmp1 = similar(d, n1, n1)
    tmp2 = similar(d, n1, n1)
    x1   = similar(d, n1, n1)
    x2   = similar(d, n2, n1)
    luws1 = LUWs(tmp1)
    luws2 = LUWs(n2)
    schurws = GeneralizedSchurWs(d)
    GSSolverWs(tmp1,tmp2,x1,x2, luws1, luws2, schurws)
end

"""
    solve!([ws::GSSolverWs,] d::Matrix, e::Matrix, n1::Int64, qz_criterium)

The solution is returned in `ws.x1` and `ws.x2`.
`d` and `e` are mutated during the solving.
`n1` determines how many stable solutions should be found.
"""
function solve!(ws::GSSolverWs{T}, d::Matrix{T}, e::Matrix{T}, n1::Int64, qz_criterium::Number = 1 + 1e-6) where {T<:AbstractFloat}
    gges!(ws.schurws, 'N', 'V', e, d; select = (αr, αi, β) -> αr^2 + αi^2 < qz_criterium * β^2)
    nstable = ws.schurws.sdim[]::Int
    n = size(d, 1)
    if nstable < n1
        throw(UnstableSystemException())
    elseif nstable > n1
        throw(UndeterminateSystemException())
    end
    
    transpose!(ws.x2, view(ws.schurws.vsr, 1:nstable, nstable+1:n))
    lu_t = LU(factorize!(ws.luws2, view(ws.schurws.vsr,nstable+1:n, nstable+1:n))...)
    ldiv!(lu_t', ws.x2)
    lmul!(-1.0,ws.x2)
    
    transpose!(ws.tmp1, view(ws.schurws.vsr, 1:nstable, 1:nstable))
    lu_t = LU(factorize!(ws.luws1, view(d, 1:nstable,1:nstable))...)
    ldiv!(lu_t', ws.tmp1)

    transpose!(ws.tmp2, view(e,1:nstable,1:nstable))
    lu_t = LU(factorize!(ws.luws1, view(ws.schurws.vsr,1:nstable, 1:nstable))...)
    ldiv!(lu_t', ws.tmp2)
    mul!(ws.x1, ws.tmp1', ws.tmp2', 1.0, 0.0)
    return ws.x1, ws.x2
end

solve!(d::Matrix, e::Matrix, n1::Int, args...) =
    solve!(GSSolverWs(d, n1), d, e, n1, args...)

solve(d::Matrix, e::Matrix, n1::Int, args...) =
    solve!(similar(d), similar(e), n1, args...)
