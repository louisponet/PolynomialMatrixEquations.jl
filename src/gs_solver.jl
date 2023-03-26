using LinearAlgebra.LAPACK: gges!

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
function GSSolverWs(d::AbstractMatrix, n1::Int)
    n = size(d, 1)
    n2 = n - n1
    tmp1 = similar(d, n1, n1)
    tmp2 = similar(d, n1, n1)
    g1   = similar(d, n1, n1)
    g2   = similar(d, max(0,n2), n1)
    luws1 = LUWs(n1)
    luws2 = LUWs(max(0,n2))
    schurws = GeneralizedSchurWs(d)
    GSSolverWs(tmp1,tmp2,g1,g2, similar(d), similar(d), luws1, luws2, schurws)
end

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

function solve!(ws::GSSolverWs{T}, a0::Matrix{T}, a1::Matrix{T}, a2::Matrix{T}, nstable::Int, args...) where {T<:AbstractFloat}
    n = size(a1,2)
    # nstable = size(ws.tmp1, 1)
    @views begin
        ws.d = [a1 a2; diagm(0 => ones(n)) zeros(n, n)]
        ws.e = [-a0 -a1; zeros(n,n) diagm(0 => ones(n))]
        ws.tmp1 = similar(ws.tmp1, nstable, nstable)
        ws.tmp1 = similar(ws.tmp2, nstable, nstable)
        # ws.luws1 = LUWs(nstable)
        ws.g1 = similar(ws.g1, nstable, nstable)
        ws.g2 = similar(ws.g2, n-nstable, nstable)
        solve!(ws, nstable)
    end
    # for i = 1+n:2n
    #     ws.d[i, i-n] = 1.0
    #     ws.e[i, i] = 1.0
    # end
    # solve!(ws, args...)
end

function solve!(ws::GSSolverWs, n1::Int64, qz_criterium::Number = 1 + 1e-6) 
    gges!(ws.schurws, 'N', 'V', ws.e, ws.d; select = (αr, αi, β) -> αr^2 + αi^2 < qz_criterium * β^2)
    nstable = ws.schurws.sdim[]::Int
    n = size(ws.d, 1)
    if nstable < n1
        throw(UnstableSystemException())
    elseif nstable > n1
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
    return ws.g1, ws.g2
end

solve!(d::Matrix, e::Matrix, n1::Int, args...) =
    solve!(GSSolverWs(d, n1), d, e, n1, args...)

solve(d::Matrix, e::Matrix, n1::Int, args...) =
    solve!(similar(d), similar(e), n1, args...)
