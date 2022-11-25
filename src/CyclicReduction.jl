using LinearAlgebra
using LinearAlgebra.BLAS: gemm!
export CyclicReductionWs, cyclic_reduction!, cyclic_reduction_check

mutable struct CyclicReductionWs{MT,WS}
    linsolve_ws::WS
    ahat1::MT
    a1copy::MT
    m::MT
    m1::MT
    m2::MT
    info::Int
end
function CyclicReductionWs(n::Int)
    linsolve_ws = LUWs(n)
    ahat1 = Matrix{Float64}(undef, n,n)
    a1copy = Matrix{Float64}(undef, n,n)
    m = Matrix{Float64}(undef, 2*n,2*n)
    m1 = Matrix{Float64}(undef, n, 2*n)
    m2 = Matrix{Float64}(undef, 2*n, n)
    CyclicReductionWs(linsolve_ws, ahat1, a1copy, m, m1, m2,0) 
end

"""
    cyclic_reduction!(x::Array{Float64},a0::Array{Float64},a1::Array{Float64},a2::Array{Float64},ws::CyclicReductionWs, cvg_tol::Float64, max_it::Int64)

Solve the quadratic matrix equation a0 + a1*x + a2*x*x = 0, using the cyclic reduction method from Bini et al. (???).

The solution is returned in matrix x. In case of nonconvergency, x is set to NaN and 
UndeterminateSystemExcpetion or UnstableSystemException is thrown

# Example
```meta
DocTestSetup = quote
     using CyclicReduction
     n = 3
     ws = CyclicReductionWs(n)
     a0 = [0.5 0 0; 0 0.5 0; 0 0 0];
     a1 = eye(n)
     a2 = [0 0 0; 0 0 0; 0 0 0.8]
     x = zeros(n,n)
end
```

```jldoctest
julia> display(names(CyclicReduction))
```

```jldoctest
julia> cyclic_reduction!(x,a0,a1,a2,ws,1e-8,50)
```
"""
function cyclic_reduction!(x::MT,
                           a0::MT,
                           a1::MT,
                           a2::MT,
                           ws::CyclicReductionWs{MT},
                           cvg_tol::Float64,
                           max_it::Int) where {MT}
    n = size(a0,1)
    x .= a0
    m  = ws.m
    m1 = ws.m1
    m2 = ws.m2
    ws.ahat1 .= a1
    @views @inbounds begin
        m1[1:n, 1:n] .= a0
        m1[1:n, n+1:2n] .= a2
        m2[1:n, 1:n] .= a0
        m2[n+1:2n, 1:n] .= a2
    end
    it = 0
    # nonzero_ranges = UnitRange[]
    # col_r = axes(m1, 2)
    # start = findfirst(i -> any(!iszero, view(m1, :, i)), col_r)
    # last  = findnext(j -> !any(!iszero, view(m1, :, j)), col_r, start)
    
    # while last !== nothing && start !== nothing
    #     t = findnext(j -> !any(!iszero, view(m1, :, j)), col_r, start)
    #     if t !== nothing
    #         last = t - 1
    #         push!(nonzero_ranges, start:last)
    #         start = findnext(i -> any(!iszero, view(m1, :, i)), col_r, t)
    #     else
    #         break
    #     end
    # end
    # if start !== nothing
    #     push!(nonzero_ranges, start:col_r[end])
    # end
        
    @inbounds while it < max_it
        #        ws.m = [a0; a2]*(a1\[a0 a2])
        ws.a1copy .= a1
        lu_t = LU(factorize!(ws.linsolve_ws, ws.a1copy)...)
        ldiv!(lu_t, m1)
        
        m = mul!(m, m2, m1, -1.0, 0.0)
        crit, crit2, issue = process_m!(m1, m2, a1, ws.ahat1, m)
        
        if issue 
            fill!(x, NaN)
            if norm(view(m1, 1:n, 1:n)) < Inf
                throw(UndeterminateSystemException())
            else
                throw(UnstableSystemException())
            end
        end
        if crit < cvg_tol
        # keep iterating until condition on a2 is met
            if crit2 < cvg_tol
                break
            end
        end
        it += 1
    end
    if it == max_it
        println("max_it")
        if norm(view(m1, 1:n, 1:n)) < cvg_tol
            throw(UnstableSystemException())
        else
            throw(UndeterminateSystemException())
        end
        fill!(x,NaN)
        return
    else
        lu_t = LU(factorize!(ws.linsolve_ws, ws.ahat1)...)
        ldiv!(lu_t, x)
        @inbounds lmul!(-1.0,x)
        ws.info = 0
    end
end

function cyclic_reduction_check(x::Array{Float64,2},a0::Array{Float64,2}, a1::Array{Float64,2}, a2::Array{Float64,2},cvg_tol::Float64)
    res = a0 + a1*x + a2*x*x
    if (sum(sum(abs.(res))) > cvg_tol)
        print("the norm of the residuals, ", res, ", compared to the tolerance criterion ",cvg_tol)
    end
    nothing
end

function process_m!(m1::Matrix, m2::Matrix, a1::Matrix, ahat1::Matrix, m::Matrix)
    n = size(m1, 1)
    crit = 0.0
    crit2 = 0.0
    issue = false
    for i in 1:n
        for j in 1:n
            t  = m[j, i]
            t2 = m[j+n, i+n] 
            t3 = m[j, i+n] 
            t4 = m[j+n, i] 
            issue = issue || isinf(t) || isnan(t)
            m1[j, i] = t 
            m2[j, i] = t
            crit += abs(t)
            issue = issue || isinf(t2) || isnan(t2)
            m1[j, i+n] = t2
            m2[j+n, i] = t2
            crit2 += abs(t2)
            issue = issue || isinf(t3) || isnan(t3)
            a1[j, i] += t3
            issue = issue || isinf(t4) || isnan(t4)
            a1[j, i] += t4
            ahat1[j, i] += t4
        end
    end
    return crit, crit2, issue
end

using SparseArrays

function cyclic_reduction_sparse!(x::Matrix,
                           a0::MT,
                           a1::MT,
                           a2::MT,
                           ws::CyclicReductionWs,
                           cvg_tol::Float64,
                           max_it::Int) where {MT}
    x .= a0
    m1 = [Matrix(a0) Matrix(a2)]
    m2 = [a0;a2]
    is, js = Int[], Int[]    
    ahat1 = copy(a1)
    it = 0
    @inbounds while it < max_it
        ws.a1copy .= a1
        lu_t = LU(factorize!(ws.linsolve_ws, ws.a1copy)...)
        ldiv!(lu_t, m1)
        m1s = sparse(m1)
        fill!(m1, 0.0)
        if it == 0
            m = m2 * m1s
            t = zeros(size(m2))
            is, js = findnz(m)[1:2]
            crit, crit2, issue = process_m!(m1, t, a1, ahat1, m, is, js, m.nzval)
            m2 = sparse(t)
        else
            m = m2 * m1s
            crit, crit2, issue = process_m!(m1, m2, a1, ahat1, m, is, js, m.nzval)
        end
        if issue 
            fill!(x, NaN)
            if norm(view(m1,:, 1:size(a0, 2))) < Inf
                throw(UndeterminateSystemException())
            else
                throw(UnstableSystemException())
            end
        end
        if crit < cvg_tol
        # keep iterating until condition on a2 is met
            if crit2 < cvg_tol
                break
            end
        end
        it += 1
    end
    if it == max_it
        println("max_it")
        if norm(view(m1,:, 1:size(a0, 2))) < cvg_tol
            throw(UnstableSystemException())
        else
            throw(UndeterminateSystemException())
        end
        fill!(x,NaN)
        return
    else
        lu_t = lu(ahat1)
        ldiv!(lu_t, x)
        @inbounds lmul!(-1.0,x)
        # ws.info = 0
    end
end


function process_m!(m1::Matrix, m2, a1::SparseMatrixCSC, ahat1::SparseMatrixCSC, m::SparseMatrixCSC, is, js, vs)
    n = size(m1, 1)
    crit = 0.0
    crit2 = 0.0
    issue = false
    @inbounds for id in 1:length(vs)
        j = is[id]
        i = js[id]
        v = -vs[id]
        issue = issue || isinf(v) || isnan(v)
        if j <= n
            if i <= n
                m1[j, i] = v
                m2[j, i] = v
                crit += abs(v)
            else
                a1[j, i-n] += v
            end
        else
            if i <= n
                a1[j-n, i] += v
                ahat1[j-n, i] += v
            else
                m1[j - n, i] = v
                m2[j, i-n] = v
                crit2 += abs(v)
            end
        end
    end
    return crit, crit2, false
end




