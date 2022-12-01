using LinearAlgebra
using LinearAlgebra.BLAS: gemm!
export CyclicReductionWs, cyclic_reduction!, cyclic_reduction_check
using SparseArrays
using LoopVectorization

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
function cyclic_reduction!(x::MT, a0::MT, a1::MT, a2::MT,
                           ws::CyclicReductionWs{MT},
                           cvg_tol::Float64,
                           max_it::Int) where {MT<:Matrix}
    n = size(a0,1)
    x .= a0
    m  = ws.m
    m1 = ws.m1
    m2 = ws.m2
    ws.ahat1 .= a1
    
    @views @inbounds begin
        m1[1:n, 1:n]    .= a0
        m1[1:n, n+1:2n] .= a2
        m2[1:n, 1:n]    .= a0
        m2[n+1:2n, 1:n] .= a2
    end
    it = 0
    
    @inbounds while it < max_it
        # ws.m = [a0; a2]*(a1\[a0 a2])
        ws.a1copy .= a1
        lu_t = LU(factorize!(ws.linsolve_ws, ws.a1copy)...)
        ldiv!(lu_t, m1)
        
        m = mul!(m, m2, m1, -1.0, 0.0)
        crit = 0.0
        crit2 = 0.0
        for i in 1:n
            for j in 1:n
                t  = m[j, i]
                j2 = j + n
                i2 = i + n
                m1[j, i] = m2[j, i] = t 
                crit += abs(t)
                t2 = m[j2, i2] 
                t3 = m[j, i2] 
                t4 = m[j2, i] 
                t5 = t3 + t4
                m1[j, i2] = m2[j2, i] = t2
                crit2 += abs(t2)
                ws.ahat1[j, i] += t4
                a1[j, i] += t5
            end
        end
        
        if crit + crit2 == NaN 
            fill!(x, NaN)
            if norm(view(m1, 1:n, 1:n)) < Inf
                throw(UndeterminateSystemException())
            else
                throw(UnstableSystemException())
            end
        end
        # keep iterating until condition on a2 is met
        if crit < cvg_tol
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
        @inbounds lmul!(-1.0, x)
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

function cyclic_reduction!(x, a0::SparseMatrixCSC, a1_, a2::SparseMatrixCSC,
                           ws::CyclicReductionWs, cvg_tol, max_it::Int)
    n = size(a0,1)

    # Copy a0 for the final ldiv!
    x .= a0
    
    fill!(ws.m1, 0.0)
    fill!(ws.m2, 0.0)
    m1_ = ws.m1
    m2_ = ws.m2
    ws.ahat1 .= a1_

    # Always better to use a regular matrix for a1
    a1 = Matrix(a1_)

    # fill m1 = [a0 a2]
    @inbounds begin
        #=
        We pack nonzero columns in the left hand side of m1
        i.e. m1 = [a0n a2n 0], where a0n and a2n contain nonzero
        columns of a0 and a2, respectively.
        =#
        m1_nonzero_cols = Int[]
        cur_m1_col = 1
        
        rows = rowvals(a0)
        vals = nonzeros(a0)
        for j in 1:n
            a0r = nzrange(a0, j)
            if length(a0r) != 0
                push!(m1_nonzero_cols, j)
                for i in a0r
                    m1_[rows[i], cur_m1_col] = vals[i]
                end
                cur_m1_col += 1
            end
        end

        # Separates a0n and a2n columns (last a0n column) 
        col_divisor = cur_m1_col - 1
        rows = rowvals(a2)
        vals = nonzeros(a2)
        for j in 1:n
            a2r = nzrange(a2, j)
            if length(a2r) != 0
                push!(m1_nonzero_cols, j+n)
                for i in a2r
                    m1_[rows[i], cur_m1_col] = vals[i]
                end
                cur_m1_col += 1
            end
        end
        m1_n_cols = length(m1_nonzero_cols)
        
    end

    # fill m2 = [a0; a2]
    @inbounds begin
        m2_rows_hit = falses(2n)
        m2_nonzero_cols = Int[]
        cur_m2_col = 1
        
        #=
        We pack the nonzero columns similarly to m1,
        while keeping track of any row that contains
        at least 1 nonzero entry.
        Later we pack the rows of m2 such that all nonzero
        are on top.
        =#
        rows0 = rowvals(a0)
        vals0 = nonzeros(a0)
        rows2 = rowvals(a2)
        vals2 = nonzeros(a2)
        for j in 1:n
            a0r = nzrange(a0, j)
            a2r = nzrange(a2, j)
            if length(a0r) + length(a2r) != 0
                push!(m2_nonzero_cols, j)
                for i in a0r
                    rid = rows0[i] 
                    m2_[rid, cur_m2_col] = vals0[i]
                    m2_rows_hit[rid] = true
                end
                for i in a2r
                    rid = rows2[i]
                    m2_[rid + n, cur_m2_col] = vals2[i]
                    m2_rows_hit[rid + n] = true
                end
                cur_m2_col += 1
            end
        end
        m2_n_cols = length(m2_nonzero_cols)

        #=
        Now that we know the nonzero rows,
        we pack them to the top of m2.
        =#
        # This will separate the nonzero a0 rows from the nonzero a2 rows
        row_divisor = 0
         
        m2_nonzero_rows = findall(m2_rows_hit)
        m2_zero_rows = findall(!, m2_rows_hit)
        m2_n_rows = length(m2_nonzero_rows)
        for j in 1:m2_n_cols
            for (i, rid) in enumerate(m2_nonzero_rows)
                if row_divisor == 0 && rid > n 
                    row_divisor = i - 1
                end
                # rid >= i always
                m2_[i, j] = m2_[rid, j]
            end
        end
        
    end
    #=
    By having left packed nonzero m2 columns,
    we have effectively reordered the variables, which we have to
    mimic in a1.
    =#
    @inbounds for (i, id) in enumerate(m2_nonzero_cols)
        for j in 1:n
            a1[j, id], a1[j, i] = a1[j,i], a1[j,id]
        end
    end
    
    m1 = view(ws.m1, :,           1:m1_n_cols)
    m2 = view(ws.m2, 1:m2_n_rows, 1:m2_n_cols)
    m  = view(ws.m,  1:m2_n_rows, 1:m1_n_cols)
    
    m20_zero_rows = filter(x -> x <= n, m2_zero_rows)
    m22_zero_rows = filter(x -> x > n, m2_zero_rows) .- n 

    it = 0
    @inbounds while it < max_it
        # Solve m = [a0; a2]*(a1\[a0 a2])
        ws.a1copy .= a1
        lu_t = LU(factorize!(ws.linsolve_ws, ws.a1copy)...)
        ldiv!(lu_t, m1)
        m = mul!(m, m2, view(m1, 1:m2_n_cols, :), -1.0, 0.0)
        
        crit  = 0.0
        crit2 = 0.0
        
        # Process a0 columns of m
        nid = 0
        for i in 1:col_divisor
            nid2 = m1_nonzero_cols[i]
            nid = findnext(isequal(nid2), m2_nonzero_cols, nid+1)
            v = view(m, :, i)
            m1v = view(m1, :, i)
            m2v = view(m2, :, nid)

            # Copy m[a0, a0] into m2[a0]
            unsafe_copyto!(pointer(m2v), pointer(v), row_divisor)
            
            # Extract m1[a0] from m[a0,a0]
            @turbo for j in 1:row_divisor
                t = v[j]
                tid = m2_nonzero_rows[j] 
                crit += abs(t)
                m1v[tid] = t
            end
            
            # increment a1s with m[a2,a0]
            @turbo for j in row_divisor+1:m2_n_rows
                t = v[j]
                tid = m2_nonzero_rows[j] - n
                a1[tid, nid] += t
                ws.ahat1[tid, nid2] += t
            end

            # Restore zero rows of m1
            m1v[m20_zero_rows] .= 0.0
            
        end

        # Process a2 columns of m
        nid = 0
        for i in col_divisor+1:m1_n_cols
            nid = findnext(isequal(m1_nonzero_cols[i] - n), m2_nonzero_cols, nid+1)
            
            # increment a1 with m[a0,a2]
            for j in 1:row_divisor
                tid = m2_nonzero_rows[j]
                a1[tid, nid] += m[j, i]
            end
            
            # Extract m1[a2] and m2[a2] from m[a2,a2]
            @turbo for j in row_divisor+1:m2_n_rows
                t = m[j, i]
                m1[m2_nonzero_rows[j]-n, i] = t
                m2[j, nid] = t
                crit2 += abs(t)
            end
            m1[m22_zero_rows, i] .= 0.0
        end

        # Check whether something went wrong
        if crit + crit2 == NaN 
            fill!(x, NaN)
            if norm(m1) < Inf
                throw(UndeterminateSystemException())
            else
                throw(UnstableSystemException())
            end
        end
        
        # Check whether convergence criterion is met
        if crit < cvg_tol
            if crit2 < cvg_tol
                break
            end
        end
        it += 1
    end
    if it == max_it
        @error "Max iterations reached without converging"
        if norm(m1) < cvg_tol
            throw(UnstableSystemException())
        else
            throw(UndeterminateSystemException())
        end
        fill!(x, NaN)
        return x
    else
        lu_t = LU(factorize!(ws.linsolve_ws, ws.ahat1)...)
        ldiv!(lu_t, x)
        @inbounds lmul!(-1.0, x)
        return x
    end
end
