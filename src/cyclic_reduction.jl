using LinearAlgebra.BLAS: gemm!

"""
    CRSolverWs

Workspace used for solving with the cyclic reduction algorithm.
Can be constructed as `CRSolverWs(n)` with `n` the leading dimension of
the matrics \$A_0\$, \$A_1\$ and \$A_2\$, i.e. the number of equations.
"""
mutable struct CRSolverWs{T, WS, MT<:AbstractMatrix{T}}
    linsolve_ws::WS
    ahat1::Matrix{T}
    a1copy::Matrix{T}
    x::MT
    m::Matrix{T}
    m1::Matrix{T}
    m2::Matrix{T}
end

function CRSolverWs(a0::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(a0,1)
    linsolve_ws = LUWs(n)
    ahat1 = Matrix{T}(undef, n,n)
    a1copy = Matrix{T}(undef, n,n)
    m = Matrix{T}(undef, 2*n,2*n)
    m1 = Matrix{T}(undef, n, 2*n)
    m2 = Matrix{T}(undef, 2*n, n)
    CRSolverWs(linsolve_ws, ahat1, a1copy, similar(a0), m, m1, m2)
end

"""
    solve!([ws::CRSolverWs, ],
           x::AbstractMatrix,
           a0::AbstractMatrix,
           a1::AbstractMatrix,
           a2::AbstractMatrix;
           tolerance=1e-8,
           iterations=100)

Solves the quadratic matrix equation `a0 + a1*x + a2*x*x = 0`, using the cyclic reduction method from Bini et al. (???).
If `a0` and `a2` are `SparseMatrixCSC`, a variation will be used that optimally packs the equations. `a1` will always be used (i.e. potentially converted) as standard `Matrix`.

The solution is returned in `x`. In case of nonconvergency, `x` is set to `NaN` and 
`UndeterminateSystemExcpetion` or `UnstableSystemException` is thrown.

During the solving, `x`, `a1` and `ws` are mutated.
Use `solve(a0, a1, a2)` for a non-mutating version.

# Example
```jldoctest
julia> using PolynomialMatrixEquations

julia> using LinearAlgebra

julia> n = 3;

julia> ws = CRSolverWs(n);

julia> a0 = [0.5 0 0; 0 0.5 0; 0 0 0];

julia> a1 = Matrix(1.0I, n, n);

julia> a2 = [0 0 0; 0 0 0; 0 0 0.8];

julia> x = zeros(n,n);

julia> PolynomialMatrixEquations.solve!(ws, x, a0, a1, a2, tolerance = 1e-8, iterations = 50)
3×3 Matrix{Float64}:
 -0.5  -0.0  -0.0
 -0.0  -0.5  -0.0
 -0.0  -0.0  -0.0
```
"""
function solve!(ws::CRSolverWs{T}, a0::Matrix{T}, a1::Matrix{T}, a2::Matrix{T};
                tolerance::Number = 1e-8, iterations::Int=100) where {T<:AbstractFloat}
    n = size(a0, 1)
    
    x = ws.x
    copy!(x, a0)
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
    
    @inbounds while it < iterations
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
        check_convergence!(x, it, crit, crit2, view(m1, 1:n, 1:n), tolerance, iterations) && break
        it += 1
    end
    lu_t = LU(factorize!(ws.linsolve_ws, ws.ahat1)...)
    ldiv!(lu_t, x)
    @inbounds lmul!(-1.0, x)
    return x
end

function solve!(ws::CRSolverWs, a0::SparseMatrixCSC, a1_::AbstractMatrix, a2::SparseMatrixCSC;
                           tolerance::Number = 1e-8, iterations::Int=100)
    n = size(a0,1)
    l1 = length(a1_)

    # Copy a0 for the final ldiv!
    x = ws.x
    copy!(ws.x, a0)
    
    fill!(ws.m1, 0.0)
    fill!(ws.m2, 0.0)
    m1_ = ws.m1
    m2_ = ws.m2
    a1 = Matrix(a1_)
    unsafe_copyto!(ws.ahat1, 1, a1, 1, l1)

    # Always better to use a regular matrix for a1

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
    @inbounds while it <= iterations
        # Solve m = [a0; a2]*(a1\[a0 a2])
        unsafe_copyto!(ws.a1copy, 1, a1, 1, l1)
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
        check_convergence!(x, it, crit, crit2, m1, tolerance, iterations) && break
        it += 1
    end
    
    lu_t = LU(factorize!(ws.linsolve_ws, ws.ahat1)...)
    ldiv!(lu_t, x)
    @inbounds lmul!(-1.0, x)
    return x
end

solve!(x::AbstractMatrix{T}, a0::SparseMatrixCSC, a1, a2::SparseMatrixCSC; kwargs...) where {T} =
    solve!(CRSolverWs(T, size(a1, 1)), x, a0, a1, a2; kwargs...)
    
solve(a0, a1, a2; kwargs...) =
    solve!(similar(Matrix(a1)), a0, copy(a1), a2;  kwargs...)

function check_convergence!(x, it, crit1, crit2, m1, tolerance, iterations)
    if isnan(crit1 + crit2) 
        fill!(x, NaN)
        if m1[1] < Inf
            throw(UndeterminateSystemException())
        else
            throw(UnstableSystemException())
        end
    end
    
    # Check whether convergence criterion is met
    if crit1 < tolerance
        if crit2 < tolerance
            return true
        end
    end
    if it == iterations
        @error "Max iterations reached without converging"
        fill!(x, NaN)
        if norm(m1) < tolerance
            throw(UnstableSystemException())
        else
            throw(UndeterminateSystemException())
        end
        return true
    end
    return false
end
