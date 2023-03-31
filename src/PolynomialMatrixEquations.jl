module PolynomialMatrixEquations

struct UnstableSystemException <: Exception end
struct UndeterminateSystemException <: Exception end
    
using FastLapackInterface
using FastLapackInterface: Workspace
using LinearAlgebra
using SparseArrays
using LoopVectorization

include("cyclic_reduction.jl")
include("gs_solver.jl")

"""
    Workspace(a0, a1, a2)
    Workspace(d, e, n)

Will return the right workspace for [`solve!`](@ref) on a given set of `a0`, `a1` and `a2`, or matrices `d`, `e` and number of stable solutions `n`.
If `DenseMatrices` are passed in, or `d` and `e`, it will return a [`GSSolverWs`](@ref), else a [`CRSolverWs`](@ref). 
"""
Workspace(a0::DenseMatrix, a1::DenseMatrix, a2::DenseMatrix, args...)            = GSSolverWs(a0)
Workspace(d::DenseMatrix, e::DenseMatrix, n::Int, args...)                       = GSSolverWs(d, n)
Workspace(a0::SparseMatrixCSC, a1::AbstractMatrix, a2::SparseMatrixCSC, args...) = CRSolverWs(a0)

"""
    solve

Will first construct the right [`Workspace`](@ref) and then call [`solve!`](@ref).
"""
solve(args...; kwargs...) = solve!(Workspace(args...); kwargs...)

"""
    solve!(ws, a0, a1, a2; kwargs...)

In place solve of the Quadrilateral Matrix Equation \$a0 + a1 * x + a2 * x * x = 0\$ for `x` and returns `x`.

# Uses:
- [Generalized Schur](https://www.sciencedirect.com/science/article/pii/S0165188999000457) when `ws` is a [`GSSolverWs`](@ref) and `a0`, `a1` and `a2` are `DenseMatrices`
- [Cyclic Reduction](https://link.springer.com/article/10.1007/s11075-008-9253-0) when `ws` is a [`CRSolverWs`](@ref) and at least `a0` and `a2` are `SparseMatrixCSC`

# Further `kwargs`:
- `tolerance` <1e-8>: convergence tolerance when using cyclic reduction,
                      or tolerance for stable eigenvalues of the schur decomposition (`eigval > 1 + tolerance` is stable)
- `max_iterations` <100>: maximum iterations allowed in cyclic reduction

# Examples:

```jldoctest
julia> using PolynomialMatrixEquations

julia> const PME = PolynomialMatrixEquations
PolynomialMatrixEquations

julia> a0 = [1.21 0.0 0.0
             0.27 0.0 0.0
             1.93 0.0 0.0];

julia> a1 = [0.81  -0.68 0.22
             -1.12 0.14 -0.88
             -1.1 0.12 -1.14];

julia> a2 = [0.0 -0.42 -0.42
             0.0 0.29 -1.36
             0.0 0.23 -1.09];

julia> ws2 = PME.Workspace(a0, a1, a2)
PolynomialMatrixEquations.GSSolverWs{Float64}
  tmp1: 1×1 Matrix{Float64}
  tmp2: 1×1 Matrix{Float64}
  g1: 1×1 Matrix{Float64}
  g2: 2×1 Matrix{Float64}
  x: 3×3 Matrix{Float64}
  d: 3×3 Matrix{Float64}
  e: 3×3 Matrix{Float64}
  luws1: FastLapackInterface.LUWs
  luws2: FastLapackInterface.LUWs
  schurws: FastLapackInterface.GeneralizedSchurWs{Float64}

julia> PME.solve!(ws2, a0, a1, a2)
3×3 Matrix{Float64}:
 -0.511005  0.0  0.0
  5.72536   0.0  0.0
  4.29884   0.0  0.0

julia> using SparseArrays

julia> a0 = sparse([1.21 0.0 0.0
                    0.27 0.0 0.0
                    1.93 0.0 0.0]);

julia> a1 = sparse([0.81  -0.68 0.22
                    -1.12 0.14 -0.88
                    -1.1 0.12 -1.14]);

julia> a2 = sparse([0.0 -0.42 -0.42
                    0.0 0.29 -1.36
                    0.0 0.23 -1.09]);

julia> ws2 = PME.Workspace(a0, a1, a2)
PolynomialMatrixEquations.CRSolverWs{Float64, FastLapackInterface.LUWs, SparseMatrixCSC{Float64, Int64}}
  linsolve_ws: FastLapackInterface.LUWs
  ahat1: 3×3 Matrix{Float64}
  a1copy: 3×3 Matrix{Float64}
  x: 3×3 SparseMatrixCSC{Float64, Int64} with 3 stored entries
  m: 6×6 Matrix{Float64}
  m1: 3×6 Matrix{Float64}
  m2: 6×3 Matrix{Float64}

julia> PME.solve!(ws2, a0, a1, a2)
3×3 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
 -0.511005   ⋅    ⋅
  5.72536    ⋅    ⋅
  4.29884    ⋅    ⋅
```
"""
solve!

end # module
