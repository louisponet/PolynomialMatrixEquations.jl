module PolynomialMatrixEquations

struct UnstableSystemException <: Exception end
struct UndeterminateSystemException <: Exception end
    
using FastLapackInterface
using LinearAlgebra
using SparseArrays
using LoopVectorization


include("cyclic_reduction.jl")
export CyclicReductionWs

include("gs_solver.jl")
export GSSolverWs

end # module
