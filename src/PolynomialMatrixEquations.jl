module PolynomialMatrixEquations

struct UnstableSystemException <: Exception end
struct UndeterminateSystemException <: Exception end
    
using FastLapackInterface
using LinearAlgebra
using SparseArrays
using LoopVectorization


include("CyclicReduction.jl")
export CyclicReductionWs

include("GeneralizedSchurDecompositionSolver.jl")
export GSSolverWs

end # module
