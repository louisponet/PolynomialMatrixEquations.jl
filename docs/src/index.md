# PolynomialMatrixEquations.jl

This package contains algorithm to solve various polynomial matrix equations.

# Cyclic Reduction
This algorithm adapted from Bini et al. (??) solves the second order matrix equation iteratively:
```math
A_0 + A_1 X + A_2 X^2 = \bm{0}
```
for $X$. It is implemented both for $A_0$, $A_1$ and $A_2$ dense or sparse (i.e. `SparseMatrixCSC`).
In the sparse version, a special packing is performed that maximally exploits the column and row sparsity
of the $A$ matrices.

```@docs
PolynomialMatrixEquations.solve!(::CyclicReductionWs{T}, ::Matrix{T}, ::Matrix{T}, ::Matrix{T}, ::Matrix{T}) where {T<:AbstractFloat}
```

# Generalized Schur Solver
This solver finds a solution to the following equation:
```math
D \begin{bmatrix}
I \\ X_2
\end{bmatrix} X_1
=
E \begin{bmatrix}
I \\ X_2
\end{bmatrix}
```
 where matrices $X_1$ and $X_2$ contain columns from the solution matrix $X$.

 ```@docs
PolynomialMatrixEquations.solve!(::GSSolverWs{T}, ::Matrix{T}, ::Matrix{T}, ::Int, ::Number) where {T<:AbstractFloat}
```
