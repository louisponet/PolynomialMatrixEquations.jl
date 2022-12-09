var documenterSearchIndex = {"docs":
[{"location":"#PolynomialMatrixEquations.jl","page":"Home","title":"PolynomialMatrixEquations.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package contains algorithm to solve various polynomial matrix equations.","category":"page"},{"location":"#Cyclic-Reduction","page":"Home","title":"Cyclic Reduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This algorithm adapted from Bini et al. [1] solves the second order matrix equation iteratively:","category":"page"},{"location":"","page":"Home","title":"Home","text":"A_0 + A_1 X + A_2 X^2 = bm0","category":"page"},{"location":"","page":"Home","title":"Home","text":"for X. It is implemented both for A_0, A_1 and A_2 dense or sparse (i.e. SparseMatrixCSC). In the sparse version, a special packing is performed that maximally exploits the column and row sparsity of the A matrices.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CyclicReductionWs\nPolynomialMatrixEquations.solve!(::CyclicReductionWs{T}, ::Matrix{T}, ::Matrix{T}, ::Matrix{T}, ::Matrix{T}) where {T<:AbstractFloat}","category":"page"},{"location":"#PolynomialMatrixEquations.CyclicReductionWs","page":"Home","title":"PolynomialMatrixEquations.CyclicReductionWs","text":"CyclicReductionWs\n\nWorkspace used for solving with the cyclic reduction algorithm. Can be constructed as CyclicReductionWs(n) with n the leading dimension of the matrics A_0, A_1 and A_2, i.e. the number of equations.\n\n\n\n\n\n","category":"type"},{"location":"#PolynomialMatrixEquations.solve!-Union{Tuple{T}, Tuple{CyclicReductionWs{T}, Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}}} where T<:AbstractFloat","page":"Home","title":"PolynomialMatrixEquations.solve!","text":"solve!([ws::CyclicReductionWs, ],\n       x::AbstractMatrix,\n       a0::AbstractMatrix,\n       a1::AbstractMatrix,\n       a2::AbstractMatrix;\n       tolerance=1e-8,\n       iterations=100)\n\nSolves the quadratic matrix equation a0 + a1*x + a2*x*x = 0, using the cyclic reduction method from Bini et al. (???). If a0 and a2 are SparseMatrixCSC, a variation will be used that optimally packs the equations. a1 will always be used (i.e. potentially converted) as standard Matrix.\n\nThe solution is returned in x. In case of nonconvergency, x is set to NaN and  UndeterminateSystemExcpetion or UnstableSystemException is thrown.\n\nDuring the solving, x, a1 and ws are mutated. Use solve(a0, a1, a2) for a non-mutating version.\n\nExample\n\njulia> using PolynomialMatrixEquations\n\njulia> using LinearAlgebra\n\njulia> n = 3;\n\njulia> ws = CyclicReductionWs(n);\n\njulia> a0 = [0.5 0 0; 0 0.5 0; 0 0 0];\n\njulia> a1 = Matrix(1.0I, n, n);\n\njulia> a2 = [0 0 0; 0 0 0; 0 0 0.8];\n\njulia> x = zeros(n,n);\n\njulia> PolynomialMatrixEquations.solve!(ws, x, a0, a1, a2, tolerance = 1e-8, iterations = 50)\n3×3 Matrix{Float64}:\n -0.5  -0.0  -0.0\n -0.0  -0.5  -0.0\n -0.0  -0.0  -0.0\n\n\n\n\n\n","category":"method"},{"location":"#Generalized-Schur-Decomposition-Solver","page":"Home","title":"Generalized Schur Decomposition Solver","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The polynomial matrix equation","category":"page"},{"location":"","page":"Home","title":"Home","text":"A_0 + A_1 X + A_2 X^2 = bm0","category":"page"},{"location":"","page":"Home","title":"Home","text":"can be rewritten in the form","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginbmatrix\n0  A_2 \nI  0\nendbmatrix \nbeginbmatrix\nI  X\nendbmatrix X\n=\n-beginbmatrix\nA_0  A_1 \n0  I\nendbmatrix \nbeginbmatrix\nI  X\nendbmatrix","category":"page"},{"location":"","page":"Home","title":"Home","text":"In practive, it is possible to eliminate empty columns in A_0, A_1 and A_2 (See Klein, 2000, and . More generally, this solver finds a solution to the following equation:","category":"page"},{"location":"","page":"Home","title":"Home","text":"D beginbmatrix\nI  X_2\nendbmatrix X_1\n=\nE beginbmatrix\nI  X_2\nendbmatrix","category":"page"},{"location":"","page":"Home","title":"Home","text":"where matrices X_1 and X_2 contain columns from the solution matrix X.","category":"page"},{"location":"","page":"Home","title":"Home","text":"GSSolverWs\nPolynomialMatrixEquations.solve!(::GSSolverWs{T}, ::Matrix{T}, ::Matrix{T}, ::Int, ::Number) where {T<:AbstractFloat}","category":"page"},{"location":"#PolynomialMatrixEquations.GSSolverWs","page":"Home","title":"PolynomialMatrixEquations.GSSolverWs","text":"GSSolverWs\n\nWorkspace for solving with the GeneralizedSchur solver. Can be constructed using a matrix and the number of stable solutions, i.e. GSSolverWs(A, n) with A an example Matrix and n the number of solutions. \n\n\n\n\n\n","category":"type"},{"location":"#PolynomialMatrixEquations.solve!-Union{Tuple{T}, Tuple{GSSolverWs{T}, Matrix{T}, Matrix{T}, Int64, Number}} where T<:AbstractFloat","page":"Home","title":"PolynomialMatrixEquations.solve!","text":"solve!([ws::GSSolverWs,] d::Matrix, e::Matrix, n1::Int64, qz_criterium)\n\nThe solution is returned in ws.x1 and ws.x2. d and e are mutated during the solving. n1 determines how many stable solutions should be found.\n\n\n\n\n\n","category":"method"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"D.A. Bini, B. Iannazzo, B. Meini, *Numerical Solution of Algebraic","category":"page"},{"location":"","page":"Home","title":"Home","text":"Riccati Equations*, SIAM Book Series Fundamentals of Algorithms, 2012.","category":"page"},{"location":"","page":"Home","title":"Home","text":"P. Klein Using the generalized Schur form to solve a linear rational expectations model, Journal of Economci Dynamics and Control, 2000.\nC. Sims Solving rational expectations models, Computational Economics, 2002.","category":"page"}]
}
