# PolynomialMatrixEquations.jl

This package solves matrix polynomial equations in the form
```math
A_0 + A_1 X + A_2 X^2 = \bm{0}
```
or
```math
D \begin{bmatrix}
I \\ X_2
\end{bmatrix} X_1
=
E \begin{bmatrix}
I \\ X_2
\end{bmatrix}
```
 where matrices ```math X_1``` and ```math X_2` contain columns from the solution matrix
`X`.

