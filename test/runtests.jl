using LinearAlgebra
using PolynomialMatrixEquations
using PolynomialMatrixEquations: solve!, solve
using PolynomialMatrixEquations.SparseArrays: sparse
const PME = PolynomialMatrixEquations
using Random
using Test

undeterminatcase = false
unstablecas = false
numberundeterminate = 0
numberunstable = 0

qz_criterium = 1 + 1e-6
@testset "all" begin
    Random.seed!(123)
    ncases = 20
    n=10
    d_origs = [randn(n,n) for i = 1:ncases]
    e_origs = [randn(n,n) for i = 1:ncases]
    for i = 1:ncases
        @testset "$i" begin
            d_orig = d_origs[i]
            e_orig = e_origs[i]
            F = schur(e_orig, d_orig)
            eigenvalues = F.α ./ F.β
            nstable = count(abs.(eigenvalues) .< 1+1e-6)

            d = copy(d_orig)
            e = copy(e_orig)

            @testset "GS Solve" begin
                ws1 = GSSolverWs(d, nstable)
                solve!(ws1, d, e, nstable, qz_criterium)
                @test d_orig*[I(nstable); ws1.g2]*ws1.g1 ≈ e_orig*[I(nstable); ws1.g2]
                
                nstable1 = nstable + 1
                @test_throws PME.UnstableSystemException solve!(ws1, d, e, nstable + 1, qz_criterium)
                
                nstable1 = nstable - 1
                 
                @test_throws PME.UndeterminateSystemException solve!(ws1, d, e, nstable1, qz_criterium)
            end
            @testset "CRSolverWs Reduction" begin
                @testset "dense" begin
                    a0 = Matrix([-e[:, 1:nstable] zeros(n, n-nstable)])
                    a1 = Matrix([d[:, 1:nstable] -e[:, (nstable+1):n]])
                    a2 = Matrix([zeros(n, nstable) d[:, (nstable+1):n]])

                    x = zeros(n, n)
                    ws2 = CRSolverWs(n)
                    solve!(ws2, x, a0, a1, a2, tolerance = 1e-8, iterations = 50)
                    @test isapprox(a0 + a1*x + a2*x*x, zeros(n, n); atol = 1e-12)

                    nstable1 = nstable + 1
                    
                    a0 = Matrix([-e[:, 1:nstable1] zeros(n, n-nstable1)])
                    a1 = Matrix([d[:, 1:nstable1] -e[:, (nstable1+1):n]])
                    a2 = Matrix([zeros(n, nstable1) d[:, (nstable1+1):n]])

                    x = zeros(n, n)
                    ws2 = CRSolverWs(n)
                    @test_throws PME.UnstableSystemException solve!(ws2, x, a0, a1, a2, tolerance=1e-8, iterations=50)

                    nstable1 = nstable - 1
                    a0 = Matrix([-e[:, 1:nstable1] zeros(n, n-nstable1)])
                    a1 = Matrix([d[:, 1:nstable1] -e[:, (nstable1+1):n]])
                    a2 = Matrix([zeros(n, nstable1) d[:, (nstable1+1):n]])

                    x = zeros(n, n)
                    ws2 = CRSolverWs(n)
                    @test_throws PME.UndeterminateSystemException solve!(ws2, x, a0, a1, a2, tolerance=1e-8, iterations=50)
                end
                @testset "sparse" begin
                    a0 = sparse([-e[:, 1:nstable] zeros(n, n-nstable)])
                    a1 = sparse([d[:, 1:nstable] -e[:, (nstable+1):n]])
                    a2 = sparse([zeros(n, nstable) d[:, (nstable+1):n]])

                    x = zeros(n, n)
                    ws2 = CRSolverWs(n)
                    solve!(ws2, x, a0, a1, a2, tolerance = 1e-8, iterations = 50)
                    @test isapprox(a0 + a1*x + a2*x*x, zeros(n, n); atol = 1e-12)

                    nstable1 = nstable + 1
                    
                    a0 = sparse([-e[:, 1:nstable1] zeros(n, n-nstable1)])
                    a1 = sparse([d[:, 1:nstable1] -e[:, (nstable1+1):n]])
                    a2 = sparse([zeros(n, nstable1) d[:, (nstable1+1):n]])

                    x = zeros(n, n)
                    ws2 = CRSolverWs(n)

                    @test_throws PME.UnstableSystemException solve!(ws2, x, a0, a1, a2, tolerance=1e-8, iterations=50)

                    nstable1 = nstable - 1

                    a0 = sparse([-e[:, 1:nstable1] zeros(n, n-nstable1)])
                    a1 = [d[:, 1:nstable1] -e[:, (nstable1+1):n]]
                    a2 = sparse([zeros(n, nstable1) d[:, (nstable1+1):n]])

                    x = zeros(n, n)
                    ws2 = CRSolverWs(n)
                    @test_throws PME.UndeterminateSystemException solve!(ws2, x, a0, a1, a2, tolerance=1e-8, iterations=50)
                end
            end
        end

    end
end
