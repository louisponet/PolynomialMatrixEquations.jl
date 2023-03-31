using LinearAlgebra
using PolynomialMatrixEquations
using PolynomialMatrixEquations: solve!, solve, Workspace, CRSolverWs, GSSolverWs
using PolynomialMatrixEquations.SparseArrays: sparse
const PME = PolynomialMatrixEquations
using Random
using Test

undeterminatcase = false
unstablecas = false
numberundeterminate = 0
numberunstable = 0

tolerance = 1e-6
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

            a0_orig = Matrix([-e_orig[:, 1:nstable] zeros(n, n-nstable)])
            a1_orig = Matrix([d_orig[:, 1:nstable] -e_orig[:, (nstable+1):n]])
            a2_orig = Matrix([zeros(n, nstable) d_orig[:, (nstable+1):n]])
            
            gs_result = nothing
            
            @testset "GS Solve" begin
                @testset "d, e" begin
                    d = copy(d_orig)
                    e = copy(e_orig)
                    
                    ws1 = Workspace(d, e, nstable)
                    gs_result = deepcopy(solve!(ws1, d, e; tolerance = tolerance))
                    @test d_orig*[I(nstable); ws1.g2]*ws1.g1 ≈ e_orig*[I(nstable); ws1.g2]
                    
                    nstable1 = nstable + 1
                    a0 = Matrix([-e_orig[:, 1:nstable1] zeros(n, n-nstable1)])
                    ws1 = GSSolverWs(a0)
                    @test_throws PME.UnstableSystemException solve!(ws1, d, e; tolerance=tolerance)
                    
                    nstable1 = nstable - 1
                    a0 = Matrix([-e_orig[:, 1:nstable1] zeros(n, n-nstable1)])
                    ws1 = GSSolverWs(a0)
                    @test_throws PME.UndeterminateSystemException solve!(ws1, d, e; tolerance=tolerance)
                end
                @testset "a0, a1, a2" begin
                    a0 = copy(a0_orig)
                    a1 = copy(a1_orig)
                    a2 = copy(a2_orig)
                    ws1 = GSSolverWs(a0)
                    x = solve!(ws1, a0, a1, a2; tolerance=tolerance)

                    @test isapprox(a1 * x + a2 * x * x + a0, zeros(n, n), atol= 1e-8)
                    
                end
            end
            @testset "CRSolverWs Reduction" begin
                @testset "dense" begin

                    a0 = copy(a0_orig)
                    a1 = copy(a1_orig)
                    a2 = copy(a2_orig)
                    ws2 = CRSolverWs(a0)
                    x = deepcopy(solve!(ws2, a0, a1, a2, tolerance = 1e-20, max_iterations = 500))
                    @test isapprox(a0_orig + a1_orig*x + a2_orig*x*x, zeros(n, n); atol = 1e-8)
                    @test isapprox(x, gs_result; atol=1e-8)

                    nstable1 = nstable + 1
                    
                    a0 = Matrix([-e_orig[:, 1:nstable1] zeros(n, n-nstable1)])
                    a1 = Matrix([d_orig[:, 1:nstable1] -e_orig[:, (nstable1+1):n]])
                    a2 = Matrix([zeros(n, nstable1) d_orig[:, (nstable1+1):n]])

                    ws2 = CRSolverWs(a0)
                    @test_throws PME.UnstableSystemException solve!(ws2, a0, a1, a2, tolerance=1e-8, max_iterations=500)

                    nstable1 = nstable - 1
                    a0 = Matrix([-e_orig[:, 1:nstable1] zeros(n, n-nstable1)])
                    a1 = Matrix([d_orig[:, 1:nstable1] -e_orig[:, (nstable1+1):n]])
                    a2 = Matrix([zeros(n, nstable1) d_orig[:, (nstable1+1):n]])

                    ws2 = CRSolverWs(a0)
                    @test_throws PME.UndeterminateSystemException solve!(ws2, a0, a1, a2, tolerance=1e-8, max_iterations=500)
                end
                @testset "sparse" begin
                    a0 = sparse(a0_orig)
                    a1 = sparse(a1_orig)
                    a2 = sparse(a2_orig)

                    ws2 = Workspace(a0, a1, a2)
                    x = solve!(ws2, a0, a1, a2, tolerance = 1e-20, max_iterations = 500)
                    @test isapprox(a0 + a1*x + a2*x*x, zeros(n, n); atol = 1e-8)

                    nstable1 = nstable + 1
                    
                    a0 = sparse([-e_orig[:, 1:nstable1] zeros(n, n-nstable1)])
                    a1 = sparse([d_orig[:, 1:nstable1] -e_orig[:, (nstable1+1):n]])
                    a2 = sparse([zeros(n, nstable1) d_orig[:, (nstable1+1):n]])

                    ws2 = CRSolverWs(a0)

                    @test_throws PME.UnstableSystemException solve!(ws2, a0, a1, a2, tolerance=1e-12, max_iterations=500)

                    nstable1 = nstable - 1

                    a0 = sparse([-e_orig[:, 1:nstable1] zeros(n, n-nstable1)])
                    a1 = sparse([d_orig[:, 1:nstable1] -e_orig[:, (nstable1+1):n]])
                    a2 = sparse([zeros(n, nstable1) d_orig[:, (nstable1+1):n]])

                    ws2 = CRSolverWs(a0)
                    @test_throws PME.UndeterminateSystemException solve!(ws2,  a0, a1, a2, tolerance=1e-12, max_iterations=500)
                end
            end
        end

    end
end
