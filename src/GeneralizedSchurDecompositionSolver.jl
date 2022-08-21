using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK: gges!
struct GsSolverWs
    tmp1::Matrix{Float64}
    tmp2::Matrix{Float64}
    tmp3::Matrix{Float64}
    Z22::Matrix{Float64}
    g1::Matrix{Float64}
    g2::Matrix{Float64}
    eigval::Vector{ComplexF64}
    
    function GsSolverWs(d,n1)
        n = size(d,1)
        n2 = n - n1
        tmp1 = Matrix{Float64}(undef, n1, n1)
        tmp2 = Matrix{Float64}(undef, n1, n1)
        tmp3 = Matrix{Float64}(undef, n1, n1)
        Z22 = Matrix{Float64}(undef, n2, n2)
        g1 = Matrix{Float64}(undef, n1, n1)
        g2 = Matrix{Float64}(undef, n2, n1)
        eigval = Vector{ComplexF64}(undef, n)
        new(tmp1,tmp2,tmp3,Z22,g1,g2, eigval)
    end
end

function gs_solver!(ws::GsSolverWs,schurws::GeneralizedSchurWs, luws::LUWs, d::Matrix{Float64},e::Matrix{Float64}, n1::Int64, qz_criterium::Float64=1 + 1e-6)

    gges!(schurws, 'N', 'V', e, d; select = (αr, αi, β) -> αr^2 + αi^2 < qz_criterium * β^2)
    nstable = schurws.sdim[]::Int
    n = size(d, 1)
    if nstable < n1
        throw(UnstableSystemException())
    elseif nstable > n1
        throw(UndeterminateSystemException())
    end
    transpose!(ws.g2, view(schurws.vsr, 1:nstable, nstable+1:n))
    ws.Z22 .= view(schurws.vsr,nstable+1:n, nstable+1:n)
    lu_t = LU(factorize!(luws, ws.Z22)...)
    ldiv!(lu_t', ws.g2)
    lmul!(-1.0,ws.g2)
    
    transpose!(ws.tmp2, view(schurws.vsr, 1:nstable, 1:nstable))
    ws.g1 .= view(d, 1:nstable,1:nstable)
    lu_t = LU(factorize!(luws, ws.g1)...)
    ldiv!(lu_t', ws.tmp2)
   
    transpose!(ws.tmp3, view(e,1:nstable,1:nstable))
    ws.g1 .= view(schurws.vsr,1:nstable, 1:nstable)
    lu_t = LU(factorize!(luws, ws.g1)...)
    ldiv!(lu_t', ws.tmp3)
    gemm!('T','T',1.0,ws.tmp2,ws.tmp3,0.0,ws.g1)
end

