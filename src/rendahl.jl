struct RendahlWs
    A_tmp::Matrix{Float64}
    B_tmp::Matrix{Float64}
    C_tmp::Matrix{Float64}
    S0::Matrix{Float64}
    linws::LUWs
end
RendahlWs(n::Int) = RendahlWs(Matrix{Float64}(undef, n, n),
                              Matrix{Float64}(undef, n, n),
                              Matrix{Float64}(undef, n, n),
                              Matrix{Float64}(undef, n, n),
                              LUWs(n))
                              
    
function rendahl_solve!(X::Matrix, A::Matrix, B::Matrix, C::Matrix, ws::RendahlWs, maxiter=1000, tol=1e-6)
    
    err = one(tol) + tol
    iter = 0
    fill!(ws.S0, 0.0)
    fill!(X, 0.0)
    while err > tol && iter <= maxiter
        copy!(ws.B_tmp, B)
        mul!(ws.B_tmp, C, X, 1.0, 1.0)
        
        lu_t = LU(factorize!(ws.linws, ws.B_tmp)...)
        copy!(ws.A_tmp, A)
        ldiv!(lu_t, ws.A_tmp)
        lmul!(-1.0, ws.A_tmp)
        X .= ws.A_tmp

        copy!(ws.C_tmp, C)
        copy!(ws.B_tmp, B)
        mul!(ws.B_tmp, A, ws.S0, 1.0, 1.0)

        lu_t = LU(factorize!(ws.linws, ws.B_tmp)...)
        ldiv!(lu_t, ws.C_tmp)
        lmul!(-1.0, ws.C_tmp)
        ws.S0 .= ws.C_tmp
        
        copy!(ws.B_tmp, A)
        mul!(ws.B_tmp, B, X, 1.0, 1.0)
        mul!(ws.A_tmp, X, X, 1.0, 0.0)
        mul!(ws.B_tmp, C, ws.A_tmp, 1.0, 1.0)

        err = maximum(ws.B_tmp)
        iter += 1
        
    end
    if iter > maxiter
        XP = maximum(x->abs(x), eigvals(X))
        XS = maximum(x->abs(x), eigvals(ws.S0))
    
        if XP > 1.0
            throw(UnstableSystemException())
        elseif XS > 1.0
            throw(UndeterminateSystemException())
        else
            error("System not converged.")
        end
    end 
    return X
end

