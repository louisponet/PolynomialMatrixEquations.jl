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
    iter = 0
    fill!(X, 0.0)
    while iter <= maxiter
        copy!(ws.B_tmp, B)
        mul!(ws.B_tmp, C, X, 1.0, 1.0)
        
        lu_t = LU(factorize!(ws.linws, ws.B_tmp)...)
        copy!(ws.A_tmp, A)
        ldiv!(lu_t, ws.A_tmp)
        lmul!(-1.0, ws.A_tmp)

        # Conv
        err = 0.0
        for (a, b) in zip(X, ws.A_tmp) 
            err += abs(a-b)
        end
        if err < tol
            return X
        end
        
        X .= ws.A_tmp
        iter += 1
    end
    if iter > maxiter
        XP = maximum(x->abs(x), eigvals(X))
    
        if XP > 1.0
            throw(UnstableSystemException())
        else
            throw(UndeterminateSystemException())
        end
    end 
    return X
end

