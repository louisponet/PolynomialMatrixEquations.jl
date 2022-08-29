function solve_system(A, B, C, linws, maxiter=1000, tol=1e-6)
    
    #==
    Solves for P and Q using Rehndal's Algorithm
    ==#
    C_orig = copy(C)
    B_orig = similar(B)
    A_orig = copy(A)
    C_buf = similar(C)
    A_buf = similar(A)
    B_buf = copy(B)
    
    F0 = zero(A)
    S0 = zero(A)
    error = one(tol) + tol
    iter = 0
    
    while error > tol && iter <= maxiter
        copy!(B_buf, B)
        mul!(B_buf, C, F0, 1.0, 1.0)
        lu_t = LU(factorize!(linws, B_buf)...)
        copy!(A_buf, A)
        copy!(A_orig, A)
        copy!(B_orig, B)
        ldiv!(lu_t, A_buf)
        lmul!(-1.0, A_buf)
        F1 = A_buf
        copy!(C_buf, C)
        copy!(B_buf, B)
        mul!(B_buf, A, S0, 1.0, 1.0)
        # S1 = -(A * S0 + B) \ C
        lu_t = LU(factorize!(linws, B_buf)...)
        ldiv!(lu_t, C_buf)
        lmul!(-1.0, C_buf)
        S1 = C_buf
        mul!(A_orig, B_orig, F1, 1.0, 1.0)
        mul!(C_orig, F1, F1, 1.0, 0.0)
        mul!(A_orig, C, C_orig, 1.0, 1.0)
        
        # error = maximum(C * F1 * F1  + B * F1 + A)
        error = maximum(A_orig)
        
        F0 .= F1
        S0 .= S1
        
        iter += 1
        
    end
    # fill!(v, 0.0)
    # v[2] = 1
    # XP = power_eigenvalue(v, F0, v_buf,1e-2)
    # fill!(v, 0.0)
    # v[1] = 1
    # XS = power_eigenvalue(v, S0, v_buf, 1e-2)
    
    # if iter == maxiter
    #     outmessage = "Convergence Failed. Max Iterations Reached. Error: $error"
    # elseif XP > 1.0
    #     outmessage = "No Stable Solution Exists!"
    # elseif XS > 1.0
    #     outmessage = "Multiple Solutions Exist!"
    # else
    #     outmessage = "Convergence Successful!"
    # end
    
    # Q = -(C * F0 + B) \ E


    return F0

    
end

