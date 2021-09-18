module Utils

#= 
Create matrix of KKT conditions
    Q A^T
    A 0
to solve the lagrangian_relaxation
=#
function construct_full_matrix(Q, A, K)
    Full_mat = vcat(Q, A)
    
    zero_mat = zeros((K, K))

    Right_side = vcat(A', zero_mat)

    Full_mat = hcat(Full_mat, Right_side)

    return Full_mat
end


# Helper function to construct the matrix A
# See Algorithm 2 of the report
function construct_A(K, n, I_K)
    A = Array{Int64}(undef, K, n)
    for k in 1:K
        a_k = zeros(Int64, n)
        for i in 1:n
            if i in I_K[k]
                a_k[i] = 1
            end 
        end
        A[k,:] = a_k
    end 
    return A
end


end