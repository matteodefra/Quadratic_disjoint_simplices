module Utils

using LinearAlgebra
using SparseArrays

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

    return SparseMatrixCSC(Symmetric(Full_mat))
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
    return SparseMatrixCSC(A)
end


#
function primal_function(Q, q, x)
    return (x ⋅ (Q * x)) + q ⋅ x
end

# # Compute dual function value
function compute_dualvalue(Q, q, x, λ)
    return (x ⋅ (Q * x)) + (q ⋅ x) - (λ ⋅ x)
end

#=
    Solve the problem of 

        x_t = \argmin_{x ∈ X} \{ x^T * Q * x + q * x - λ_{t-1} * x \}
    
    where X is the constraint set of the disjoint simplices (eliding non negativity 
    constraint, since it's included in the relaxation).
    Being only linear constraint, this problem can be easily solved in O((n+K)^2) time,
    using the LU factorization of the matrix through forward and back substitution

    First of all multiply the permutation matrix for the last term 
            
        c = P * [ λ_{t-1} - q, b ]

    Then use forward substitution to solve the linear system and find the vector d

        L * d = c 

    Finally use back substitution to solve the linear system 

        U * [x μ] = d 

    and obtain the result.
    Returns a tuple with the value x and μ
=#
# function solve_lagrangian_relaxation(solver)

#     # Create vector b = [λ_{t-1} - q, b]
#     o = ones((solver.K,1))

#     diff = solver.λ - solver.q

#     b = vcat(diff, o)

#     # Find vector c using the Permutation matrix
#     c = solver.F.P * b

#     dim = solver.n + solver.K

#     # Use forward substitution to find vector 
#     d = zeros((dim, 1))

#     d[1] = c[1] / solver.F.L[1,1]
#     for i = 2:+1:dim
#         s = sum( solver.F.L[i,j]*d[j] for j=1:i-1 )
#         d[i] = ( c[i] - s ) / solver.F.L[i,i]
#     end

#     # Then use back substitution to find [x μ]
#     x_μ = zeros((dim,1))

#     x_μ[dim] = d[dim]/solver.F.U[dim, dim]
#     for i = dim-1:-1:1
#         s = sum( solver.F.U[i,j]*x_μ[j] for j=i+1:dim )
#         x_μ[i] = ( d[i] - s ) / solver.F.U[i,i]
#     end
    
#     return x_μ[1:solver.n] , x_μ[solver.n + 1 : dim]
# end



end