module ADAGRAD_Solver

using Random
using LinearAlgebra
using ForwardDiff

export Solver

# Create struct solver to approach the problem
mutable struct Solver
    n :: Int
    iteration :: Int
    λ :: Array{Float64}
    K :: Int
    I_K :: Vector{Array{Int64}}
    x :: Array{Float64}
    grads :: Array{Float64}
    G_t :: Diagonal{Float64, Vector{Float64}}
    s_t :: Array{Float64}
    avg_gradient :: Array{Float64}
    Q :: Matrix{Float64}
    q :: Array{Float64}
    η :: Float64
    δ :: Float64
    max_iter :: Int
    ϵ :: Float64
    num_iterations :: Vector{Float64}
    relaxation_values :: Vector{Float64}
    x_values :: Array{Float64}
    λ_values :: Array{Float64}
    λ_distances :: Vector{Float64}
    update_formula :: Int
    Full_mat :: Matrix{Float64}
    F :: LU
    A :: Array{Int64}
end


#=
    Compute the function value of the Lagrangian relaxation given the current value of λ
    and x
        
        L(x,λ) = x' Q x + q' x - λ' x

=#
function lagrangian_relaxation(solver, previous_x, previous_λ)
    return (previous_x' * solver.Q * previous_x) .+ (solver.q' * previous_x) .- (previous_λ' * previous_x)
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
function solve_lagrangian_relaxation(solver)

    # Create vector b = [λ_{t-1} - q, b]
    o = ones((solver.K,1))

    diff = solver.λ - solver.q

    b = vcat(diff, o)

    # Find vector c using the Permutation matrix
    c = solver.F.P * b

    dim = solver.n + solver.K

    # Use forward substitution to find vector 
    d = zeros((dim, 1))

    d[1] = c[1] / solver.F.L[1,1]
    for i = 2:+1:dim
        s = sum( solver.F.L[i,j]*d[j] for j=1:i-1 )
        d[i] = ( c[i] - s ) / solver.F.L[i,i]
    end

    # Then use back substitution to find [x μ]
    x_μ = zeros((dim,1))

    x_μ[dim] = d[dim]/solver.F.U[dim, dim]
    for i = dim-1:-1:1
        s = sum( solver.F.U[i,j]*x_μ[j] for j=i+1:dim )
        x_μ[i] = ( d[i] - s ) / solver.F.U[i,i]
    end
    
    return x_μ[1:solver.n] , x_μ[solver.n + 1 : dim]
end


#= 

    Compute one among the three possible update_rule specified in the report.
    
    The first update rule is a general update rule given by:

        λ_t = λ_{t-1} + η \diag(G_t)^{-1/2} g_t

    where G_t is the full outer product of all the stored subgradient

    The second update rule is:

        λ_t = - H_{t-1}^{-1} t η g_t 

    The third one employ the following:

        λ_t = λ_{t-1} - H_{t-1}^{-1} \[ Ψ_t(λ_t) - η g_t \]

    The value of Ψ explode the second term of the latter update_rule. As a consequence 
    the next value of λ becomes bigger and bigger. A minus sign instead constrain the 
    value of λ to be smaller but at the same time reduce also the value of Ψ

=#
function compute_update_rule(solver, H_t, Ψ)
    
    if solver.update_formula == 1

        # Add only the latter subgradient, since summation moves along with 
        last_subgrad = solver.grads[:, end]

        # Add the latter g_t * g_t' component-wise to the matrix G_t
        solver.G_t .+= Diagonal(last_subgrad * last_subgrad')

        pow = -0.5

        # Create a copy for Diagonal operation and exponentiation
        G_t = abs.(solver.G_t)

        # Apply exponentiation
        G_t = G_t^pow

        # Replace all the NaN values with 0.0 to avoid NaN values in the iterates
        replace!(G_t, NaN => 0.0)

        λ = solver.λ + (solver.η * G_t * solver.grads[:,end])
        
    elseif solver.update_formula == 2

        # Sum the latter subgradient found
        solver.avg_gradient .+= solver.grads[:, end]

        # Average the row sum of the gradient based on the current iteration in a new variable
        avg_gradient_copy = solver.avg_gradient ./ solver.iteration

        λ = solver.iteration * solver.η * (- H_t^(-1) * avg_gradient_copy)

    else

        val = Ψ .- (solver.η * solver.grads[:,end])

        update_part = H_t^(-1) * val

        # Minus needed: constrain λ to be smaller, otherwise the Ψ term explode the value of λ
        λ = solver.λ - update_part

    end

    λ = max.(0, λ)

    # println("Updated λ")
    # display(λ)
    # print("\n")

    return λ

end


#=
    Check the other stopping condition, i.e. when
        \| λ_t - λ_{t-1} \|_2 ≤ ϵ 
    whenever this condition is met we have reached an optimal value of multipliers λ,
    hence we met complementary slackness and we can stop
=#
function check_λ_norm(solver, current_λ, previous_λ)
    
    res = current_λ .- previous_λ

    distance = norm(res)

    # println("Distance between λ's")
    # display(distance)
    # print("\n")

    push!(solver.λ_distances, distance)

    if distance <= solver.ϵ
        # We should exit the loop
        println("Distance between λ's")
        display(distance)
        print("\n")
        return true
    end

    return false

end

#= 
    Compute the gradient of the lagrangian relaxation of the problem. Given the value of x_{t-1}, the 
    subgradient (which coincide with the gradient in this case) is the slope of the hyperplane defined 
    by the known vector x_{t-1} and the variables λ_{t-1}
=#
function get_grad(solver)

    L(var) = solver.x' * solver.Q * solver.x .+ solver.q' * solver.x .- var' * solver.x

    gradient(val) = ForwardDiff.jacobian(var -> L(var), val)

    subgradient = gradient(solver.λ)'

    return subgradient

end


#=
    Loop function which implements customized ADAGRAD algorithm. The code is the equivalent of Algorithm 3 
    presented in the report.
    Takes as parameters:
        @param solver: struct solver containing all the required data structures
        @param update_rule: an integer specifying what update rule to use for the λ's
=#  
function my_ADAGRAD(solver)

    solver.iteration = 0

    while solver.iteration < solver.max_iter

        solver.iteration += 1

        push!(solver.num_iterations, solver.iteration)

        solver.η = 1 / solver.iteration

        previous_λ = solver.λ

        #= 
        Compute subgradient of ϕ (check report for derivation)
        The subgradient is given by the derivative of the Lagrangian function w.r.t. λ
            
            ∇_λ (x_{star}) ( x_{star} * Q * x_{star} + q * x_{star} - λ * x_{star} )
        
        where the given x_{star} is computed at the previous step. As a consequence the L function
        is given by  
        
            ∇_λ (x_{star}) ( λ * x_{star} + constant )
            
        which is always differentiable since it is an hyperplane
        =#
        subgrad = get_grad(solver)

        # Store subgradient in matrix
        solver.grads = [solver.grads subgrad]
        
        # Solution of lagrangian_relaxation of problem 2.4 (see report)
        # Accumulate the squared summation into solver.s_t structure
        solver.s_t .+= (subgrad.^2)

        # Create a copy of solver.s_t (avoid modifying the original one)
        s_t = solver.s_t 

        # Vectorize s_t applying the √ to each element
        s_t = vec(sqrt.(abs.(s_t)))

        # Create diagonal matrix H_t
        # Create diagonal matrix starting from s_t 
        mat = Diagonal(s_t)

        # Construct Identity matrix
        Iden = Diagonal(ones(solver.n,solver.n))

        δ_Id = solver.δ .* Iden

        # Sum them (hessian approximation)
        H_t = δ_Id + mat

        #= 
        Proximal term computation: dot(x,A,y) faster way of computing 
            ⟨ x, A*y ⟩
        the result A*y is not stored and this allow for a memory saving
        =#   
        Ψ = 0.5 * dot(previous_λ, H_t, previous_λ)
       
        # In the general case, the most appropriate factorization in this particular case is the 
        # LU factorization with partial pivoting, which can be applied to every rectangular matrix 
        # is stable since only in the worst case suffer from numerical instability
        solver.x, μ = solve_lagrangian_relaxation(solver)

        # println("x value:")
        # display(solver.x)
        # print("\n")
    
        #=
        Compute the update rule for the lagrangian multipliers λ: can use 
        one among the three showed, then soft threshold the result
        =#
        solver.λ = compute_update_rule(solver, H_t, Ψ)

        # Compute Lagrangian function value
        L_val = lagrangian_relaxation(solver, solver.x, solver.λ)

        # println("Lagrangian relaxation value: $(L_val[1])")
        # print("\n")

        # Storing current relaxation value
        push!(solver.relaxation_values, L_val[1])

        # Storing x_{solver.iteration}
        solver.x_values = [solver.x_values solver.x]

        # Storing λ_{solver.iteration}
        solver.λ_values = [solver.λ_values solver.λ]

        if check_λ_norm(solver, solver.λ, previous_λ)
            println("Optimal λ found")
            break
        end

    end

    return solver

end


end