module ADAGRAD_Solver

using Random
using LinearAlgebra
using Printf
using DataFrames
using CSV

export Solver

#= 
Create struct solver to approach the problem:
    n::Int                                      Identify the problem dimension
    iteration::Int                              Keep track of the current iteration
    λ::Array{Float64}                           Store the current value of the lagrangian multipliers      
    K::Int                                      Identify the number of simplices
    I_K::Vector{Array{Int64}}                   Store the indexes of the corresponding simplices
    x::Array{Float64}                           Store the current value of the lagrangian primal iterates
    grads::Array{Float64}                       Keep track of the subgradient at each iteration 
    G_t::Diagonal{Float64, Vector{Float64}}     Cumulative sum of the outer product of the gradients, keep in Diagonal way to save memory
    s_t::Array{Float64}                         Keep track of the solution of the problem 2.4 (see report)
    avg_gradient::Array{Float64}                Keep track of the average of the subgradients
    d_i::Array{Float64}                         Keep track of the current deflected direction
    deflection::Bool                            Whether to use deflection or not 
    Q::Matrix{Float64}                          Q matrix of the function problem
    q::Array{Float64}                           q vector of the function problem
    η::Float64                                  Stepsize modified at each iteration
    δ::Float64                                  Random quantity used to compute H_t matrix
    max_iter::Int                               Maximum number of iterations allowed
    ε::Float64                                  Tolerance on the norm of the λ iterates
    τ::Float64                                  Tolerance on the dual gap 
    best_lagrangian::Float64                    Keep track of the best lagrangian value found
    best_iteration::Int64                       Keep track of the best iteration
    best_x::Array{Float64}                      Keep track of the best value of x found
    best_λ::Array{Float64}                      Keep track of the best value of λ found 
    num_iterations::Vector{Float64}             Store each iteration
    relaxation_values::Vector{Float64}          Store each lagrangian evaluation 
    x_values::Array{Float64}                    Store every value of x
    λ_values::Array{Float64}                    Store every value of λ
    λ_distances::Array{Float64}                 Store each distance between the current λ and the previous one 
    x_distances::Array{Float64}                 Store each distance between the current x and the previous one  
    timings::Vector{Float64}                    Store timing execution for each iteration
    gaps::Vector{Float64}                       Store dual gap found at each iteration
    update_formula::Int                         Update rule to be used  
    Full_mat::Matrix{Float64}                   Save the full matrix to solve the lagrangian relaxation
    F::Any                                      Save the factorization of Full_mat
    A::Array{Int64}                             Save the constraint matrix A 
    Off_the_shelf_primal::Float64               f(x*) computed with an off-the-shelf solver    
=#
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
    d_i :: Array{Float64}
    deflection :: Bool
    Q :: Matrix{Float64}
    q :: Array{Float64}
    η :: Float64
    δ :: Float64
    max_iter :: Int
    ε :: Float64
    τ :: Float64
    best_lagrangian :: Float64
    best_iteration :: Int64
    best_x :: Array{Float64}
    best_λ :: Array{Float64}
    num_iterations :: Vector{Float64}
    relaxation_values :: Vector{Float64}
    x_values :: Array{Float64}
    λ_values :: Array{Float64}
    λ_distances :: Vector{Float64}
    x_distances :: Vector{Float64}
    timings :: Vector{Float64}
    gaps :: Vector{Float64}
    update_formula :: Int
    Full_mat :: Matrix{Float64}
    F :: Any
    A :: Array{Int64}
    Off_the_shelf_primal :: Float64
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

    Compute one among the three possible update_rule specified in the report.
    
    The first update rule is a general update rule given by:

        λ_t = λ_{t-1} + η \diag(G_t)^{-1/2} g_t

    where G_t is the full outer product of all the stored subgradient

    The second update rule is:

        λ_t = - H_{t-1}^{-1} t η g_t 

    The third one employ the following:

        λ_t = λ_{t-1} - η H_{t-1}^{-1} g_t

=#
# PROVARE CON -
function compute_update_rule(solver, H_t)
    
    if solver.update_formula == 1

        # Add only the latter subgradient, since summation moves along with 
        last_subgrad = solver.grads[:, end]

        # Add the latter g_t * g_t' component-wise to the matrix G_t
        solver.G_t .+= Diagonal(last_subgrad * last_subgrad')

        pow = -0.5

        # Create a copy for Diagonal operation and exponentiation (apply abs for possible errors with negative values)
        G_t = abs.(solver.G_t)

        # Apply exponentiation
        G_t = G_t^pow

        # Replace all the NaN values with 0.0 to avoid NaN values in the iterates
        replace!(G_t, NaN => 0.0)

        if solver.deflection
            
            λ = solver.λ + (solver.η * G_t * solver.d_i)
        
        else

            λ = solver.λ + (solver.η * G_t * solver.grads[:,end])
        
        end

    elseif solver.update_formula == 2

        # Sum the latter subgradient found
        solver.avg_gradient .+= solver.grads[:, end]

        # Average the row sum of the gradient based on the current iteration in a new variable
        avg_gradient_copy = solver.avg_gradient ./ solver.iteration

        λ = solver.iteration * solver.η * (- H_t^(-1) * avg_gradient_copy)

    else

        if solver.deflection
    
            update_part = solver.η * H_t^(-1) * solver.d_i
    
        else

            update_part = solver.η * H_t^(-1) * solver.grads[:,end]

        end
        # Plus needed: constrain λ to be bigger, we are maximizing the dual function
        λ = solver.λ - update_part

    end

    λ = max.(0, λ)

    return λ

end


#=
    Check the other stopping condition, i.e. when
        ∥ λ_t - λ_{t-1} ∥_2 ≤ ε 
    whenever this condition is met we have reached an optimal value of multipliers λ,
    hence we met complementary slackness and we can stop
=#
function check_λ_norm(solver, current_λ, previous_λ)
    
    res = current_λ .- previous_λ

    distance = norm(res)

    push!(solver.λ_distances, distance)

    if distance <= solver.ε
        # We should exit the loop
        # Log result of the last iteration
        @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%s \t\t%1.5e \n" solver.iteration solver.timings[end] solver.relaxation_values[end] solver.x_distances[end] "OPT" solver.gaps[end]

        println("\nOptimal distance between λ's found:")
        display(distance)
        print("\n")
        return true
    end

    return false

end


#=
    Compute the subgradient of ϕ() at the point λ_{t-1}. The subgradient is taken by computing the left limit 
    and the right limit and then choosing the maximum norm of them 

        s = argmax { ∥ s ∥ : s ∈ ∂ϕ(λ_{t-1}) }

    where the set of subgradient is the interval [a, b] where 

        a = lim λ -> λ_{t-1}^-  ( ϕ(λ) - ϕ(λ_{t-1}) ) / (λ - λ_{t-1})

        b = lim λ -> λ_{t-1}^+  ( ϕ(λ) - ϕ(λ_{t-1}) ) / (λ - λ_{t-1})
=#
function get_subgrad(solver)

    # First create the values λ_{t-1}^- and λ_{t-1}^+ with an ϵ > 0
    ϵ = 1e-2

    λ_minus = solver.λ .- ϵ
    λ_plus = solver.λ .+ ϵ

    # Compute the value of a
    a = lagrangian_relaxation(solver, solver.x, λ_minus) - lagrangian_relaxation(solver, solver.x, solver.λ)
    a = a ./ (λ_minus - solver.λ)

    # Compute value of b
    b = lagrangian_relaxation(solver, solver.x, λ_plus) - lagrangian_relaxation(solver, solver.x, solver.λ)
    b = b ./ (λ_plus - solver.λ)

    # If norm(a-b) ≈ 0, then the gradient exist
    difference = norm(a-b)

    if difference <= 1e-12
        # The gradient exists and coincide with the normal derivation of ϕ(λ_{t-1})
        return - solver.x
    end

    # Otherwise compute the maximum norm between a and b
    a_norm = norm(a)
    b_norm = norm(b)

    max_norm = max(a_norm, b_norm)

    if max_norm == a_norm
        return a
    else 
        return b
    end

end


#=
    Compute 

        ∥ x_t - x_{t-1} ∥_2
    
    for the sake of log and visualization
=#
function x_norm(previous_x, current_x)

    res = current_x .- previous_x

    distance = norm(res)

    return distance

end


#=
    Compute the optimal γ resulting from the solution of the problem
    
        γ = argmin { ∥ γ g^i + (1 - γ) d^{i-1} ∥^2 }
=#
function compute_gamma(solver, subgrad, previous_d)

    if solver.iteration == 1
        
        γ = ones((solver.n,1))
    else 
    
        γ = ( previous_d.^(2) .- subgrad ) ./ ( subgrad .- previous_d ).^(2)
    
    end

    return γ

end


#=
    Loop function which implements customized ADAGRAD algorithm. The code is the equivalent of Algorithm 3 
    presented in the report.
=#  
function my_ADAGRAD(solver)

    # Log result of each iteration
    print("Iteration\tTime\t\tL value\t\tx_norm\t\tλ_norm\t\tcurrent gap\n\n")

    # Prepare a dataframe to store the values
    df = DataFrame( Iteration = Int[],
                    Time = Float64[],
                    LagrangianValue = Float64[],
                    x_norm_residual = Float64[],
                    λ_norm_residual = Any[],
                    Dual_gap = Any[] )

    # Set the first optimal values
    solver.iteration = 0

    L_val = lagrangian_relaxation(solver, solver.x, solver.λ)[1]

    current_gap = solver.Off_the_shelf_primal - L_val

    if current_gap > 0 && L_val > solver.best_lagrangian
        solver.best_lagrangian = L_val
        solver.best_iteration = 0
        solver.best_x = solver.x
        solver.best_λ = solver.λ
    end

    @printf "%d\t\t%.8f \t%.8f \t%.8f \t%.8f \t%.8f \n" solver.iteration 0.00000000 L_val norm(solver.x) norm(solver.λ) current_gap

    while solver.iteration < solver.max_iter

        # Set starting time
        starting_time = time()

        # Increment iteration
        solver.iteration += 1

        # Save iteration
        push!(solver.num_iterations, solver.iteration)

        # Modify η in DSS way
        solver.η = 1 / solver.iteration

        # Assign previous_λ
        previous_λ = solver.λ

        # Assign previous_x
        previous_x = solver.x

        #= 
        Compute subgradient of ϕ (check report for derivation)
        The subgradient is given by the derivative of the Lagrangian function w.r.t. λ
            
            ∇_λ (x_{star}) ( x_{star} * Q * x_{star} + q * x_{star} - λ * x_{star} )
        
        where the given x_{star} is computed at the previous step. As a consequence the L function
        is given by  
        
            ∇_λ (x_{star}) ( - λ * x_{star} )
            
        which is always differentiable since it is an hyperplane
        =#
        subgrad = get_subgrad(solver)

        if solver.deflection

            previous_d = isempty(solver.grads) ? subgrad : solver.d_i

            γ = compute_gamma(solver, subgrad, previous_d)

            solver.d_i = γ .* subgrad .+ (ones((solver.n,1)) .- γ) .* previous_d

            replace!(solver.d_i, NaN => 0.0)

        end

        # Store subgradient in matrix
        # solver.grads = [solver.grads subgrad]
        solver.deflection ? solver.grads = [solver.grads solver.d_i] : solver.grads = [solver.grads subgrad]
        
        # Solution of lagrangian_relaxation of problem 2.4 (see report)
        # Accumulate the squared summation into solver.s_t structure
        solver.deflection ? solver.s_t .+= (solver.d_i.^2) : solver.s_t .+= (subgrad.^2)

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
       
        # Create vector b = [λ_{t-1} - q, b]
        o = ones((solver.K,1))

        diff = solver.λ - solver.q

        b = vcat(diff, o)

        #= 
            Solve linear system efficiently using \ of Julia: will automatically use techniques like 
            backward and forward substitution to optimize the computation    
        =# 
        x_μ = solver.F \ b

        solver.x, μ = x_μ[1:solver.n], x_μ[solver.n+1 : solver.n + solver.K]

        #=
        Compute the update rule for the lagrangian multipliers λ: can use 
        one among the three showed, then soft threshold the result
        =#
        solver.λ = compute_update_rule(solver, H_t)
        
        # Compute Lagrangian function value
        L_val = lagrangian_relaxation(solver, solver.x, solver.λ)

        # Storing current relaxation value
        push!(solver.relaxation_values, L_val[1])

        # Storing x_{solver.iteration}
        solver.x_values = [solver.x_values solver.x]

        # Storing λ_{solver.iteration}
        solver.λ_values = [solver.λ_values solver.λ]

        # Compute \| x_t - x_{t-1} \|_2 and save it
        push!(solver.x_distances, x_norm(previous_x, solver.x))

        # Store timing result of this iteration
        finish_time = time()    

        # Compute timing needed for this iteration
        time_step = finish_time - starting_time
            
        # Save time step
        push!(solver.timings, time_step)

        # Compute current dual_gap
        current_gap = solver.Off_the_shelf_primal - solver.relaxation_values[end]

        if L_val[1] > solver.best_lagrangian && current_gap > 0
            solver.best_lagrangian = L_val[1]
            solver.best_iteration = solver.iteration
            solver.best_x = solver.x
            solver.best_λ = solver.λ
        end

        # Store the current gap
        push!(solver.gaps, current_gap)

        if check_λ_norm(solver, solver.λ, previous_λ)
            # Add last row
            push!(df, [ solver.iteration, solver.timings[end], solver.relaxation_values[end], solver.x_distances[end], "OPT", solver.gaps[end] ])
            break
        end

        if current_gap > 0 && current_gap <= solver.τ
            println("Found optimal dual gap")
            # Log result of the current iteration
            @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%s \n" solver.iteration solver.timings[end] solver.relaxation_values[end] solver.x_distances[end] solver.λ_distances[end] "OPT"
            push!(df, [solver.iteration, solver.timings[end], solver.relaxation_values[end], solver.x_distances[end], solver.λ_distances[end], "OPT" ])
            break   
        end

        if (solver.iteration == 1) || (solver.iteration % 10 == 0)
            # Log result of the current iteration
            @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" solver.iteration solver.timings[end] solver.relaxation_values[end] solver.x_distances[end] solver.λ_distances[end] current_gap
        end

        if isnan( solver.relaxation_values[end] ) || isnan( solver.x_distances[end] ) || isnan( solver.λ_distances[end] ) || isnan( current_gap )
            println("Some NaN values detected")
            break
        end
        # Add to DataFrame to save results
        push!(df, [solver.iteration, solver.timings[end], solver.relaxation_values[end], solver.x_distances[end], solver.λ_distances[end], current_gap ])

    end

    # Log total time and iterations
    print("\n")
    print("Iterations: $(solver.iteration)\tTotal time: $(round(sum(solver.timings), digits=6))\n")

    # Save results in CSV file
    CSV.write("logs/results_n=$(solver.n)_K=$(solver.K)_update=$(solver.update_formula)_defl=$(solver.deflection).csv", df)

    return solver

end


end