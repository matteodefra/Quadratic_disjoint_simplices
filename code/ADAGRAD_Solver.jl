module ADAGRAD_Solver

using Random
using LinearAlgebra
using ForwardDiff
using Printf
using DataFrames
using CSV

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

        λ_t = λ_{t-1} + η H_{t-1}^{-1} g_t

    The value of Ψ explode the second term of the latter update_rule. As a consequence 
    the next value of λ becomes bigger and bigger. A minus sign instead constrain the 
    value of λ to be smaller but at the same time reduce also the value of Ψ

=#
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

        λ = solver.λ + (solver.η * G_t * solver.grads[:,end])
        
    elseif solver.update_formula == 2

        # Sum the latter subgradient found
        solver.avg_gradient .+= solver.grads[:, end]

        # Average the row sum of the gradient based on the current iteration in a new variable
        avg_gradient_copy = solver.avg_gradient ./ solver.iteration

        λ = solver.iteration * solver.η * (- H_t^(-1) * avg_gradient_copy)

    else

        update_part = solver.η * H_t^(-1) * solver.grads[:,end]

        # Plus needed: constrain λ to be bigger, we are maximizing the dual function
        λ = solver.λ + update_part

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
        # Log result of the last iteration
        @printf "%d\t\t%.5f \t%.5f \t%.5f \t%.5f \t%.5f \n" solver.iteration solver.timings[end] solver.relaxation_values[end] solver.x_distances[end] solver.λ_distances[end] (solver.Off_the_shelf_primal - solver.relaxation_values[end])

        println("\nOptimal distance between λ's found:")
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
    Compute 

        \| x_t - x_{t-1} \|_2
    
    for the sake of log and visualization
=#
function x_norm(previous_x, current_x)

    res = current_x .- previous_x

    distance = norm(res)

    # println("Distance between λ's")
    # display(distance)
    # print("\n")

    return distance

end

#=
    Loop function which implements customized ADAGRAD algorithm. The code is the equivalent of Algorithm 3 
    presented in the report.
    Takes as parameters:
        @param solver: struct solver containing all the required data structures
        @param update_rule: an integer specifying what update rule to use for the λ's
=#  
function my_ADAGRAD(solver)

    # Log result of each iteration
    print("Iteration\tTime\t\tL value\t\tx_norm\t\tλ_norm\t\tcurrent gap\n\n")

    df = DataFrame( Iteration = Int[],
                    Time = Float64[],
                    LagrangianValue = Float64[],
                    x_norm_residual = Float64[],
                    λ_norm_residual = Any[],
                    Dual_gap = Float64[] )

    solver.iteration = 0

    while solver.iteration < solver.max_iter

        starting_time = time()

        solver.iteration += 1

        push!(solver.num_iterations, solver.iteration)

        solver.η = 1 / solver.iteration

        previous_λ = solver.λ

        previous_x = solver.x

        #= 
        Compute subgradient of ϕ (check report for derivation)
        The subgradient is given by the derivative of the Lagrangian function w.r.t. λ
            
            ∇_λ (x_{star}) ( x_{star} * Q * x_{star} + q * x_{star} - λ * x_{star} )
        
        where the given x_{star} is computed at the previous step. As a consequence the L function
        is given by  
        
            ∇_λ (x_{star}) ( - λ * x_{star} + constant )
            
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

        # println("x value:")
        # display(solver.x)
        # print("\n")
    
        #=
        Compute the update rule for the lagrangian multipliers λ: can use 
        one among the three showed, then soft threshold the result
        =#
        solver.λ = compute_update_rule(solver, H_t)

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

        # Compute \| x_t - x_{t-1} \|_2 and save it
        push!(solver.x_distances, x_norm(previous_x, solver.x))

        # Store timing result of this iteration
        finish_time = time()    

        time_step = finish_time - starting_time
         
        push!(solver.timings, time_step)

        current_gap = solver.Off_the_shelf_primal - solver.relaxation_values[end]

        # Store the current gap
        push!(solver.gaps, current_gap)

        if check_λ_norm(solver, solver.λ, previous_λ)
            # Add last row
            push!(df, [solver.iteration, solver.timings[end], solver.relaxation_values[end], solver.x_distances[end], "OPT", (solver.Off_the_shelf_primal - solver.relaxation_values[end]) ])
            break
        end

        # FIND A FIX FOR DUALITY GAP
        # if current_gap < 0
        #     println("Found negative dual gap, fix it")
        #     # Log result of the current iteration
        #     @printf "%d\t\t%.5f \t%.5f \t%.5f \t%.5f \t%.5f \n" solver.iteration solver.timings[end] solver.relaxation_values[end] solver.x_distances[end] solver.λ_distances[end] (solver.Off_the_shelf_primal - solver.relaxation_values[end])
        #     push!(df, [solver.iteration, solver.timings[end], solver.relaxation_values[end], solver.x_distances[end], solver.λ_distances[end], (solver.Off_the_shelf_primal - solver.relaxation_values[end]) ])
        #     break      
        # end

        # Log result of the current iteration
        @printf "%d\t\t%.5f \t%.5f \t%.5f \t%.5f \t%.5f \n" solver.iteration solver.timings[end] solver.relaxation_values[end] solver.x_distances[end] solver.λ_distances[end] (solver.Off_the_shelf_primal - solver.relaxation_values[end])
       
        # Add to DataFrame to save results
        push!(df, [solver.iteration, solver.timings[end], solver.relaxation_values[end], solver.x_distances[end], solver.λ_distances[end], (solver.Off_the_shelf_primal - solver.relaxation_values[end]) ])

    end

    print("\n")
    print("Iterations: $(solver.iteration)\tTotal time: $(round(sum(solver.timings), digits=6))\n")

    CSV.write("logs/results_n=$(solver.n)_K=$(solver.K)_update=$(solver.update_formula).csv", df)

    return solver

end


end