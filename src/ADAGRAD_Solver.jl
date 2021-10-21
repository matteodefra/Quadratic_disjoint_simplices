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
    μ::Array{Float64}                           Current value of lagrangian multipliers
    K::Int                                      Identify the number of simplices
    I_K::Vector{Array{Int64}}                   Store the indexes of the corresponding simplices
    x::Array{Float64}                           Store the current value of the lagrangian primal iterates
    grads::Array{Float64}                       Keep track of the subgradient at each iteration 
    G_t::Array{Float64}                         Cumulative sum of the outer product of the gradients, keep only diagonal
    s_t::Array{Float64}                         Keep track of the solution of the problem 2.4 (see report)
    H_t::Array{Float64}                         Keep the diagonal δ*ones() vector in the struct, and add the s_t iteratively
    d_i::Array{Float64}                         Keep track of the current deflected direction
    deflection::Bool                            Whether to use deflection or not 
    Q::Matrix{Float64}                          Q matrix of the function problem
    q::Array{Float64}                           q vector of the function problem
    η::Float64                                  Stepsize modified at each iteration
    δ::Float64                                  Random quantity used to compute H_t matrix
    max_iter::Int                               Maximum number of iterations allowed
    ε::Float64                                  Tolerance on the norm of the λ iterates
    τ::Float64                                  Tolerance on the dual gap 
    best_dual::Float64                          Keep track of the best dual value found
    best_iteration::Int64                       Keep track of the best iteration
    best_x::Array{Float64}                      Keep track of the best value of x found
    best_λ::Array{Float64}                      Keep track of the best value of λ found 
    num_iterations::Vector{Float64}             Store each iteration
    dual_values::Vector{Float64}                Store each lagrangian evaluation 
    λ_distances::Array{Float64}                 Store each distance between the current λ and the previous one 
    x_distances::Array{Float64}                 Store each distance between the current x and the previous one  
    timings::Vector{Float64}                    Store timing execution for each iteration
    gaps::Vector{Float64}                       Store dual gap found at each iteration
    update_formula::Int                         Update rule to be used  
    stepsize_choice::Int                        Stepsize choice to use
    F::Any                                      Save the factorization of Full_mat
    A::Array{Int64}                             Save the constraint matrix A 
    Off_the_shelf_primal::Float64               f(x*) computed with an off-the-shelf solver    
=#
mutable struct Solver
    n :: Int
    iteration :: Int
    λ :: Array{Float64}
    μ :: Array{Float64}
    K :: Int
    I_K :: Vector{Array{Int64}}
    x :: Array{Float64}
    grads :: Array{Float64}
    G_t :: Array{Float64}
    s_t :: Array{Float64}
    H_t :: Array{Float64}
    d_i :: Array{Float64}
    deflection :: Bool
    Q :: Matrix{Float64}
    q :: Array{Float64}
    η :: Float64
    δ :: Float64
    max_iter :: Int
    ε :: Float64
    τ :: Float64
    best_dual :: Float64
    best_iteration :: Int64
    best_x :: Array{Float64}
    best_λ :: Array{Float64}
    num_iterations :: Vector{Float64}
    dual_values :: Vector{Float64}
    λ_distances :: Vector{Float64}
    x_distances :: Vector{Float64}
    timings :: Vector{Float64}
    gaps :: Vector{Float64}
    update_formula :: Int
    stepsize_choice :: Int
    F :: Any
    A :: Array{Int64}
    Off_the_shelf_primal :: Float64
end

#=
    Compute the dual function value given the minimum x found and the value of λ
        
        ϕ(λ) = x' Q x + q' x + λ' x 

=#
function dual_function(solver, previous_x, previous_λ)
    return BLAS.dot(solver.n, previous_x, 1, BLAS.symv('U' ,solver.Q, previous_x),1) + BLAS.dot(solver.n, solver.q, 1, previous_x, 1) + BLAS.dot(solver.n, previous_λ, 1, previous_x, 1)
    # return (previous_x ⋅ (solver.Q * previous_x)) + (solver.q ⋅ previous_x) + (previous_λ ⋅ previous_x) + ( solver.μ ⋅ ( (solver.A * previous_x) - ones((solver.K,1)) ) )
end

#=
    Compute the prima; function value given the minimum x found
        
        f(x) = x' Q x + q' x

=#
function primal_function(solver, feasible_x) 
    return BLAS.dot(solver.n, feasible_x, 1, BLAS.symv('U' ,solver.Q, feasible_x),1) + BLAS.dot(solver.n, solver.q, 1, feasible_x, 1) 
    # return (feasible_x ⋅ (solver.Q * feasible_x)) + (solver.q ⋅ feasible_x)
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
function compute_update_rule(solver, H_t)

    λ = Array{Float64,1}(undef, solver.n)

    G_t = Array{Float64,1}(undef, solver.n)

    if solver.update_formula == 1

        # Add only the latter subgradient, since summation moves along with 
        last_subgrad = solver.grads[:, end]
        
        # Add the latter diagonal of g_t * g_t' component-wise to the vector G_t
        full_prod = map(*, last_subgrad, last_subgrad)

        if any( isnan, full_prod )
            println("full_prod is nan!")
        end

        solver.G_t .+= full_prod

        G_t = solver.G_t 

        if any( isnan, G_t )
            println("G_t is nan!")
        end

        if any(x -> x<0, G_t) 
            println("G_t is negative")
        end

        # Apply exponentiation and square root
        G_t .= map(sqrt, G_t)

        λ .= map(/, solver.η, G_t)
        
        λ .= solver.deflection ? map(*, λ, solver.d_i) : map(*, λ, last_subgrad)

        axpy!(1, solver.λ, λ)

    elseif solver.update_formula == 2

        # Average the row sum of the gradient based on the current iteration in a new variable
        avg_gradient_copy = map(/, solver.s_t, solver.iteration)

        scalar = - solver.η * solver.iteration

        BLAS.scal!(solver.n, scalar, λ, 1)

        λ .= map(/, λ, H_t)

        λ .= map(*, λ, avg_gradient_copy)

    elseif solver.update_formula == 3 

        λ .= map(/, solver.η, H_t)

        λ .= solver.deflection ? map(*, λ, solver.d_i) : map(*, λ, solver.grads[:,end])

        axpy!(1, solver.λ, λ)

    else

        BLAS.scal!(solver.n, Float64(solver.η), λ, 1)
        # part = solver.grads[:, end]
        λ .= map(*, λ, solver.grads[:,end])

        axpy!(1, solver.λ, λ)
        # λ = solver.λ + update_part

    end

    λ .= max.(0, λ)

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
        # @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%s \t\t%1.5e \n" solver.iteration solver.timings[end] solver.dual_values[end] solver.x_distances[end] "OPT" solver.gaps[end]

        # println("\nOptimal distance between λ's found:")
        # display(distance)
        # print("\n")
        return true
    end

    return false

end


#=
    Compute the subgradient of -ϕ() at the point λ_{t-1}. The subgradient is taken by computing the left limit 
    and the right limit and then choosing the maximum norm of them 

        s = argmin { ∥ s ∥ : s ∈ ∂ϕ(λ_{t-1}) }

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
    a = dual_function(solver, solver.x, λ_minus) - dual_function(solver, solver.x, solver.λ)
    a = a ./ (λ_minus - solver.λ)

    # Compute value of b
    b = dual_function(solver, solver.x, λ_plus) - dual_function(solver, solver.x, solver.λ)
    b = b ./ (λ_plus - solver.λ)

    # If norm(a-b) ≈ 0, then the gradient exist
    difference = norm(a-b)

    if difference <= 1e-12
        # The gradient exists and coincide with the normal derivation of ϕ(λ_{t-1})
        return solver.x
    end

    # Otherwise compute the maximum norm between a and b
    a_norm = norm(a)
    b_norm = norm(b)

    min_norm = min(a_norm, b_norm)

    if min_norm == a_norm

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



function isincreasing(solver)

    # Check last 20 gaps 
    # latter_minimum = minimum(solver.gaps[1:end-1])

    if solver.iteration - solver.best_iteration < 500#solver.gaps[end] < latter_minimum
        return false
    else
        return true
    end

end


# function make_feasible(solver, mode) x = fill(0.0, solver.n) for set in solver.I_K len = length(set) if len == 1 x[set[1]] = 1.0 else if mode == 1 x[set[1]] = solver.x[set[1]] for i=2:1:len x[set[1]] += solver.x[set[i]] x[set[i]] = 0.0 end else for i in set x[i] = 1 / len end end end end return x end

#=
    Loop function which implements customized ADAGRAD algorithm. The code is the equivalent of Algorithm 3 
    presented in the report.
=#  
function my_ADAGRAD(solver)

    h = 100 # rand()

    β = 1 # or rand()

    α = 1 # or rand()

    # To create vector b = [λ_{t-1} - q, b]
    o = ones((solver.K,1))

    o2 = ones((solver.n,1))

    # H_t allocation
    H_t = Array{Float64,1}(undef, solver.n)

    # s_t allocation
    s_t = Array{Float64,1}(undef, solver.n)

    # gamma allocation
    γ = Array{Float64,1}(undef, solver.n)

    # Log result of each iteration
    print("Iteration\tTime\t\tL value\t\tx_norm\t\tλ_norm\t\tcurrent gap\n\n")

    # Prepare a dataframe to store the values
    df = DataFrame( Iteration = Int[],
                    Time = Float64[],
                    DualValue = Float64[],
                    x_norm_residual = Float64[],
                    λ_norm_residual = Any[],
                    Dual_gap = Any[] )

    # Set the first optimal values
    solver.iteration = 0

    ϕ_λ = dual_function(solver, solver.x, solver.λ)[1]

    current_gap = solver.Off_the_shelf_primal - ϕ_λ

    if current_gap > 0 && ϕ_λ > solver.best_dual
        solver.best_dual = ϕ_λ
        solver.best_iteration = 0
        solver.best_x = solver.x
        solver.best_λ = solver.λ
    end

    @printf "%d\t\t%.8f \t%.8f \t%.8f \t%.8f \t%.8f \n" solver.iteration 0.00000000 ϕ_λ norm(solver.x) norm(solver.λ) current_gap

    while solver.iteration < solver.max_iter

        # Set starting time
        starting_time = time()

        # Increment iteration
        solver.iteration += 1

        # Save iteration
        push!(solver.num_iterations, solver.iteration)

        # Assign previous_λ
        previous_λ = solver.λ

        # Assign previous_x
        previous_x = solver.x

        # Compute subgradient of ϕ (check report for derivation)
        subgrad = get_subgrad(solver)

        if any( isnan, subgrad )
            println("Subgrad is nan!")
        end

        if solver.deflection

            previous_d = isempty(solver.grads) ? subgrad : solver.d_i

            γ = compute_gamma(solver, subgrad, previous_d)

            solver.d_i = map(*, γ, subgrad) .+  (o2 .- γ) .* previous_d

        end

        current_gap < 0 ? subgrad = - subgrad : subgrad = subgrad

        # Store subgradient in matrix
        solver.grads = [solver.grads subgrad]

        # Modify η in different ways
        if solver.stepsize_choice == 0

            solver.η = h

        elseif solver.stepsize_choice == 1

            solver.η = h / norm( solver.grads[:, end] )

        elseif solver.stepsize_choice == 2

            solver.η = α / (β + solver.iteration)

        elseif solver.stepsize_choice == 3

            solver.η = α / sqrt(solver.iteration)

        else 

            solver.η = isempty(solver.dual_values) ? 1 : (solver.Off_the_shelf_primal - solver.dual_values[end]) / norm(solver.grads[:,end])^2

        end
        
        # Solution of dual_function of problem 2.4 (see report)
        # Accumulate the squared summation into solver.s_t structure
        solver.s_t += subgrad

        # Create a copy of solver.s_t (avoid modifying the original one)
        s_t = solver.s_t 

        # Compute s_t
        s_t .= norm.(s_t)

        # Sum them (hessian approximation)
        H_t = solver.H_t

        H_t .+= s_t

        if any( isnan, H_t )
            println("H_t is nan!")
        end

        diff = map(-, solver.λ, solver.q)

        b = vcat(diff, o)

        #= 
            Solve linear system efficiently using \ of Julia: will automatically use techniques like 
            backward and forward substitution to optimize the computation    
        =# 
        x_μ = \(solver.F, b)

        if any( isnan, x_μ )
            println("x_μ is nan!")
        end

        solver.x, solver.μ = x_μ[1:solver.n], x_μ[solver.n+1 : solver.n + solver.K]

        #=
        Compute the update rule for the lagrangian multipliers λ: can use 
        one among the three showed, then soft threshold the result
        =#
        solver.λ = compute_update_rule(solver, H_t)
        
        # Compute Lagrangian function value
        ϕ_λ = dual_function(solver, solver.x, solver.λ)[1]

        if any( isnan, solver.λ )
            println("solver.λ is nan!")
            break
        elseif any( isnan, solver.x )
            println("solver.x is nan!")
            break
        elseif isnan( ϕ_λ ) 
            println("ϕ_λ is nan!")
            break
        end 

        # Storing current relaxation value
        push!(solver.dual_values, ϕ_λ)

        # Compute \| x_t - x_{t-1} \|_2 and save it
        push!(solver.x_distances, x_norm(previous_x, solver.x))

        # Store timing result of this iteration
        finish_time = time()    

        # Compute timing needed for this iteration
        time_step = finish_time - starting_time
            
        # Save time step
        push!(solver.timings, time_step)

        # Compute current dual_gap
        current_gap = solver.Off_the_shelf_primal - solver.dual_values[end]

        if isnan( current_gap )
            println("Some NaN values detected")
            break
        end

        # Update the best solution if conditions are met
        if ϕ_λ > solver.best_dual && current_gap > 0
            solver.best_dual = ϕ_λ
            solver.best_iteration = solver.iteration
            solver.best_x .= solver.x
            solver.best_λ .= solver.λ
        end

        if current_gap > 0 && current_gap < 1e2
            solver.stepsize_choice = 4
        end

        # Store the current gap
        push!(solver.gaps, current_gap)

        # If λ does not change anymore, we typically violate constraints obtaining negative dual gap, or we reached the objective dual gap
        if check_λ_norm(solver, solver.λ, previous_λ)
            # Add last row
            # push!(df, [ solver.iteration, solver.timings[end], solver.dual_values[end], solver.x_distances[end], "OPT", solver.gaps[end] ])
            # break
        end

        if current_gap > 0 && current_gap <= solver.τ
            println("Found optimal dual gap")
            # Log result of the current iteration
            @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%s \n" solver.iteration solver.timings[end] solver.dual_values[end] solver.x_distances[end] solver.λ_distances[end] current_gap
            push!(df, [solver.iteration, solver.timings[end], solver.dual_values[end], solver.x_distances[end], solver.λ_distances[end], current_gap ])
            break   
        end

        if current_gap < -1e5
            # Gap is diverging, reset λ
            solver.λ = ones((solver.n,1))
        end

        if (solver.iteration == 1) || (solver.iteration % 10 == 0)
            # Log result of the current iteration
            @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" solver.iteration solver.timings[end] solver.dual_values[end] solver.x_distances[end] solver.λ_distances[end] current_gap
        end

        # Each one thousand iterations we clean some memory calling the Garbage collector
        if solver.iteration % 1000 == 0
            GC.gc()
        end

        # Add to DataFrame to save results
        push!(df, [solver.iteration, solver.timings[end], solver.dual_values[end], solver.x_distances[end], solver.λ_distances[end], current_gap ])

    end

    GC.gc()

    # Log total time and iterations
    print("\n")
    print("Iterations: $(solver.iteration)\tTotal time: $(round(sum(solver.timings), digits=6))\n")

    # Save results in CSV file
    CSV.write("logs/results_n=$(solver.n)_K=$(solver.K)_update=$(solver.update_formula)_defl=$(solver.deflection)_step=$(solver.stepsize_choice).csv", df)

    return solver

end


end