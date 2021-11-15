using LinearAlgebra
using Random
using MAT
using Printf
using DataFrames
using CSV

#=
    Compute the dual function value given the minimum x found and the value of λ
        
        ϕ(λ) = x' Q x + q' x - λ' x 

=#
function dual_function(Q, q, n, K, x, λ, μ, A, o)
    X = similar(o)
    return (x ⋅ (Q * x)) .+ (q ⋅ x) .- (λ ⋅ x)
    # return BLAS.dot(n, x, 1, BLAS.symv('U' ,Q, x),1) + BLAS.dot(n, q, 1, x, 1) - BLAS.dot(n, λ, 1, x, 1) + BLAS.dot(K, μ, 1, map(-, mul!(X, A, x), o), 1)
end



#= 

    Compute one among the three possible update_rule specified in the report.
    
    The first update rule is a general update rule given by:

        λ_t = λ_{t-1} + η diag(G_t)^{-1/2} g_t

    where G_t is the full outer product of all the stored subgradient

    The second update rule is:

        λ_t = - H_{t-1}^{-1} t η ̅g_t
        
    where ̅g_t is the average of the gradients at step t

    The third one employ the following:

        λ_t = λ_{t-1} + η H_{t-1}^{-1} g_t

=#
## COMPLETARE QUESTO
function compute_update_rule(n, λ, η, update_rule, s_t, iteration, G_t, H_t, last_subgrad)

    # Preallocate λ and G_t
    λ_new = Array{Float64,1}(undef, n)

    G_t_local = Array{Float64,1}(undef, n)

    if update_rule == 1
        
        # Add the latter diagonal of g_t * g_t' component-wise to the vector G_t
        full_prod = map(*, last_subgrad, last_subgrad)

        if any( isnan, full_prod )
            println("full_prod is nan!")
        end

        G_t .+= full_prod

        G_t_local = G_t 

        if any( isnan, G_t_local )
            println("G_t is nan!")
        end

        if any(x -> x<0, G_t_local) 
            println("G_t is negative")
        end

        # Apply square root
        G_t_local .= map(sqrt, G_t_local)

        # Update λ using the corresponding formula
        λ_new .= map(/, η, G_t_local)
        
        λ_new .= map(*, λ_new, last_subgrad)

        axpy!(1, λ, λ_new)

        # λ_new = λ + last_subgrad .* (G_t_local .* (1/η))

    elseif update_rule == 2

        # Average the row sum of the gradient based on the current iteration in a new variable
        avg_gradient_copy = map(/, s_t, iteration)

        λ_new .= map(/, avg_gradient_copy, H_t)

        scalar = η * iteration

        BLAS.scal!(n, scalar, λ_new, 1)

    elseif update_rule == 3 

        λ_new .= map(/, η, H_t)

        λ_new .= map(*, λ_new, last_subgrad)

        axpy!(1, λ, λ_new)

    else

        # BLAS.scal!(n, Float64(η), λ, 1)

        # λ_new .= map(*, λ, last_subgrad)

        # axpy!(1, λ, λ_new)

        λ_new = λ + (η * last_subgrad)

    end

    λ_new .= max.(0, λ_new)

    return λ_new

end


#=
    Compute the subgradient of -ϕ() at the point λ_{t-1}. The subgradient is taken by computing the left limit 
    and the right limit and then choosing the maximum norm of them 

        s = argmin { ∥ s ∥ : s ∈ ∂ϕ(λ_{t-1}) }

    where the set of subgradient is the interval [a, b] where 

        a = lim λ -> λ_{t-1}^-  ( ϕ(λ) - ϕ(λ_{t-1}) ) / (λ - λ_{t-1})

        b = lim λ -> λ_{t-1}^+  ( ϕ(λ) - ϕ(λ_{t-1}) ) / (λ - λ_{t-1})

    If the a-b ≈ 0, then the gradient exists and is simply given by 

        - g^t = ∇_λ ϕ(λ) = x

=#
function get_subgrad(Q, q, n, K, x, λ, μ, A, o)

    # First create the values λ_{t-1}^- and λ_{t-1}^+ with an ϵ > 0
    ϵ = 1e-2

    λ_minus = λ .- ϵ
    λ_plus = λ .+ ϵ

    # Compute the value of a
    a = dual_function(Q, q, n, K, x, λ_minus, μ, A, o) - dual_function(Q, q, n, K, x, λ, μ, A, o)
    a = a ./ (λ_minus - λ)

    # Compute value of b
    b = dual_function(Q, q, n, K, x, λ_plus, μ, A, o) - dual_function(Q, q, n, K, x, λ, μ, A, o)
    b = b ./ (λ_plus - λ)

    # If norm(a-b) ≈ 0, then the gradient exist
    difference = norm(a-b)

    if difference <= 1e-12
        # The gradient exists and coincide with the normal derivation of ϕ(λ_{t-1})
        return - x
    end

    # println("Subgradient with limit")

    # Otherwise compute the maximum norm between a and b
    a_norm = norm(a)
    b_norm = norm(b)

    min_norm = min(a_norm, b_norm)

    if min_norm == a_norm

        return - a

    else 

        return - b

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
    Compute

        ∥ λ_t - λ_{t-1} ∥_2 
        
=#
function λ_norm(previous_λ, current_λ)
    
    res = current_λ .- previous_λ

    distance = norm(res)

    return distance

end

#=

    Compute the complementary slackness

        ∑ λ h(x) = - ∑ λ x

=#
function complementary_slackness(x, λ) 

    complementarity = λ ⋅ (- x)

    println("Complementary slackness is $complementarity")

end


function my_ADAGRAD(n, K, Q, q, A, λ, μ, x, δ, max_iter, ε, τ, stepsize_choice, F, Off_the_shelf_primal, update_rule)

    h = 500 # or rand(), or experimentations

    β = 1 # or rand(), or experimentations

    α = 10 # or rand(), or experimentations

    best_dual = -Inf

    # To create vector b = [λ_{t-1} - q, b]
    o = ones((K,1))

    # H_t allocation
    H_t = δ .* ones((n,1))

    # s_t allocation
    s_t = zeros((n,1))
    
    num_iterations = Vector{Float64}() # num_iterations
    dual_values = Vector{Float64}() # dual_values
    λ_distances = Vector{Float64}() # λ_distances
    x_distances = Vector{Float64}() # x_distances
    timings = Vector{Float64}() # timings
    gaps = Vector{Float64}() # gaps
    grads = Array{Float64}(undef, n, 0) # grads 
    G_t = zeros((n,1)) # G_t 
    s_t_local = zeros((n,1)) # s_t
    H_t_local = zeros((n,1))

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
    iteration = 0

    ϕ_λ = dual_function(Q, q, n, K, x, λ, μ, A, o)[1]

    current_gap = Off_the_shelf_primal - ϕ_λ

    if current_gap > 0 && ϕ_λ > best_dual
        best_dual = ϕ_λ
        best_iteration = 0
        best_x = x
        best_λ = λ
    end

    @printf "%d\t\t%.8f \t%.8f \t%.8f \t%.8f \t%.8f \n" iteration 0.00000000 ϕ_λ norm(x) norm(λ) current_gap

    while iteration < max_iter

        try

            # Set starting time
            starting_time = time()

            # Increment iteration
            iteration += 1

            # Save iteration
            push!(num_iterations, iteration)

            # Assign previous_λ
            previous_λ = λ

            # Assign previous_x
            previous_x = x

            # Compute subgradient of ϕ (check report for derivation)
            subgrad = get_subgrad(Q, q, n, K, x, λ, μ, A, o)

            if any( isnan, subgrad )
                println("Subgrad is nan!")
            end

            # Revert subgrad direction if we gap is diverging
            # current_gap < 0 ? subgrad = - subgrad : subgrad = subgrad

            # Store subgradient in matrix
            grads = [grads subgrad]

            # Modify η in the proper way
            if stepsize_choice == 1
                η = h / norm( grads[:, end] )
            elseif stepsize_choice == 2
                η = α / (β + iteration)
            elseif stepsize_choice == 3
                η = α / sqrt(iteration)
            else 
                η = isempty(dual_values) ? 1 : ( (Off_the_shelf_primal - dual_values[end]) / norm(grads[:,end])^2 )
            end
            
            #= 
                Solution of dual_function of problem 2.4 (see report)
                Accumulate the squared summation into s_t structure
            =#
            s_t += subgrad

            if any( isnan, s_t )
                println("s_t is nan!")
            end

            # Create a copy of s_t (avoid modifying the original one)
            s_t_local = s_t 

            if any( isnan, s_t_local )
                println("s_t_local is nan!")
            end

            # Compute s_t
            s_t_local .= norm.(s_t_local)

            if any( isnan, s_t_local )
                println("s_t_local is nan!")
            end

            # Sum them (hessian approximation)
            H_t_local = H_t

            H_t_local .+= s_t_local

            if any( isnan, H_t_local )
                println("H_t is nan!")
            end

            diff = map(-, λ, q)

            b = vcat(diff, o)

            #= 
                Solve linear system efficiently using \ of Julia: will automatically use techniques like 
                backward and forward substitution to optimize the computation    
            =# 
            x_μ = \(F, b)

            if any( isnan, x_μ )
                println("x_μ is nan!")
            end

            x, μ = x_μ[1:n], x_μ[n+1 : n + K]

            #=
                Compute the update rule for the lagrangian multipliers λ: can use 
                one among the three showed, then soft threshold the result
            =#
            λ = compute_update_rule(n, λ, η, update_rule, s_t, iteration, G_t, H_t, grads[:,end])
            
            # Compute Lagrangian function value
            ϕ_λ = dual_function(Q, q, n, K, x, λ, μ, A, o)[1]

            if any( isnan, λ )
                println("λ is nan!")
                break
            elseif any( isnan, x )
                println("x is nan!")
                break
            elseif isnan( ϕ_λ ) 
                println("ϕ_λ is nan!")
                break
            end 

            # complementary_slackness(x, λ)

            # Storing current relaxation value
            push!(dual_values, ϕ_λ)

            # Compute \| x_t - x_{t-1} \|_2 and save it
            push!(x_distances, x_norm(previous_x, x))

            # Store timing result of this iteration
            finish_time = time()    

            # Compute timing needed for this iteration
            time_step = finish_time - starting_time
                
            # Save time step
            push!(timings, time_step)

            # Compute current dual_gap
            current_gap = Off_the_shelf_primal - dual_values[end]

            if isnan( current_gap )
                println("Some NaN values detected")
                break
            end

            # Update the best solution if conditions are met
            if ϕ_λ > best_dual && current_gap > 0
                best_dual = ϕ_λ
                best_iteration = iteration
                best_x .= x
                best_λ .= λ
            end

            # if current_gap < 2e4
            #     stepsize_choice = 4
            # end

            # Store the current gap
            push!(gaps, current_gap)

            push!(λ_distances, λ_norm(previous_λ, λ))

            if current_gap > 0 && current_gap <= τ
                println("Found optimal dual gap")
                # Log result of the current iteration
                @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%s \n" iteration timings[end] dual_values[end] x_distances[end] λ_distances[end] current_gap
                push!(df, [iteration, timings[end], dual_values[end], x_distances[end], λ_distances[end], current_gap ])
                break   
            end

            if λ_distances[end] < ε && iteration > 10
                println("λ not changing anymore")
                # Log result of the current iteration
                @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%s \n" iteration timings[end] dual_values[end] x_distances[end] λ_distances[end] current_gap
                push!(df, [iteration, timings[end], dual_values[end], x_distances[end], λ_distances[end], current_gap ])
                break   
            end

            if norm(subgrad) < ε
                println("Subgradient is zero")
                # Log result of the current iteration
                @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%s \n" iteration timings[end] dual_values[end] x_distances[end] λ_distances[end] current_gap
                push!(df, [iteration, timings[end], dual_values[end], x_distances[end], λ_distances[end], current_gap ])
                break   
            end

            # if current_gap < -1e5
            #     # Gap is diverging, reset λ
            #     λ = best_λ
            # end

            if (iteration == 1) || (iteration % 1 == 0)
                # Log result of the current iteration
                @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" iteration timings[end] dual_values[end] x_distances[end] λ_distances[end] current_gap
            end

            # Each one thousand iterations we clean some memory calling the Garbage collector
            if iteration % 1000 == 0
                GC.gc()
            end

            # Add to DataFrame to save results
            push!(df, [iteration, timings[end], dual_values[end], x_distances[end], λ_distances[end], current_gap ])

        catch y

            if isa(y, InterruptException)

                GC.gc()

                # Log total time and iterations
                print("\n")
                print("Iterations: $(iteration)\tTotal time: $(round(sum(timings), digits=6))\n")

                # Save results in CSV file
                CSV.write("logs/results_n=$(n)_K=$(K)_update=$(update_rule)_alpha=$(α)_step=$(stepsize_choice).csv", df)

                break

            else

                continue
            
            end
            

        end

    end

    GC.gc()

    # Log total time and iterations
    print("\n")
    print("Iterations: $(iteration)\tTotal time: $(round(sum(timings), digits=6))\n")

    # Save results in CSV file
    CSV.write("logs/results_n=$(n)_K=$(K)_update=$(update_rule)_alpha=$(α)_step=$(stepsize_choice).csv", df)


    #------------------------------------------------------#
    #------------     Results for rules     ---------------#
    #------------------------------------------------------#

    print("\n\n\n\n")
    print("------------------- Rule $(update_rule) results -------------------\n\n")

    println("Optimal x found (rule $(update_rule)):")
    display(best_x)
    print("\n")

    println("Optimal λ found (rule $(update_rule)):")
    display(best_λ)
    print("\n")

    println("Best value of dual function at iteration $(best_iteration) (rule $(update_rule)):")
    display(best_dual)
    print("\n")

    println("Duality gap between f( x* ) and ϕ( λ ) (rule $(update_rule)):")

    dual_gap = Off_the_shelf_primal - best_dual

    display(dual_gap)
    print("\n")

end


# Utility function to create the different sets I^k given the number of sets K
function initialize_sets(indexes, K, n)
    # Put at least one element in each set I^k
    instances = ones(Int64, K)
    # Distribute the remaining values randomly inside instances
    remaining = n - K
    while remaining > 0
        Random.seed!(abs(rand(Int,1)[1]))
        random_index = rand(1:K)
        instances[random_index] += 1
        remaining -= 1
    end
    # Create empty multinensional Array
    I = Vector{Array{Int64,1}}()
    # Iterate through the required cardinality of each index set
    for value in instances
        tmp = value
        values = []        
        # Shuffle each time the index array and put the value inside the current set I^k
        while tmp > 0
            shuffle!(indexes)
            push!(values, pop!(indexes)) 
            tmp -= 1
        end
        # Add the current I^k to I
        push!(I, values)
    end
    return I
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


function initialize() 

    #------------------------------------------------------#
    #---------     Initialize all parameters    -----------#
    #------------------------------------------------------#
    print("Use stored .mat?[y/n] ")
    y = readline()

    vars = y == "y" ? matread("mat/structs_n1000_K100.mat") : []

    print("Input n value: ")
    n = y == "y" ? length(vars["q"]) : parse(Int64, readline())

    print("Enter K value for n=$(n): ")
    K = y == "y" ? size(vars["A"],1) : parse(Int64, readline())

    #=
        0: Constant step size                   η = h               with h > 0
        1: Constant step length                 η = h / ∥ g_k ∥_2   with h = ∥ λ_{t+1} - λ_t ∥_2 
        2: Square summable but not summable     η = α / (b + t)     with α > 0 and b ≥ 0
        3: Nonsummable diminishing              η = α / √t          with α > 0 
        4: Polyak                               η = f(x*) - ϕ(λ_t) / ∥ g_k ∥^2
    =#
    stepsize_choice = 2

    println("Initializing random disjoint sets")

    # Indexes to access the different x_i
    indexes = collect(1:1:n)

    # I_K contains all the I^k required to create the simplices constraints
    I_K = y == "y" ? vars["I"] : initialize_sets(indexes, K, n)

    # Fix I_K structure
    I_K = vec(I_K)

    for i=1:1:length(I_K)
        if (typeof(I_K[i]) == Float64) || (typeof(I_K[i]) == Int64) 
            I_K[i] = [I_K[i]]
        end

        map(y -> round.(Int, y) , I_K[i])
    end


    println("Set of I_K arrays:")
    display(I_K)
    print("\n")

    # Initialize λ iterates to ones
    λ = ones((n,1))

    # Create random matrix A_help
    A_help = rand(Float64, n, n)

    Q = y == "y" ? vars["Q"] : A_help' * A_help

    println("Q matrix:")
    display(Q)
    print("\n")

    # Initialize also random q vector 
    q = y == "y" ? vars["q"] : rand(Float64, (n,1))

    println("q vector:")
    display(q)
    print("\n")

    # Initialize δ
    δ = 1 # or abs(rand())

    # Initialize max_iter
    max_iter = 10000

    # Initialize ε
    ε = 1e-14

    # Initialize τ
    τ = 1e-7

    A = y == "y" ? vars["A"] : construct_A(K, n, I_K)
    # A = y == "y" ? vars["A"] : construct_A(K, n, I_K)

    println("A constraint matrix:")
    display(A)
    print("\n")

    Full_mat = vcat(Q, A)
        
    zero_mat = zeros((K, K))

    Right_side = vcat(A', zero_mat)

    Full_mat = hcat(Full_mat, Right_side)

    println("Full matrix:")
    display(Full_mat)
    print("\n")

    #=
        Let Julia automatically determine the best factorization:
            
            Cholesky ( if Full_mat ≻ 0 )
            Bunch-Kaufman ( if Q=Q^T )
            pivoted LU ( otherwise )
    =#
    F = factorize(Full_mat)

    println("Factorization:")
    display(F)
    print("\n")

    GC.gc()

    x_μ = \( F, vcat(λ - q, ones((K,1))) )

    if any( isnan, x_μ )
        println("x_μ is nan!")
    end

    x, μ = x_μ[1:n], x_μ[n+1 : n + K]

    println("Starting x:")
    display(x)
    print("\n")

    println("Starting λs:")
    display(λ)
    print("\n")

    # matwrite("mat/structs_n$(n)_K$(K).mat", Dict(
    #     "Q" => Q,
    #     "A" => A,
    #     "q" => q,
    #     "I" => I_K
    # );compress = true) 


    # Optimal value for structs_n100_K10
    # opt_val = 2.144799092996162e+03
    # time 1.091 seconds 
    # iterations 21
    
    # Optimal value for structs_n1000_K20
    # opt_val = 9.260942226793662e+04
    # time 1.091 seconds 
    # iterations 21

    # Optimal value for structs_n1000_K100
    opt_val = 2.365974236431491e+06
    # time 1.242 seconds
    # iterations 17 

    # Optimal value for structs_n1000_K500
    # opt_val = 6.158714971181200e+07
    # time 0.938 seconds
    # iterations 17

    # Optimal value for structs_n5000_K10
    # opt_val = 1.202577688949518e+05
    # time 41.414 seconds
    # iterations 23

    # Optimal value for structs_n5000_K1000
    # opt_val = 1.228530174591432e+09
    # time 76.352 seconds
    # iterations 26

    # Optimal value for structs_n5000_K2500
    # opt_val = 7.759039617014161e+09
    # time 69.911 seconds
    # iterations 28

    # Optimal value for structs_n10000_K10
    # opt_val = 2.428012021226884e+05
    # time 401.688 seconds
    # iterations 28

    # Optimal value for structs_n10000_K2500
    # opt_val = 1.544997631806922e+10
    # time 971.025 seconds
    # iterations 34

    # Modify last parameter for update rule
    my_ADAGRAD(n, K, Q, q, A, λ, μ, x, δ, max_iter, 
                ε, τ, stepsize_choice, F, opt_val, 4)


end



initialize()
