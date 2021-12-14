using LinearAlgebra
using Random
using Printf
using DataFrames
using CSV
using Ipopt
using MathProgBase

# #=
#     Compute the dual function value given the minimum x found and the value of λ
        
#         ϕ(λ) = x' Q x + q' x - λ' x 

# =#
# function dual_function(Q, q, x, λ)
#     return (dot(x, Q * x) + dot(q, x) - dot(λ, x))[1]
# end



# #=
#     Compute the subgradient of -ϕ() at the point λ_{t-1}. The subgradient is taken by computing the left limit 
#     and the right limit and then choosing the maximum norm of them 

#         s = argmin { ∥ s ∥ : s ∈ ∂ϕ(λ_{t-1}) }

#     where the set of subgradient is the interval [a, b] where 

#         a = lim λ -> λ_{t-1}^-  ( ϕ(λ) - ϕ(λ_{t-1}) ) / (λ - λ_{t-1})

#         b = lim λ -> λ_{t-1}^+  ( ϕ(λ) - ϕ(λ_{t-1}) ) / (λ - λ_{t-1})

#     If the a-b ≈ 0, then the gradient exists and is simply given by 

#         g^t = ∇_λ ϕ(λ) = - x

# =#
# function get_subgrad(Q, q, x, λ)

#     # First create the values λ_{t-1}^- and λ_{t-1}^+ with an ϵ > 0
#     ϵ = 1e-2

#     λ_minus = λ .- ϵ
#     λ_plus = λ .+ ϵ

#     # Compute the value of a
#     a = (dot(x, Q * x) + dot(q, x) - dot(λ_minus, x))[1] - (dot(x, Q * x) + dot(q, x) - dot(λ, x))[1] #dual_function(Q, q, x, λ) - dual_function(Q, q, x, λ)
#     a = a ./ (λ_minus - λ)

#     # Compute value of b
#     b = (dot(x, Q * x) + dot(q, x) - dot(λ_plus, x))[1] - (dot(x, Q * x) + dot(q, x) - dot(λ, x))[1] #dual_function(Q, q, x, λ) - dual_function(Q, q, x, λ)
#     b = b ./ (λ_plus - λ)

#     # If norm(a-b) ≈ 0, then the gradient exist
#     difference = norm(a-b)

#     if difference <= 1e-12
#         # The gradient exists and coincide with the normal derivation of ϕ(λ_{t-1})
#         return - x
#     end

#     # println("Subgradient with limit")

#     # Otherwise compute the maximum norm between a and b
#     a_norm = norm(a)
#     b_norm = norm(b)

#     min_norm = min(a_norm, b_norm)

#     return (sum.(a + b))./2

#     if min_norm == a_norm

#         return - a

#     else 

#         return - b

#     end

# end


# #=
#     Compute 

#         ∥ x_t - x_{t-1} ∥_2

# =#
# function x_norm(previous_x, current_x)

#     res = current_x .- previous_x

#     distance = norm(res)

#     return distance

# end

# #=
#     Compute

#         ∥ λ_t - λ_{t-1} ∥_2 
        
# =#
# function λ_norm(previous_λ, current_λ)
    
#     res = current_λ .- previous_λ

#     distance = norm(res)

#     return distance

# end

#=
    Compute the complementary slackness

        ∑ λ h(x) = - ∑ λ x

=#
# function complementary_slackness(x, λ) 

#     complementarity = λ ⋅ (- x)

#     println("Complementary slackness is $complementarity")

# end


#=

=#
function my_ADAGRAD(n, K, Q, q, A, λ, μ, x, δ, max_iter, ε, τ, stepsize_choice, F, Off_the_shelf_primal, α, β, h, rule)

    defl = true

    h = h # or rand(), or experimentations

    β = β # or rand(), or experimentations

    α = α # or rand(), or experimentations

    η = 1

    ϵ = 1e-10

    best_dual = -Inf

    # To create vector b = [λ_{t-1} - q, b]
    o = ones((K,1))

    o2 = ones((n,1))
    
    avg = zeros((n,1)) # avg
    G_t = zeros((n,1)) # G_t 
    s_t = zeros((n,1)) # s_t
    H_t = δ .* ones((n,1)) # H_t allocation
    d_i = zeros((n,1))

    total_time = 0.0
    
    # Log result of each iteration
    print("iter\t\ttime\t\tϕ(λ)\t\t∥x_t - x'∥\t∥λ_t-λ'∥\tf(x*)-ϕ(λ)\t\t∥g∥\t\tη\n\n")

    # Prepare a dataframe to store the values
    df = DataFrame( Iteration = Int[],
                    Time = Float64[],
                    DualValue = Float64[],
                    x_norm_residual = Float64[],
                    λ_norm_residual = Any[],
                    Dual_gap = Any[],
                    ∇g_norm = Any[],
                    η = Any[] )

    # Set the first optimal values
    iteration = 0

    ϕ_λ = (dot(x, Q * x) + dot(q, x) - dot(λ, x))[1] #dual_function(Q, q, x, λ)

    current_gap = (Off_the_shelf_primal - ϕ_λ) / abs(Off_the_shelf_primal)

    if current_gap > 0 && ϕ_λ > best_dual
        best_dual = ϕ_λ
        best_iteration = 0
        best_x = x
        best_λ = λ
    end

    times = 0

    @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" iteration 0.00000000 ϕ_λ norm(x) norm(λ) current_gap 0.000000000 η

    while iteration < max_iter

        try

            # Set starting time
            starting_time = time()

            # Increment iteration
            iteration += 1

            # Save iteration
            # push!(num_iterations, iteration)

            # Assign previous_λ
            previous_λ = λ

            # Assign previous_x
            previous_x = x

            # diff = map(-, λ, q)
            diff = λ - q

            b = [diff ; o]

            #= 
                Solve linear system efficiently using \ of Julia: will automatically use techniques like 
                backward and forward substitution to optimize the computation    
            =# 
            x_μ = F \ b

            x, μ = x_μ[1:n], x_μ[n+1 : n + K]

            # # Compute subgradient of ϕ (check report for derivation)
            λ_minus = λ .- ϵ
            λ_plus = λ .+ ϵ

            # Compute the value of a
            a = (dot(x, Q * x) + dot(q, x) - dot(λ_minus, x))[1] - (dot(x, Q * x) + dot(q, x) - dot(λ, x))[1] #dual_function(Q, q, x, λ) - dual_function(Q, q, x, λ)
            a = a ./ (λ_minus .- λ)

            # Compute value of b
            b = (dot(x, Q * x) + dot(q, x) - dot(λ_plus, x))[1] - (dot(x, Q * x) + dot(q, x) - dot(λ, x))[1] #dual_function(Q, q, x, λ) - dual_function(Q, q, x, λ)
            b = b ./ (λ_plus .- λ)

            # If norm(a-b) ≈ 0, then the gradient exist
            difference = norm(a-b)

            if difference ≤ 1e-6
                # The gradient exists and coincide with the normal derivation of ϕ(λ_{t-1})
                subgrad = - x
            else
                a_norm = norm(a)
                b_norm = norm(b)
                min_norm = min(a_norm, b_norm)
                if min_norm == a_norm
                    subgrad = - a
                else 
                    subgrad = - b
                end
            end
            # subgrad = -x

            if any( isnan, subgrad )
                println("Subgrad is nan!")
            end

            if defl

                if iteration == 1
                    d_i = subgrad
                else 
                    previous_d = iteration == 1 ? subgrad : d_i

                    # γ = ( d_i .* ( d_i .- subgrad ) ) ./ ( subgrad .- d_i ).^2
                    γ = 0.5 .* ones((n,1))
 
                    d_i = (γ .* subgrad) .+ ((o2 .- γ) .* previous_d)
                end

            end

            # Modify η in the proper way
            if stepsize_choice == 0
                η = h
            elseif stepsize_choice == 1
                η = h / norm( subgrad )
            elseif stepsize_choice == 2
                η = α / (β + iteration)
            elseif stepsize_choice == 3
                η = α / sqrt(iteration)
            else 
                η = 1.99 * (Off_the_shelf_primal - ϕ_λ) / (norm(subgrad)^2) 
            end
            
            #= 
                Solution of dual_function of problem 2.4 (see report)
                Accumulate the squared summation into s_t structure
            =#
            s_t .+= subgrad.^2

            avg .+= subgrad

            if any( isnan, s_t )
                println("s_t is nan!")
            end

            ϕ_λ = (dot(x, Q * x) + dot(q, x) - dot(λ, x))[1]

            if rule == 1
                #=
                    Update rule 1
                =#
                G_t .+= (subgrad .* subgrad)

                λ = defl ? λ + η * ( d_i ./ (sqrt.(G_t)) ) : λ .+ η .* ( subgrad ./ (sqrt.(G_t)) )   
            elseif rule == 2
                #=
                    Update rule 2
                =#  
                λ = η * iteration * ( (avg ./ iteration) ./ ( H_t .+ sqrt.(s_t) ) )
            elseif rule == 3
                #=
                    Update rule 3
                =#  
                λ = defl ? λ + η .* ( d_i ./ ( H_t .+ sqrt.(s_t) ) ) : λ + η .* ( subgrad ./ ( H_t .+ sqrt.(s_t) ) )
            else
                #=
                    Update rule 4
                =#
                λ = defl ? λ + (η .* d_i) : λ + (η .* subgrad)
            end

            # Projection over nonnegative orthant
            λ = max.(0, λ)


            # Check for NaN values
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

            # Compute \| x_t - x_{t-1} \|_2 and save it
            x_distance = norm(x .- previous_x)

            # Compute current dual_gap
            current_gap = (Off_the_shelf_primal - ϕ_λ) / abs(Off_the_shelf_primal)

            if isnan( current_gap )
                println("Some NaN values detected")
                break
            end

            # Update the best solution if conditions are met
            if ϕ_λ > best_dual && current_gap > 0
                best_dual = ϕ_λ
                best_iteration = iteration
                best_x .= x
                best_λ .= previous_λ
            end

            λ_distance = norm(λ .- previous_λ)

            # Store timing result of this iteration
            finish_time = time()    

            # Compute timing needed for this iteration
            time_step = finish_time - starting_time
                
            # Save time step
            total_time += time_step

            g_norm = norm(subgrad)

            complementarity = dot(λ, - x)

            if complementarity ≤ ϵ && complementarity > 0
                println("Saddle point reached")
                # Log result of the current iteration
                @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" iteration time_step ϕ_λ x_distance λ_distance current_gap g_norm η
                push!(df, [iteration, time_step, ϕ_λ, x_distance, λ_distance, current_gap, g_norm, η])
                break   
            end

            # Reached optimal gap values
            if current_gap > 0 && current_gap ≤ τ
                println("Found optimal dual gap")
                # Log result of the current iteration
                @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" iteration time_step ϕ_λ x_distance λ_distance current_gap g_norm η
                push!(df, [iteration, time_step, ϕ_λ, x_distance, λ_distance, current_gap, g_norm, η])
                break   
            end

            # λ is not changing anymore
            if λ_distance < ε && iteration > 10
                if times == 100
                    println("λ not changing anymore")
                    # Log result of the current iteration
                    @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" iteration time_step ϕ_λ x_distance λ_distance current_gap g_norm η
                    push!(df, [iteration, time_step, ϕ_λ, x_distance, λ_distance, current_gap, g_norm, η])
                    break
                else 
                    times += 1
                end  
            end

            # The gradient vanished
            if norm(subgrad) < ε
                println("Subgradient is zero")
                # Log result of the current iteration
                @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" iteration time_step ϕ_λ x_distance λ_distance current_gap g_norm η
                push!(df, [iteration, time_step, ϕ_λ, x_distance, λ_distance, current_gap, g_norm, η])
                break   
            end

            # Log result of the current iteration
            @printf "%d\t\t%.8f \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \t%1.5e \n" iteration time_step ϕ_λ x_distance λ_distance current_gap g_norm η
                
            # Add to DataFrame to save results
            push!(df, [iteration, time_step, ϕ_λ, x_distance, λ_distance, current_gap, g_norm, η])

        catch y

            if isa(y, InterruptException)

                complementarity = dot(best_λ, - best_x)

                println("Complementary slackness is $complementarity")

                # Log total time and iterations
                print("\n")
                print("Iterations: $(iteration)\tTotal time: $(round(total_time, digits=6))\n")

                # Save results in CSV file
                CSV.write("logs/results_n=$(n)_K=$(K)_update=$(rule)_alpha=$(α)_step=$(stepsize_choice).csv", df)

                break

            else

                continue
            
            end
            

        end

    end

    complementarity = dot(best_λ, - best_x)

    println("Complementary slackness is $complementarity")

    # Log total time and iterations
    print("\n")
    print("Iterations: $(iteration)\tTotal time: $(round(total_time, digits=6))\n")

    # Save results in CSV file
    CSV.write("logs/results_n=$(n)_K=$(K)_update=$(rule)_alpha=$(α)_step=$(stepsize_choice).csv", df)


    #------------------------------------------------------#
    #------------     Results for rules     ---------------#
    #------------------------------------------------------#

    print("\n\n\n\n")
    print("------------------- Rule $(rule) results -------------------\n\n")

    println("Optimal x found (rule $(rule)):")
    display(best_x)
    print("\n")

    println("Optimal λ found (rule $(rule)):")
    display(best_λ)
    print("\n")

    println("Best value of dual function at iteration $(best_iteration) (rule $(rule)):")
    display(best_dual)
    print("\n")

    println("Duality gap between f( x* ) and ϕ( λ ) (rule $(rule)):")

    dual_gap = (Off_the_shelf_primal - best_dual) / abs(Off_the_shelf_primal)

    display(dual_gap)
    print("\n")

end


function initialize(step, rule, α, β, δ, h, max_iter) 

    #------------------------------------------------------#
    #---------     Initialize all parameters    -----------#
    #------------------------------------------------------#
    print("Input n value: ")
    n = parse(Int64, readline())

    print("Enter K value for n=$(n): ")
    K = parse(Int64, readline())

    #=
        0: Constant step size                   η = h                               with h > 0
        1: Constant step length                 η = h / ∥ g_k ∥_2                   with h = ∥ λ_{t+1} - λ_t ∥_2 
        2: Square summable but not summable     η = α / (b + t)                     with α > 0 and b ≥ 0
        3: Nonsummable diminishing              η = α / √t                          with α > 0 
        4: Polyak                               η = β * f(x*) - ϕ(λ_t) / ∥ g_k ∥^2  with β ∈ (0,2)
    =#
    stepsize_choice = step

    println("Initializing random disjoint sets")

    # Indexes to access the different x_i
    indexes = collect(1:1:n)

    # I_K contains all the I^k required to create the simplices constraints
    # I_K = y == "y" ? vars["I"] : initialize_sets(indexes, K, n)

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
    I_K = Vector{Array{Int64,1}}()
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
        push!(I_K, values)
    end

    println("Set of I_K arrays:")
    display(I_K)
    print("\n")

    # Initialize λ iterates to ones
    λ = ones((n,1))

    # Create random matrix A_help
    A_help = rand(Float64, n, n)

    Q = A_help' * A_help

    # eigsQ=eigvals(Q)
    # println("Q is a positive definite matrix with max eigenvalue ", maximum(eigsQ), "and minimumeigen value ", minimum(eigsQ))

    println("Q matrix:")
    display(Q)
    print("\n")

    # Initialize also random q vector 
    q = rand(Float64, (n,1))

    println("q vector:")
    display(q)
    print("\n")

    # Initialize δ
    δ = δ # or abs(rand())

    # Initialize max_iter
    max_iter = max_iter

    # Initialize ε
    ε = 1e-14

    # Initialize τ
    τ = 1e-6

    # A = y == "y" ? vars["A"] : construct_A(K, n, I_K)
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

    println("A constraint matrix:")
    display(A)
    print("\n")

    Full_mat = vcat(2*Q, A)
        
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

    x_μ = F \ vcat(λ - q, ones((K,1))) 

    x, μ = x_μ[1:n], x_μ[n+1 : n + K]

    println("Starting x:")
    display(x)
    print("\n")

    println("Starting λs:")
    display(λ)
    print("\n")


    # Optimal solution using quadprog
    sol = quadprog(vec(q), 2*Q, A, '=', vec(ones(K,1)), vec(zeros(n,1)), +Inf, IpoptSolver())

    if sol.status == :Optimal
        println("Optimal objective value is $(sol.objval)")
        # println("Optimal solution vector is: [$(sol.sol[1]), $(sol.sol[2]), $(sol.sol[3])]")
    else
        println("Error: solution status $(sol.status)")
    end

    opt_val = sol.objval

    GC.gc()

    # Modify last parameter for update rule
    @time my_ADAGRAD(n, K, Q, q, A, λ, μ, x, δ, max_iter, 
                ε, τ, stepsize_choice, F, opt_val, α, 
                β, h, rule)

end


# Modify steps
# initialize(0, 3, 0.1, 0, 1e1, 1e1, 10000) #= steprule, rule, α, β, δ, h, max_iter =#
