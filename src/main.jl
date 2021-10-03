include("./ADAGRAD_Solver.jl")
include("./Utils.jl")
include("./JuMPSolution.jl")

using LinearAlgebra
using Random
using Plots
using Colors
using .Utils
using .ADAGRAD_Solver
# Compute the solution of JuMP.jl
using .JuMPSolution


function testing(n_list, K_list, deflections)

    for (n, Ks) in zip(n_list, K_list)

        for K in Ks

            println("Initializing random disjoint sets")

            # Indexes to access the different x_i
            global indexes = collect(1:1:n)

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

            # I_K contains all the I^k required to create the simplices constraints
            global I_K = initialize_sets(indexes, K, n)

            indexes = nothing

            println("Set of I_K arrays:")
            display(I_K)
            print("\n")

            # Initialize x iterates to ones
            global x = randn((n,1))#ones((n,1))

            # Feasible x
            for set in I_K
                for val in set
                    x[val] = 1/length(set)
                end
            end

            # Initialize λ iterates to ones
            global λ = ones((n,1))

            # # Create random matrix A_help
            global A_help = randn(Float64 ,n, n)

            global r = rank(A_help)

            println("Matrix rank:")
            display(r)

            while r < n
                global A_help
                global r
                A_help = randn(Float64 ,n, n)
                r = rank(A_help)
                println("Matrix rank:")
                display(r)
            end

            # Compute the inverse 
            global A_help_inv = inv(A_help)

            #=
                Construct Q using a surely nonsingular matrix A_H using positive random eigenvalues.
                A random probability number is used to set an eigenvalue as 0 or not,
                to manage positive semidefinite or positive definite case.
                If prob > 0.5 ⟹ some given numbers of eigenvalues are set to 0
                Else ⟹ Q is created as positive definite 
            =#

            # Eigenvalues of Q
            global vect = abs.(randn((n,1)))

            # Random number used to set zeros
            global prob = abs(rand())

            if prob > 0.5
                println("Prob greater than zero")

                # Random number of zeros to set
                global eigens = rand(1:n-1)

                # Loop over vect and modify eigenvalues
                global ind = 0
                
                while eigens > 0
                    vect[end - ind] = 0.0
                    global ind
                    global eigens
                    ind += 1
                    eigens -= 1    
                end

                display(vect)

            end

            global vect = vec(vect)

            global Q = A_help * Diagonal(vect) * A_help_inv
            # global Q = A_help' * A_help

            # Garbage collector: free up some memory
            vect = nothing
            A_help = nothing
            A_help_inv = nothing
            GC.gc()

            println("Q matrix:")
            display(Q)
            print("\n")

            # Initialize also random q vector 
            q = randn(Float64, (n,1))

            println("q vector:")
            display(q)
            print("\n")

            # Initialize η
            η = 1

            # Initialize δ
            δ = abs(rand())

            # Initialize max_iter
            max_iter = 100000

            # Initialize ε
            ε = 1e-6

            # Initialize τ
            τ = 1e-1

            A = Utils.construct_A(K, n, I_K)

            println("A constraint matrix:")
            display(A)
            print("\n")

            Full_mat = Utils.construct_full_matrix(Q, A, K)

            println("Full matrix:")
            display(Full_mat)
            print("\n")

            #= 
                Let Julia automatically determine the best factorization:
                    
                    Cholesky ( if Full_mat ≻ 0 )
                    Bunch-Kaufman ( if Q=Q^T )
                    pivoted LU ( otherwise )
            =#
            # Lu factorization (unless Full_mat is symmetric)
            F = factorize(Full_mat)

            println("Factorization:")
            display(F)
            print("\n")

            #------------------------------------------------------#
            #-----    Use JuMP.jl to compute primal solution   ----#
            #------------------------------------------------------#

            jump_sol = JuMPSolution.JuMPSol(
                n,
                K,
                A,
                Q,
                q
            )

            jump_sol = JuMPSolution.compute_solution(jump_sol)

            println("Optimal value:")
            display(jump_sol.opt_val)
            print("\n")

            #------------------------------------------------------#
            #----------     Create problem structs     ------------#
            #------------------------------------------------------#

            # val = Utils.compute_lagrangian(Q, q, x, λ)[1]

            #= 
                Check that we are not violating the lagrangian:
                compute new random values for x and λ to start in a feasible solution
            =#
            # while (convex_sol.opt_val - val) < 0
            #     global x
            #     global λ
            #     x = randn((n,1))
            #     λ = abs.(randn((n,1)))
            #     global val 
            #     val = Utils.compute_lagrangian(Q, q, x, λ)[1]
            # end

            println("Starting x:")
            display(x)
            print("\n")

            println("Starting λs:")
            display(λ)
            print("\n")

            for deflection in deflections

                # Create three different struct to exploit the three update rule
                solver_rule1 = ADAGRAD_Solver.Solver(
                    n, 0, λ, K, I_K, x, Array{Float64}(undef, n, 0), # grads 
                    Diagonal(Matrix{Float64}(undef, n, n)), # G_t 
                    Array{Float64}(undef, n, 1), # s_t
                    Array{Float64}(undef, n, 1), # avg_gradient
                    Array{Float64}(undef, n, 1), # d_i
                    deflection, # deflection
                    Q, q, η, δ, max_iter, ε, τ, -Inf, # best_lagrangian
                    0, # best_iteration
                    x, # best_x
                    λ, # best_λ
                    Vector{Float64}(), # num_iterations
                    Vector{Float64}(), # relaxation_values
                    Array{Float64}(undef, n, 0), # x_values
                    Array{Float64}(undef, n, 0), # λ_values
                    Vector{Float64}(), # λ_distances
                    Vector{Float64}(), # x_distances
                    Vector{Float64}(), # timings
                    Vector{Float64}(), # gaps
                    1, # update_rule
                    Full_mat, # Full matrix KKT
                    F, # Best factorization
                    A, # Constraint matrix
                    jump_sol.opt_val # Primal optimal value
                )

                solver_rule2 = ADAGRAD_Solver.Solver(
                    n, 0, λ, K, I_K, x, Array{Float64}(undef, n, 0), # grads 
                    Diagonal(Matrix{Float64}(undef, n, n)), # G_t 
                    Array{Float64}(undef, n, 1), # s_t
                    Array{Float64}(undef, n, 1), # avg_gradient
                    Array{Float64}(undef, n, 1), # d_i
                    deflection, # deflection
                    Q, q, η, δ, max_iter, ε, τ, -Inf, # best_lagrangian
                    0, # best_iteration
                    x, # best_x
                    λ, # best_λ
                    Vector{Float64}(), # num_iterations
                    Vector{Float64}(), # relaxation_values
                    Array{Float64}(undef, n, 0), # x_values
                    Array{Float64}(undef, n, 0), # λ_values
                    Vector{Float64}(), # λ_distances
                    Vector{Float64}(), # x_distances
                    Vector{Float64}(), # timings
                    Vector{Float64}(), # gaps
                    2, # update_rule
                    Full_mat, # Full matrix KKT
                    F, # Best factorization
                    A, # Constraint matrix
                    jump_sol.opt_val # Primal optimal value
                )

                solver_rule3 = ADAGRAD_Solver.Solver(
                    n, 0, λ, K, I_K, x, Array{Float64}(undef, n, 0), # grads 
                    Diagonal(Matrix{Float64}(undef, n, n)), # G_t 
                    Array{Float64}(undef, n, 1), # s_t
                    Array{Float64}(undef, n, 1), # avg_gradient
                    Array{Float64}(undef, n, 1), # d_i
                    deflection, # deflection
                    Q, q, η, δ, max_iter, ε, τ, -Inf, # best_lagrangian
                    0, # best_iteration
                    x, # best_x
                    λ, # best_λ
                    Vector{Float64}(), # num_iterations
                    Vector{Float64}(), # relaxation_values
                    Array{Float64}(undef, n, 0), # x_values
                    Array{Float64}(undef, n, 0), # λ_values
                    Vector{Float64}(), # λ_distances
                    Vector{Float64}(), # x_distance
                    Vector{Float64}(), # timings
                    Vector{Float64}(), # gaps
                    3, # update_rule
                    Full_mat, # Full matrix KKT
                    F, # Best factorization
                    A, # Constraint matrix
                    jump_sol.opt_val # Primal optimal value
                )

                #------------------------------------------------------#
                #--------     Calculate custom solution     -----------#
                #------------------------------------------------------#


                # Now calculate the results of ADAGRAD in the three different fashion way
                solver_rule1 = @time ADAGRAD_Solver.my_ADAGRAD(solver_rule1)

                solver_rule2 = @time ADAGRAD_Solver.my_ADAGRAD(solver_rule2)

                solver_rule3 = @time ADAGRAD_Solver.my_ADAGRAD(solver_rule3)

                gaps = Dict{String,Float64}()
                L_values = Dict{String,Float64}()

                #------------------------------------------------------#
                #-----------     Results for rule 1     ---------------#
                #------------------------------------------------------#

                print("\n\n\n\n")
                print("------------------- Rule 1 results -------------------\n\n")

                println("Optimal x found (rule 1):")
                display(solver_rule1.best_x)
                print("\n")

                println("Optimal λ found (rule 1):")
                display(solver_rule1.best_λ)
                print("\n")

                println("Best value of lagrangian at iteration $(solver_rule1.best_iteration) (rule 1):")
                display(solver_rule1.best_lagrangian)
                print("\n")

                println("Duality gap between f( x* ) and ϕ( λ ) (rule 1):")

                dual_gap = jump_sol.opt_val - solver_rule1.best_lagrangian

                display(dual_gap)
                print("\n")

                gaps["Rule 1"] = dual_gap
                L_values["Rule 1"] = solver_rule1.best_lagrangian

                #------------------------------------------------------#
                #-----------     Results for rule 2     ---------------#
                #------------------------------------------------------#

                print("\n\n\n\n")
                print("------------------- Rule 2 results -------------------\n\n")

                println("Optimal x found (rule 2):")
                display(solver_rule2.best_x)
                print("\n")

                println("Optimal λ found (rule 2):")
                display(solver_rule2.best_λ)
                print("\n")

                println("Best value of lagrangian at iteration $(solver_rule2.best_iteration) (rule 2):")
                display(solver_rule2.best_lagrangian)
                print("\n")

                println("Duality gap between f( x* ) and ϕ( λ ) (rule 2):")

                dual_gap = jump_sol.opt_val - solver_rule2.best_lagrangian

                display(dual_gap)
                print("\n")

                gaps["Rule 2"] = dual_gap
                L_values["Rule 2"] = solver_rule2.best_lagrangian

                #------------------------------------------------------#
                #-----------     Results for rule 3     ---------------#
                #------------------------------------------------------#

                print("\n\n\n\n")
                print("------------------- Rule 3 results -------------------\n\n")


                println("Optimal x found (rule 3):")
                display(solver_rule3.best_x)
                print("\n")

                println("Optimal λ found (rule 3):")
                display(solver_rule3.best_λ)
                print("\n")

                println("Best value of lagrangian at iteration $(solver_rule3.best_iteration) (rule 3):")
                display(solver_rule3.best_lagrangian)
                print("\n")

                println("Duality gap between f( x* ) and ϕ( λ ) (rule 3):")

                dual_gap = jump_sol.opt_val - solver_rule3.best_lagrangian

                display(dual_gap)
                print("\n")

                gaps["Rule 3"] = dual_gap
                L_values["Rule 3"] = solver_rule3.best_lagrangian

                print("\n\n")

                println("Gaps:")
                display(gaps)
                print("\n")
                println("Lagrangian values:")
                display(L_values)

                #------------------------------------------------------#
                #-----------     Plotting utilities     ---------------#
                #------------------------------------------------------#

                # Plots.theme(:bright)

                gr()

                solvers = [ solver_rule1, solver_rule2, solver_rule3 ]

                for i=1:1:3

                    y = ones(3) 
                    title = Plots.scatter(y, marker=0,markeralpha=0, annotations=(2, y[2], Plots.text("Update rule $(solvers[i].update_formula), n=$(solvers[i].n) and K=$(solvers[i].K)", pointsize = 12)), axis=false, fillcolor=:white, grid=false, background_color=:white,background_color_subplot=:white, framestyle=:none, leg=false,size=(200,100),foreground_color=:white)

                    ticks = range( minimum(solvers[i].relaxation_values), maximum(solvers[i].relaxation_values), length = 5 )
                    ticks_string = [ string(round(el, digits = 2)) for el in ticks ]

                    plt = plot( solvers[i].num_iterations, 
                                solvers[i].relaxation_values, 
                                # title = "Lagrangian value update=$(solvers[i].update_formula)", 
                                titlefontsize = 12,
                                label = "Lagrangian", 
                                lw = 2,
                                dpi = 360,
                                linealpha = 0.5,
                                linecolor = :green,
                                bg_inside = :whitesmoke,
                                minorgrid = true,
                                minorgridalpha = 1,
                                foreground_color_grid = :white,
                                foreground_color_minor_grid = :white,
                                gridlinewidth = 1,
                                tickdirection = :out,
                                xscale = :log10,
                                minorticks = 5,
                                showaxis = :hide,
                                legend = :bottomright,
                                isempty(ticks) ? yticks = () : yticks = ( ticks, ticks_string ),
                                tickfontsize = 4,
                                guidefontsize = 6,
                                formatter = :plain )
                    xlabel!("Iterations")
                    ylabel!("Lagrangian value")
                    # display(plt)

                    ticks = range( minimum(solvers[i].gaps), maximum(solvers[i].gaps), length = 5 )
                    ticks_string = [ string(round(el, digits = 2)) for el in ticks ]

                    plt2 = plot(solvers[i].num_iterations, 
                                solvers[i].gaps, 
                                # title = "Gaps update=$(solvers[i].update_formula)", 
                                titlefontsize = 12,
                                label = "Gap", 
                                lw = 2,
                                dpi = 360,
                                xscale = :log10,
                                # yscale = :log10,
                                linealpha = 0.5,
                                linecolor = :green,
                                bg_inside = :whitesmoke,
                                minorgrid = true,
                                minorgridalpha = 1,
                                foreground_color_grid = :white,
                                foreground_color_minor_grid = :white,
                                gridlinewidth = 1,
                                tickdirection = :out,
                                minorticks = 5,
                                showaxis = :hide,
                                isempty(ticks) ? yticks = () : yticks = ( ticks, ticks_string ),
                                tickfontsize = 4,
                                guidefontsize = 6,
                                formatter = :plain )
                    xlabel!("Iterations")
                    ylabel!("Gap ϕ(λ)-f(x*)")
                    # display(plt2)

                    plt3 = plot(solvers[i].num_iterations, 
                                replace!(val -> val <= 0 ? 1e-8 : val, solvers[i].λ_distances), 
                                # title = "Residual λ update=$(solvers[i].update_formula)", 
                                titlefontsize = 12,
                                label = "Residual λ", 
                                lw = 2, 
                                dpi = 360,
                                xscale = :log10,
                                yscale = :log10,
                                linealpha = 0.5,
                                linecolor = :green,
                                bg_inside = :whitesmoke,
                                minorgrid = true,
                                minorgridalpha = 1,
                                foreground_color_grid = :white,
                                foreground_color_minor_grid = :white,
                                gridlinewidth = 1,
                                tickdirection = :out,
                                minorticks = 5,
                                showaxis = :hide,
                                tickfontsize = 4,
                                guidefontsize = 6,
                                formatter = :plain )
                    xlabel!("Iterations")
                    ylabel!("Residual λ")
                    # display(plt3)

                    plt4 = plot(solvers[i].num_iterations, 
                                solvers[i].x_distances, 
                                # title = "Residual x update=$(solvers[i].update_formula)", 
                                titlefontsize = 12,
                                label = "Residual x", 
                                lw = 2, 
                                dpi = 360,
                                xscale = :log10,
                                yscale = :log10,
                                linealpha = 0.5,
                                linecolor = :green,
                                bg_inside = :whitesmoke,
                                minorgrid = true,
                                minorgridalpha = 1,
                                foreground_color_grid = :white,
                                foreground_color_minor_grid = :white,
                                gridlinewidth = 1,
                                tickdirection = :out,
                                minorticks = 5,
                                showaxis = :hide,
                                tickfontsize = 4,
                                guidefontsize = 6,
                                formatter = :plain )
                    xlabel!("Iterations")
                    ylabel!("Residual x")
                    # display(plt4)

                    # savefig(plt, "plots/Convergence_rule=$(solvers[i].update_formula)_n=$(solvers[i].n)_K=$(solvers[i].K)_defl=$(solvers[i].deflection)_gap=$(round(gaps["Rule $i"], digits=2)).png")
                    # savefig(plt2, "plots/Gaps_rule=$(solvers[i].update_formula)_n=$(solvers[i].n)_K=$(solvers[i].K)_defl=$(solvers[i].deflection).png")
                    # savefig(plt3, "plots/Residual_rule=$(solvers[i].update_formula)_n=$(solvers[i].n)_K=$(solvers[i].K)_defl=$(solvers[i].deflection).png")
                    # savefig(plt4, "plots/Residual_x_rule=$(solvers[i].update_formula)_n=$(solvers[i].n)_K=$(solvers[i].K)_defl=$(solvers[i].deflection).png")
                    
                    # total_plot = plot(plt, plt2, plt3, plt4, 
                    #                 layout = @layout([a b; c d]), 
                    #                 title = "Update $(solvers[i].update_formula)",
                    #                 dpi = 360)
                    
                    # savefig(total_plot, "plots/Subplot=$(solvers[i].update_formula)_n=$(solvers[i].n)_K=$(solvers[i].K).png")
                    

                    # combine the 'title' plot with your real plots
                    total_plot = Plots.plot(
                        title,
                        Plots.plot(plt, plt2, plt3, plt4, layout = 4),
                        layout=grid(2,1,heights=[0.05,0.95])
                    )
                    savefig(total_plot, "plots/Rule=$(solvers[i].update_formula)_n=$(solvers[i].n)_K=$(solvers[i].K)_defl=$(solvers[i].deflection).png")

                end

            end

        end

    end
   
end


#------------------------------------------------------#
#---------     Initialize all parameters    -----------#
#------------------------------------------------------#

print("Input n values separated by comma: ")
n_list_str = readline()
n_list_str = split(n_list_str, ",")

n_list = []
for item in n_list_str
    push!(n_list, parse(Int64, item))
end

K_list = []
for n in n_list
    print("Enter K values separated by comma for n=$(n): ")
    Ks_str = readline()
    Ks = split(Ks_str, ",")
    K_s = []
    for item in Ks
        push!(K_s, parse(Int64, item))
    end
    push!(K_list, K_s)
end 

display(n_list)
display(K_list)

print("Deflection?[y/n] ")
deflections = readline()
deflections = deflections == "y" ? [true, false] : [false]

testing(n_list, K_list, deflections)