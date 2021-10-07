include("./ADAGRAD_Solver.jl")
include("./Utils.jl")
include("./JuMPSolution.jl")
include("./ConvexSolution.jl")

using LinearAlgebra
using Random
using Plots
using Colors
using Convex
using .Utils
using .ADAGRAD_Solver
# Compute the solution of JuMP.jl
using .JuMPSolution
using .ConvexSolution


function testing(n, K, deflections, Q, q, λ, x, I_K, η, δ, max_iter, ε, τ, F, A, primal_optimal)
    #------------------------------------------------------#
    #----------     Create problem structs     ------------#
    #------------------------------------------------------#

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
            F, # Best factorization
            A, # Constraint matrix
            primal_optimal # Primal optimal value
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
            F, # Best factorization
            A, # Constraint matrix
            primal_optimal # Primal optimal value
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
            F, # Best factorization
            A, # Constraint matrix
            primal_optimal # Primal optimal value
        )

        #------------------------------------------------------#
        #--------     Calculate custom solution     -----------#
        #------------------------------------------------------#


        # Now calculate the results of ADAGRAD in the three different fashion way
        solver_rule1 = @time ADAGRAD_Solver.my_ADAGRAD(solver_rule1)

        solver_rule2 = @time ADAGRAD_Solver.my_ADAGRAD(solver_rule2)

        # solver_rule3 = @time ADAGRAD_Solver.my_ADAGRAD(solver_rule3)

        # solvers = [solver_rule1, solver_rule2, solver_rule3]
        solvers = [solver_rule1, solver_rule2]

        #------------------------------------------------------#
        #-----------     Results for rule 1     ---------------#
        #------------------------------------------------------#

        for sol in solvers
            print("\n\n\n\n")
            print("------------------- Rule $(sol.update_formula) results -------------------\n\n")

            println("Optimal x found (rule $(sol.update_formula)):")
            display(sol.best_x)
            print("\n")

            println("Optimal λ found (rule $(sol.update_formula)):")
            display(sol.best_λ)
            print("\n")

            println("Best value of lagrangian at iteration $(sol.best_iteration) (rule $(sol.update_formula)):")
            display(sol.best_lagrangian)
            print("\n")

            println("Duality gap between f( x* ) and ϕ( λ ) (rule $(sol.update_formula)):")

            dual_gap = primal_optimal - sol.best_lagrangian

            display(dual_gap)
            print("\n")

        end

        #------------------------------------------------------#
        #-----------     Plotting utilities     ---------------#
        #------------------------------------------------------#

        # Plots.theme(:bright)

        gr()

        for sol in solvers

            y = ones(3) 
            title = Plots.scatter(y, marker=0,markeralpha=0, annotations=(2, y[2], Plots.text("Update rule $(sol.update_formula), n=$(sol.n) and K=$(sol.K)", pointsize = 12)), axis=false, fillcolor=:white, grid=false, background_color=:white,background_color_subplot=:white, framestyle=:none, leg=false,size=(200,100),foreground_color=:white)

            ticks = range( minimum(sol.relaxation_values), maximum(sol.relaxation_values), length = 5 )
            ticks_string = [ string(round(el, digits = 2)) for el in ticks ]

            plt = plot( sol.num_iterations, 
                        sol.relaxation_values, 
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

            ticks = range( minimum(sol.gaps), maximum(sol.gaps), length = 5 )
            ticks_string = [ string(round(el, digits = 2)) for el in ticks ]

            plt2 = plot(sol.num_iterations, 
                        sol.gaps, 
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
                        formatter = :scientific )
            xlabel!("Iterations")
            ylabel!("Gap ϕ(λ)-f(x*)")
            # display(plt2)

            plt3 = plot(sol.num_iterations, 
                        replace!(val -> val <= 0 ? 1e-8 : val, sol.λ_distances), 
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

            plt4 = plot(sol.num_iterations, 
                        sol.x_distances, 
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
            savefig(total_plot, "plots/Rule=$(sol.update_formula)_n=$(sol.n)_K=$(sol.K)_defl=$(sol.deflection).png")

        end

    end

end



#------------------------------------------------------#
#---------     Initialize all parameters    -----------#
#------------------------------------------------------#

print("Input n value: ")
n = readline()
n = parse(Int64, n)

print("Enter K value for n=$(n): ")
K = readline()
K = parse(Int64, K)

print("Deflection?[y/n] ")
deflections = readline()
deflections = deflections == "y" ? [true, false] : [false]

println("Initializing random disjoint sets")

# Indexes to access the different x_i
indexes = collect(1:1:n)

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
I_K = initialize_sets(indexes, K, n)

indexes = nothing

println("Set of I_K arrays:")
display(I_K)
print("\n")

# Initialize x iterates to feasible solution
x = zeros((n,1))

# Feasible x
for set in I_K
    for val in set
        x[val] = 1/length(set)
    end
end

# Initialize λ iterates to ones
λ = ones((n,1))

# # Create random matrix A_help
A_help = randn(Float64 ,n, n)

# r = rank(A_help) # println("Matrix rank:") # display(r) # while r < n #     global A_help #     global r #     A_help = randn(Float64 ,n, n) #     r = rank(A_help) #     println("Matrix rank:") #     display(r) # end # Compute the inverse # A_help_inv = inv(A_help) #= Construct Q using a surely nonsingular matrix A_H using positive random eigenvalues. A random probability number is used to set an eigenvalue as 0 or not, to manage positive semidefinite or positive definite case. If prob > 0.5 ⟹ some given numbers of eigenvalues are set to 0 Else ⟹ Q is created as positive definite =# # # Eigenvalues of Q # vect = abs.(randn((n,1))) # # Random number used to set zeros # prob = abs(rand()) # if prob > 0.5 #     println("Prob greater than zero") #     # Random number of zeros to set #     global eigens = rand(1:n-1) #     # Loop over vect and modify eigenvalues #     global ind = 0 #     while eigens > 0 #         vect[end - ind] = 0.0 #         global ind #         global eigens #         ind += 1 #         eigens -= 1 #     end #     display(vect) # end # vect = vec(vect) # global Q = A_help * Diagonal(vect) * A_help_inv

Q = A_help' * A_help

# Garbage collector: free up some memory
# vect = nothing
A_help = nothing
# A_help_inv = nothing

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
ε = 1e-7

# Initialize τ
τ = 1e-4

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

Full_mat = nothing
GC.gc()

x = (F \ vcat(λ - q, ones((K,1))))[1:n]

#------------------------------------------------------#
#-----    Use JuMP.jl to compute primal solution   ----#
#------------------------------------------------------#

# jump_sol = JuMPSolution.JuMPSol(
#     n,
#     K,
#     A,
#     Q,
#     q
# )

# jump_sol = JuMPSolution.compute_solution(jump_sol)

# println("Optimal value:")
# display(jump_sol.opt_val)
# print("\n")


#------------------------------------------------------#
#-----    Use Convex.jl to compute primal solution   ----#
#------------------------------------------------------#

convex_sol = ConvexSolution.ConvexSol(
    n,
    Variable(n),
    A,
    Q,
    q
)

convex_sol = ConvexSolution.compute_solution(convex_sol)

println("Optimal value:")
display(convex_sol.opt_val)
print("\n")

val = Utils.compute_lagrangian(Q, q, x, λ)[1]

#= 
    Check that we are not violating the lagrangian:
    compute new random values for x and λ to start in a feasible solution
=#
# while (jump_sol.opt_val - val) < 10
#     global x
#     global λ
#     x = randn((n,1))
#     λ = abs.(randn((n,1)))
#     global val 
#     val = Utils.compute_lagrangian(Q, q, x, λ)[1]
#     display(val)
# end

# display(val)

println("Starting x:")
display(x)
print("\n")

println("Starting λs:")
display(λ)
print("\n")

testing(n, K, deflections, Q, q, λ, x, I_K, η, δ, 
        max_iter, ε, τ, F, A, convex_sol.opt_val)
