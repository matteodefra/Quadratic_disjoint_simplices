include("./ADAGRAD_Solver.jl")
include("./Utils.jl")
include("./ConvexSolution.jl")

using LinearAlgebra
using Random
using Plots
using Colors
using Convex
using MAT
using .Utils
using .ADAGRAD_Solver
using .ConvexSolution


function testing(n, K, deflections, Q, q, λ, x, I_K, η, δ, max_iter, ε, τ, stepsizes, Full_mat, F, A, primal_optimal)
    #------------------------------------------------------#
    #----------     Create problem structs     ------------#
    #------------------------------------------------------#

    Iden = ones((n,1))

    for deflection in deflections

        for stepsize in stepsizes

            for update_rule in [1,2,3]

                # Create three different struct to exploit the three update rule
                sol = ADAGRAD_Solver.Solver(
                    n, 0, λ, K, I_K, x, Array{Float64}(undef, n, 0), # grads 
                    #Diagonal(Matrix{Float64}(undef, n, n)), # G_t 
                    Array{Float64}(undef, n, 1), # G_t
                    Array{Float64}(undef, n, 1), # s_t
                    δ .* Iden, # H_t
                    Array{Float64}(undef, n, 1), # avg_gradient
                    Array{Float64}(undef, n, 1), # d_i
                    deflection, # deflection
                    Q, q, η, δ, max_iter, ε, τ, -Inf, # best_dual
                    0, # best_iteration
                    x, # best_x
                    λ, # best_λ
                    Vector{Float64}(), # num_iterations
                    Vector{Float64}(), # dual_values
                    Array{Float64}(undef, n, 0), # x_values
                    Array{Float64}(undef, n, 0), # λ_values
                    Vector{Float64}(), # λ_distances
                    Vector{Float64}(), # x_distances
                    Vector{Float64}(), # timings
                    Vector{Float64}(), # gaps
                    update_rule, # update_rule
                    stepsize, # stepsize_choice
                    F, # Best factorization
                    A, # Constraint matrix
                    primal_optimal # Primal optimal value
                )

                #------------------------------------------------------#
                #--------     Calculate custom solution     -----------#
                #------------------------------------------------------#


                # Now calculate the results of ADAGRAD in the three different fashion way
                sol = @time ADAGRAD_Solver.my_ADAGRAD(sol, Full_mat)

                #------------------------------------------------------#
                #------------     Results for rules     ---------------#
                #------------------------------------------------------#

                print("\n\n\n\n")
                print("------------------- Rule $(sol.update_formula) results -------------------\n\n")

                println("Optimal x found (rule $(sol.update_formula)):")
                display(sol.best_x)
                print("\n")

                println("Optimal λ found (rule $(sol.update_formula)):")
                display(sol.best_λ)
                print("\n")

                println("Best value of dual function at iteration $(sol.best_iteration) (rule $(sol.update_formula)):")
                display(sol.best_dual)
                print("\n")

                println("Duality gap between f( x* ) and ϕ( λ ) (rule $(sol.update_formula)):")

                dual_gap = primal_optimal - sol.best_dual

                display(dual_gap)
                print("\n")

                matwrite("mat/matsolution_rule=$(sol.update_formula)_defl=$(deflection)_step=$(stepsize).mat", Dict(
                    "Q" => sol.Q,
                    "q" => sol.q, 
                    "x" => sol.best_x,
                    "lambda" => sol.best_λ
                ); compress = true)

                #------------------------------------------------------#
                #-----------     Plotting utilities     ---------------#
                #------------------------------------------------------#

                # Plots.theme(:bright)

                gr()

                if sol.iteration == 1
                    break
                end

                y = ones(3) 
                title = Plots.scatter(y, marker=0,markeralpha=0, annotations=(2, y[2], Plots.text("Update rule $(sol.update_formula), n=$(sol.n) and K=$(sol.K)", pointsize = 12)), axis=false, fillcolor=:white, grid=false, background_color=:white,background_color_subplot=:white, framestyle=:none, leg=false,size=(200,100),foreground_color=:white)

                ticks = []
                ticks_string = []

                if ( !isempty(sol.dual_values) )
                    ticks = range( minimum(sol.dual_values), maximum(sol.dual_values), length = 5 )
                    ticks_string = [ string(round(el, digits = 2)) for el in ticks ]
                end

                plt = plot( sol.num_iterations, 
                            sol.dual_values, 
                            titlefontsize = 12,
                            label = "Dual", 
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
                ylabel!("Dual value")

                ticks = []
                ticks_string = []

                if ( !isempty(sol.gaps) )
                    ticks = range( minimum(sol.gaps), maximum(sol.gaps), length = 5 )
                    ticks_string = [ string(round(el, digits = 2)) for el in ticks ]
                end

                    
                plt2 = plot(sol.num_iterations, 
                            sol.gaps, 
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
                ylabel!("Gap f(x*)-ϕ(λ)")

                plt3 = plot(sol.num_iterations, 
                            replace!(val -> val <= 0 ? 1e-8 : val, sol.λ_distances), 
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

                plt4 = plot(sol.num_iterations, 
                            sol.x_distances, 
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
               
                # combine the 'title' plot with your real plots
                total_plot = Plots.plot(
                    title,
                    Plots.plot(plt, plt2, plt3, plt4, layout = 4),
                    layout=grid(2,1,heights=[0.05,0.95])
                )
                savefig(total_plot, "plots/Rule=$(sol.update_formula)_n=$(sol.n)_K=$(sol.K)_defl=$(sol.deflection)_step=$(sol.stepsize_choice).png")

            end

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

print("Use different stepsize choices?[y/n] ")
stepsizes = readline()
#=
    0: Constant step size                   η = h               with h > 0
    1: Constant step length                 η = h / ∥ g_k ∥_2   with h = ∥ λ_{t+1} - λ_t ∥_2 
    2: Square summable but not summable     η = α / (b + t)     with α > 0 and b ≥ 0
    3: Nonsummable diminishing              η = α / √t          with α > 0 
=#
stepsizes = stepsizes == "y" ? [0, 1, 2, 3] : [2]

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
x = ones((n,1))

# Feasible x
for set in I_K
    for val in set
        x[val] = 1/length(set)
    end
end

# Initialize λ iterates to ones
λ = ones((n,1))

# Create random matrix A_help
A_help = rand(Float64, n, n)

Q = A_help' * A_help

println("Q matrix:")
display(Q)
print("\n")

# Initialize also random q vector 
q = rand(Float64, (n,1))

println("q vector:")
display(q)
print("\n")

# Initialize η
η = 1

# Initialize δ
δ = abs(rand())

# Initialize max_iter
max_iter = 1000

# Initialize ε
ε = 1e-8

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
F = factorize(Full_mat)

println("Factorization:")
display(F)
print("\n")

# Full_mat = nothing
A_help = nothing
GC.gc()

# x = (F \ vcat(λ - q, ones((K,1))))[1:n]

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

println("Starting x:")
display(x)
print("\n")

println("Starting λs:")
display(λ)
print("\n")

testing(n, K, deflections, Q, q, λ, x, I_K, η, δ, 
        max_iter, ε, τ, stepsizes, Full_mat, F, A, convex_sol.opt_val)
