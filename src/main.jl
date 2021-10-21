include("./ADAGRAD_Solver.jl")
include("./Utils.jl")
include("./ConvexSolution.jl")

using LinearAlgebra
using Random
using Convex
using MAT
using .Utils
using .ADAGRAD_Solver
using .ConvexSolution

function testing(n, K, deflections, Q, q, λ, μ, x, I_K, η, δ, max_iter, ε, τ, stepsizes, F, A, primal_optimal)
    #------------------------------------------------------#
    #----------     Create problem structs     ------------#
    #------------------------------------------------------#

    Iden = ones((n,1))

    for deflection in deflections

        for stepsize in stepsizes

            for update_rule in [3]

                # Create three different struct to exploit the three update rule
                sol = ADAGRAD_Solver.Solver(
                    n, 0, λ, μ, K, I_K, x, Array{Float64}(undef, n, 0), # grads 
                    fill(0.0, n), # G_t 
                    fill(0.0, n), # s_t
                    δ .* Iden, # H_t
                    fill(0.0, n), # d_i
                    deflection, # deflection
                    Q, q, η, δ, max_iter, ε, τ, -Inf, # best_dual
                    0, # best_iteration
                    x, # best_x
                    λ, # best_λ
                    Vector{Float64}(), # num_iterations
                    Vector{Float64}(), # dual_values
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
                sol = @time ADAGRAD_Solver.my_ADAGRAD(sol)

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

            end

        end

    end

end



#------------------------------------------------------#
#---------     Initialize all parameters    -----------#
#------------------------------------------------------#
print("Use stored .mat?[y/n] ")
y = readline()

vars = y == "y" ? matread("mat/structs.mat") : []

print("Input n value: ")
n = y == "y" ? length(vars["q"]) : parse(Int64, readline())

print("Enter K value for n=$(n): ")
K = y == "y" ? size(vars["A"],1) : parse(Int64, readline())

print("Deflection?[y/n] ")
# deflection = readline()
deflections = isequal("y",readline()) ? [true, false] : [false]

print("Use different stepsize choices?[y/n] ")
# stepsizes = readline()
#=
    0: Constant step size                   η = h               with h > 0
    1: Constant step length                 η = h / ∥ g_k ∥_2   with h = ∥ λ_{t+1} - λ_t ∥_2 
    2: Square summable but not summable     η = α / (b + t)     with α > 0 and b ≥ 0
    3: Nonsummable diminishing              η = α / √t          with α > 0 
    4: Optimal                              η = f(x*) - ϕ(λ_t) / ∥ g_k ∥^2
=#
stepsizes = isequal("y",readline()) ? [0, 1, 2, 3, 4] : [0]

println("Initializing random disjoint sets")

# Indexes to access the different x_i
const indexes = collect(1:1:n)

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
I_K = y == "y" ? vars["I"] : initialize_sets(indexes, K, n)

println("Set of I_K arrays:")
display(I_K)
print("\n")

# Initialize x iterates to feasible solution
# const x = ones((n,1))

# # Feasible x
# for set in I_K
#     for val in set
#         x[val] = 1/length(set)
#     end
# end

# Initialize λ iterates to ones
λ = ones((n,1))

# Initialize μ iterates to zeros
μ = zeros((K,1))

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

# Initialize η
const η = 1

# Initialize δ
δ = 1e-2#abs(rand())

# Initialize max_iter
const max_iter = 100000

# Initialize ε
const ε = 1e-14

# Initialize τ
const τ = 1e-7

A = y == "y" ? vars["A"] : Utils.construct_A(K, n, I_K)
# A = y == "y" ? vars["A"] : construct_A(K, n, I_K)

println("A constraint matrix:")
display(A)
print("\n")

Full_mat = Utils.construct_full_matrix(Q, A, K)
# Full_mat = [ Q ; A ]

# part = [ A' ; zeros((K,K)) ]

# Full_mat = Symmetric([ Full_mat part ])

println("Full matrix:")
display(Full_mat)
print("\n")

#= 
    Let Julia automatically determine the best factorization:
        
        Cholesky ( if Full_mat ≻ 0 )
        Bunch-Kaufman ( if Q=Q^T )
        pivoted LU ( otherwise )
=#
# const F, pivoting, info = LAPACK.getrf!(Full_mat)
F = lu!(Full_mat)
# F = factorize(Full_mat)
# F = cholesky(Full_mat)

println("Factorization:")
display(F)
print("\n")

GC.gc()

x = (F \ vcat(λ - q, ones((K,1))))[1:n]

println("Starting x:")
display(x)
print("\n")

println("Starting λs:")
display(λ)
print("\n")

matwrite("mat/structs.mat", Dict(
    "Q" => Q,
    "A" => A,
    "q" => q,
    "I" => I_K
);compress = true) 

#------------------------------------------------------#
#-----    Use Convex.jl to compute primal solution   ----#
#------------------------------------------------------#

# convex_sol = ConvexSolution.ConvexSol(
#     n,
#     K, 
#     Variable(n),
#     A,
#     Q,
#     q
# )

# convex_sol = ConvexSolution.compute_solution(convex_sol)

# println("Convex.jl optimal value:")
# display(convex_sol.opt_val)
# print("\n")

opt_val = 1.202577640305848e+05


# For the current structs.mat
# const yalmip_optval = 93592.0938

testing(n, K, deflections, Q, q, λ, μ, x, I_K, η, δ, 
        max_iter, ε, τ, stepsizes, F, A, opt_val)
