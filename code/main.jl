include("./ADAGRAD_Solver.jl")
include("./Utils.jl")
include("./ConvexSolution.jl")

using LinearAlgebra
using Random
using Convex
using .Utils
using .ADAGRAD_Solver
# Compute the solution of Convex.jl
using .ConvexSolution


#------------------------------------------------------#
#---------     Initialize all parameters    -----------#
#------------------------------------------------------#

print("Enter desired dimension n: ")
n = readline()
n = parse(Int64, n)

print("Enter K value of disjoint simplices (K le n): ")
K = readline()
K = parse(Int64, K)

# Simple utility loop to check K value 
while K > n
    print("K greater than n, input correct value: ")
    global K 
    K = readline()
    K = parse(Int64, K)
end

println("Value of K is correct")

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

println("Set of I_K arrays:")
display(I_K)
print("\n")

# Initialize x iterates to zero
x = abs.(randn((n,1)))

# # Feasible x:
# for set in I_K
#     for val in set
#         x[val] = 1 / length(set)
#     end
# end

println("Starting x:")
display(x)
print("\n")

# Initialize λ iterates to zero
λ = abs.(randn((n,1)))

println("Starting λs:")
display(λ)
print("\n")

# Create random matrix A_help
A_help = randn(Float64 ,n, n)

# Multiply A with its transpose in order to have always positive semidefinite condition
Q = A_help' * A_help
# Q = A_help

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
max_iter = 10000

# Initialize ϵ
ϵ = 1e-5

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
# global F

# try 
#     global F = cholesky!(Hermitian(Full_mat))
# catch y
#     if isa(y, PosDefException)
#         println("Matrix is not positive definite")
#     end
# end

# Lu factorization (unless Full_mat is symmetric)
F = factorize(Full_mat)

println("Factorization:")
display(F)
print("\n")

#------------------------------------------------------#
#----    Use Convex.jl to compute primal solution   ---#
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

#------------------------------------------------------#
#----------     Create problem structs     ------------#
#------------------------------------------------------#


# Create three different struct to exploit the three update rule
solver_rule1 = ADAGRAD_Solver.Solver(
    n, 0, λ, K, I_K, x, Array{Float64}(undef, n, 0), # grads 
    Diagonal(Matrix{Float64}(undef, n, n)), # G_t 
    Array{Float64}(undef, n, 1), # s_t
    Array{Float64}(undef, n, 1), # avg_gradient
    Q, q, η, δ, max_iter, ϵ, 
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
    convex_sol.opt_val # Primal optimal value
)

solver_rule2 = ADAGRAD_Solver.Solver(
    n, 0, λ, K, I_K, x, Array{Float64}(undef, n, 0), # grads 
    Diagonal(Matrix{Float64}(undef, n, n)), # G_t 
    Array{Float64}(undef, n, 1), # s_t
    Array{Float64}(undef, n, 1), # avg_gradient
    Q, q, η, δ, max_iter, ϵ, 
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
    convex_sol.opt_val # Primal optimal value
)

solver_rule3 = ADAGRAD_Solver.Solver(
    n, 0, λ, K, I_K, x, Array{Float64}(undef, n, 0), # grads 
    Diagonal(Matrix{Float64}(undef, n, n)), # G_t 
    Array{Float64}(undef, n, 1), # s_t
    Array{Float64}(undef, n, 1), # avg_gradient
    Q, q, η, δ, max_iter, ϵ, 
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
    convex_sol.opt_val # Primal optimal value
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
display(solver_rule1.x)
print("\n")

println("Optimal λ found (rule 1):")
display(solver_rule1.λ)
print("\n")

optimal_dual = ADAGRAD_Solver.lagrangian_relaxation(solver_rule1, solver_rule1.x, solver_rule1.λ)[1]

println("Value of the lagrangian function (rule 1):")
display(optimal_dual)
print("\n")

println("Duality gap between f( x* ) of Convex.jl and ϕ( λ ) (rule 1):")

dual_gap = convex_sol.opt_val - optimal_dual

display(dual_gap)
print("\n")

gaps["Rule 1"] = dual_gap
L_values["Rule 1"] = optimal_dual

#------------------------------------------------------#
#-----------     Results for rule 2     ---------------#
#------------------------------------------------------#

print("\n\n\n\n")
print("------------------- Rule 2 results -------------------\n\n")

println("Optimal x found (rule 2):")
display(solver_rule2.x)
print("\n")

println("Optimal λ found (rule 2):")
display(solver_rule2.λ)
print("\n")

optimal_dual = ADAGRAD_Solver.lagrangian_relaxation(solver_rule2, solver_rule2.x, solver_rule2.λ)[1]

println("Value of the lagrangian function (rule 2):")
display(optimal_dual)
print("\n")

println("Duality gap between f( x* ) of Convex.jl and ϕ( λ ) (rule 2):")

dual_gap = convex_sol.opt_val - optimal_dual

display(dual_gap)
print("\n")

gaps["Rule 2"] = dual_gap
L_values["Rule 2"] = optimal_dual

#------------------------------------------------------#
#-----------     Results for rule 3     ---------------#
#------------------------------------------------------#

print("\n\n\n\n")
print("------------------- Rule 3 results -------------------\n\n")


println("Optimal x found (rule 3):")
display(solver_rule3.x)
print("\n")

println("Optimal λ found (rule 3):")
display(solver_rule3.λ)
print("\n")

optimal_dual = ADAGRAD_Solver.lagrangian_relaxation(solver_rule3, solver_rule3.x, solver_rule3.λ)[1]

println("Value of the lagrangian function (rule 3):")
display(optimal_dual)
print("\n")

println("Duality gap between f( x* ) of Convex.jl and ϕ( λ ) (rule 3):")

dual_gap = convex_sol.opt_val - optimal_dual

display(dual_gap)
print("\n")

gaps["Rule 3"] = dual_gap
L_values["Rule 3"] = optimal_dual

print("\n\n")

println("Gaps:")
display(gaps)
print("\n")
println("Lagrangian values:")
display(L_values)

#------------------------------------------------------#
#-----------     Plotting utilities     ---------------#
#------------------------------------------------------#

# using DataFrames
using Plots

Plots.theme(:default)

plotlyjs(size=(600,450))

solvers = [ solver_rule1, solver_rule2, solver_rule3 ]

for i=1:1:3

    plt = plot( solvers[i].num_iterations, 
                solvers[i].relaxation_values, 
                title="Lagrangian value update=$(solvers[i].update_formula)", 
                label="Convergence", 
                lw=2,
                xaxis=:log )
    xlabel!("Iterations")
    ylabel!("Lagrangian value")
    display(plt)
  

    plt2 = plot(solvers[i].num_iterations, 
                solvers[i].λ_distances, 
                title="Residual λ update=$(solvers[i].update_formula)", 
                label="Residual λ", 
                lw=2, 
                xaxis=:log,
                yaxis=:log )
    xlabel!("Iterations")
    ylabel!("Residual λ")
    display(plt2)
  
    plt3 = plot(solvers[i].num_iterations, 
                solvers[i].gaps, 
                title="Gaps update=$(solvers[i].update_formula)", 
                label="Gap", 
                lw=2,
                xaxis=:log )
    xlabel!("Iterations")
    ylabel!("Gap ϕ(λ)-f(x^*)")
    display(plt3)

    # savefig(plt, "plots/Convergence_rule=$(solvers[i].update_formula)_n=$(solvers[i].n)_K=$(solvers[i].K)_gap=$(round(gaps["Rule $i"], digits=3)).png")
    # savefig(plt2, "plots/Residual_rule=$(solvers[i].update_formula)_n=$(solvers[i].n)_K=$(solvers[i].K).png")
    # savefig(plt3, "plots/Gaps_rule=$(solvers[i].update_formula)_n=$(solvers[i].n)_K=$(solvers[i].K).png")

end
   