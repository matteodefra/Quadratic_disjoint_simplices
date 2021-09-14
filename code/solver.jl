using Convex, SCS
using Random
using LinearAlgebra
using ForwardDiff
# using JuMP
# using OSQP

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
print("\n\n")

# Initialize x iterates to zero
x = randn((n,1))
# x = ones((n,1))

# for set in I_K
#     for item in set
#         x[item] = 1 / length(set)
#     end
# end

println("Starting x:")
display(x)
print("\n\n")

# Initialize lambda iterates to zero
lambda = abs.(randn((n,1)))

println("Starting lambdas:")
display(lambda)
print("\n\n")

# Create random matrix A
A_help = randn(Float64 ,n, n)

# Multiply A with its transpose in order always positive definite condition
Q = A_help' * A_help

println("Q matrix:")
display(Q)
print("\n\n")

# Initialize also random q vector 
q = randn(Float64, (n,1))

println("q vector:")
display(q)
print("\n\n")

# Initialize eta
eta = 1

# Initialize delta
delta = rand()

# Initialize max_iter
max_iter = 60

# Initialize epsilon
epsilon = 1e-2

# Create struct solver to approach the problem
mutable struct Solver
    n :: Int
    iteration :: Int
    lambda :: Array{Float64}
    K :: Int
    I_K :: Vector{Array{Int64}}
    x :: Array{Float64}
    grads :: Array{Float64}
    G_t :: Matrix{Float64}
    Q :: Matrix{Float64}
    Full_mat :: Matrix{Float64}
    q :: Array{Float64}
    eta :: Float64
    delta :: Float64
    max_iter :: Int
    A :: Array{Int64}
    epsilon :: Float64
    num_iterations :: Vector{Float64}
    relaxation_values :: Vector{Float64}
    x_values :: Array{Float64}
    lambda_values :: Array{Float64}
    Solver() = new()
end

solver = Solver()

solver.n = n

solver.iteration = 0

solver.lambda = lambda

solver.K = K

solver.I_K = I_K

solver.x = x

solver.grads = Array{Float64}(undef,n,0)

solver.G_t = Matrix{Float64}(undef, solver.n, solver.n)

solver.Q = Q

solver.q = q

solver.eta = eta

solver.delta = delta

solver.max_iter = max_iter

solver.epsilon = epsilon

solver.num_iterations = Vector{Float64}()

solver.relaxation_values = Vector{Float64}()

solver.x_values = Array{Float64}(undef,n,0)

solver.lambda_values = Array{Float64}(undef,n,0)

#= 
Create matrix of KKT conditions
    Q A^T
    A 0
to solve the lagrangian_relaxation
=#
function construct_full_matrix(solver)
    Full_mat = vcat(solver.Q, solver.A)
    
    zero_mat = zeros((solver.K, solver.K))

    Right_side = vcat(solver.A', zero_mat)

    Full_mat = hcat(Full_mat, Right_side)

    return Full_mat
end


# Helper function to construct the matrix A
function construct_A(solver)
    A = Array{Int64}(undef,solver.K,solver.n)
    for k in 1:solver.K
        a_k = zeros(Int64, solver.n)
        for i in 1:solver.n
            if i in solver.I_K[k]
                a_k[i] = 1
            end 
        end
        A[k,:] = a_k
    end 
    return A
end

solver.A = construct_A(solver)

println("A constraint matrix:")
display(solver.A)
print("\n\n")

solver.Full_mat = construct_full_matrix(solver)

println("Full matrix:")
display(solver.Full_mat)
print("\n\n")

# function primal_function(solver, actual_x)
#     return actual_x' * solver.Q * actual_x .+ solver.q' * x
# end

# Compute function value
# f_val = primal_function(solver, previous_x)

# println("Primal function value: $f_val")
# print("\n\n")

function lagrangian_relaxation(solver, previous_x, previous_lambda)
    return previous_x' * solver.Q * previous_x .+ solver.q' * previous_x .- previous_lambda' * previous_x
end

#=
    Solve the problem of 
        x_t = \argmin_{x \in X} \{ x^T * Q * x + q * x - λ_{t-1} * x \}
    where X is the constraint set of the disjoint simplices (eliding non negativity 
    constraint, since it's included in the relaxation).
    Being only linear constraint, this problem can be easily solved in O(n) time,
    using the QR factorization of Q through backsubstitution

    Backward substitution is implemented to solve the linear system
        R [x, μ] = Q^T [λ_t - q, b]
    using the QR factorization of the KKT condition matrix.

    Returns a tuple with the value x and μ
=#
function solve_lagrangian_relaxation(solver)

    # Create vector [λ_{t-1} - q, b]
    o = ones((solver.K,1))

    diff = solver.lambda - solver.q

    b = vcat(diff, o)

    b = solver.Q_hat' .* b

    R_hat = solver.R_hat

    # First compute Q^t * known vector (O(n^2) float ops) 

    dim = solver.n + solver.K

    x_mu = zeros((dim,1))
    x_mu[dim] = b[dim]/R_hat[dim, dim]
    for i = dim-1:-1:1
        s = sum( R_hat[i,j]*x_mu[j] for j=i+1:dim )
        x_mu[i] = ( b[i] - s ) / R_hat[i,i]
    end
    
    return x_mu[1:solver.n] , x_mu[solver.n + 1 : dim]
end


#= 

    Compute one among the three possible update_rule specified in the report.
    
    The first update rule is a general update rule given by:

        λ_t = λ_{t-1} + η \diag(G_t)^{-1/2} g_t

    where G_t is the full outer product of all the stored subgradient

    The second update rule is:

        λ_t = - H_{t-1}^{-1} t η g_t 

    The third one employ the following:

        λ_t = λ_{t-1} + H_{t-1}^{-1} \[ Ψ_t(λ_t) - η g_t \]

=#
function compute_update_rule(solver, update_rule, H_t, Psi)
    
    if update_rule == 1

        # full outer product of gradients
        for col in eachcol(solver.grads)

            solver.G_t .+= col .* col'

        end

        G_t = Diagonal(solver.G_t)^(-0.5)

        lambda = solver.lambda + solver.eta * G_t * solver.grads[:,end]
        
    elseif update_rule == 2

        # Average of gradients
        avg_gradient = Vector{Float64}()

        for row in eachrow(solver.grads)

            total = sum(row)

            push!(avg_gradient, total / solver.iteration )
        
        end

        lambda = solver.iteration * solver.eta * (- H_t^(-1) * avg_gradient)

    else

        val = Psi .- (solver.eta .* solver.grads[:,end])

        update_part = H_t^(-1) * val

        lambda = solver.lambda + update_part

    end

    lambda = max.(0, lambda)

    return lambda

end


function check_duality_gap(Primals, Duals, solver)
    
    if isempty(Primals) || isempty(Duals)
        return false
    end

    duality_gap = (Primals[end] - Duals[end])[1]

    println("Duality gap:")
    display(duality_gap)
    print("\n\n")

    if abs(duality_gap) < solver.epsilon
        return true
    else
        return false
    end 

end


#=
    Check the other stopping condition, i.e. when
        \| λ_t - λ_{t-1} \|_2 ≤ ϵ 
    whenever this condition is met we have reached an optimal value of multipliers λ,
    hence we met complementary slackness and we can stop
=#
function check_lambda_norm(solver, current_lambda, previous_lambda)
    
    res = current_lambda .- previous_lambda

    distance = norm(res)

    println("Distance between lambda's")
    display(distance)
    print("\n\n")

    if distance <= solver.epsilon
        # We should exit the loop
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

    subgradient = gradient(solver.lambda)'

    return subgradient

end


#=
    Loop function which implements customized ADAGRAD algorithm. The code is the equivalent of Algorithm 3 
    presented in the report.
    Takes as parameters:
        @param solver: struct solver containing all the required data structures
        @param update_rule: an integer specifying what update rule to use for the lambda's
=#  
function my_ADAGRAD(solver, update_rule)

    solver.iteration = 1

    while solver.iteration < solver.max_iter
       
        previous_lambda = solver.lambda

        L_val = lagrangian_relaxation(solver, solver.x, solver.lambda)

        println("Lagrangian relaxation value: $L_val")
        print("\n\n")

        #= 
        Compute subgradient of \psi (check report for derivation)
        The subgradient is given by the derivative of the Lagrangian function w.r.t. λ
            
            ∇_λ (x_{star}) ( x_{star} * Q * x_{star} + q * x_{star} - λ * x_{star} )
        
        where the given x_{star} is computed at the previous step. As a consequence the L function
        is given by  
        
            ∇_λ (x_{star}) ( λ * x_{star} + constant )
            
        which is always differentiable since it is an hyperplane
        =#
        subgrad = get_grad(solver)
        # subgrad = solver.x


        # Store subgradient in matrix
        solver.grads = [solver.grads subgrad]
        
        # Solution of lagrangian_relaxation of problem 2.4 (see report)
        s_t = Vector{Float64}()

        for row in eachrow(solver.grads)
            sum = 0
            for item in row
                sum += item^2
            end
            push!(s_t, sqrt(sum))
        end

        # Create diagonal matrix H_t
        # Create diagonal matrix starting from s_t 
        mat = Diagonal(s_t)

        # Construct Identity matrix
        Iden = Diagonal(ones(solver.n,solver.n))

        delta_Id = solver.delta .* Iden

        # Sum them (hessian approximation)
        H_t = delta_Id + mat

        #= 
        Proximal term computation: dot(x,A,y) faster way of computing 
            \left\langle x, A*y \right\rangle
        the result A*y is not stored and this allow for a memory saving
        =#   
        Psi = 0.5 .* dot(previous_lambda, H_t, previous_lambda)

        o = ones((solver.K,1))

        diff = previous_lambda - solver.q

        b = vcat(diff, o)

        x_mu = solver.Full_mat \ b 

        solver.x, mu = x_mu[1:solver.n] , x_mu[solver.n + 1 : solver.n + solver.K] #solve_lagrangian_relaxation(solver)

        if all(x -> x > 0, solver.x)
            println("x value:")
            display(solver.x)
            print("\n\n")
        end
    
        #=
        Compute the update rule for the lagrangian multipliers lambda: can use 
        one among the three showed, then soft threshold the result
        =#
        solver.lambda = compute_update_rule(solver ,update_rule, H_t, Psi)

        # Storing current iteration
        push!(solver.num_iterations, solver.iteration)

        # Storing current relaxation value
        push!(solver.relaxation_values, L_val[1])

        # Storing x_{solver.iteration}
        solver.x_values = [solver.x_values solver.x]

        # Storing λ_{solver.iteration}
        solver.lambda_values = [solver.lambda_values solver.lambda]

        if check_lambda_norm(solver, solver.lambda, previous_lambda)
            println("Optimal lambda found")
            break
        end

        solver.iteration += 1

        push!(solver.num_iterations, solver.iteration)

        solver.eta = 1 / solver.iteration

    end

    return solver.x, solver.lambda

end


optimal_x, optimal_lambda = my_ADAGRAD(solver, 1)

println("Lagrangians")
display(solver.relaxation_values)
print("\n")
println("Lambda vals")
display(solver.lambda_values)
print("\n")

println("Optimal x found:")
display(optimal_x)
print("\n\n")

println("Optimal lambda found:")
display(optimal_lambda)
print("\n\n")

optimal_dual = lagrangian_relaxation(solver, optimal_x, optimal_lambda)

println("Value of the lagrangian function:")
display(optimal_dual)
print("\n\n\n\n\n\n")



println("Comparing with Convex.jl solution")

x = Variable(n)

problem = minimize( quadform(x, Q) + dot(q, x) )

for row in eachrow(solver.A)
    problem.constraints += [row' * x == 1]
end

problem.constraints += [x >= 0]

println(problem)

solve!(problem, () -> SCS.Optimizer(verbose=true), verbose=false)

println("problem status is ", problem.status) # :Optimal, :Infeasible, :Unbounded etc.
println("optimal value is ", problem.optval)

println("Primal variables of problem:")
display(x)
print("\n\n")

println("Dual variables constraints")
for constraint in problem.constraints
    display(constraint.dual)
    print("\n")
end


println("Duality gap between f( x* ) of Convex.jl and psi( lambda )")

dual_gap = problem.optval[1] - optimal_dual[1]

display(dual_gap)
print("\n")