using Random
using LinearAlgebra
using AutoGrad

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
x = ones((n,1))

println("Empty x:")
display(x)
print("\n\n")

# Initialize lambda iterates to zero
lambda = ones((n,1))

println("Empty lambdas:")
display(lambda)
print("\n\n")

# Create random matrix A
A_help = randn(Float64 ,n, n)

# Multiply A with its transpose in order always positive definite condition
Q = A_help' * A_help

println("Q matrix:")
display(Q)
print("\n\n")

# Get QR factorization of Q
Q_hat, R_hat = qr(Q)

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
max_iter = 5

# Create struct solver to approach the problem
mutable struct Solver
    n :: Int
    lambda :: Array{Float64}
    previous_lambda :: Array{Float64}
    K :: Int
    I_K :: Vector{Array{Int64}}
    x :: Array{Float64}
    previous_x :: Array{Float64}
    grads :: Array{Float64}#Vector{Array{Int64}}
    Q :: Matrix{Float64}
    Q_hat :: Matrix{Float64}
    R_hat :: Matrix{Float64}
    q :: Array{Float64}
    eta :: Float64
    delta :: Float64
    max_iter :: Int
    A :: Array{Int64}
    Solver() = new()
end

solver = Solver()

solver.n = n

solver.lambda = lambda

solver.K = K

solver.I_K = I_K

solver.x = x

solver.grads = Array{Float64}(undef,n,0)

solver.Q = Q

solver.Q_hat = Q_hat

solver.R_hat = R_hat

solver.q = q

solver.eta = eta

solver.delta = delta

solver.max_iter = max_iter

# println("Solver struct:\n$solver")


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

function primal_function(solver, actual_x)
    return actual_x' * solver.Q * actual_x + solver.q' * x
end

function lagrangian_relaxation(solver, previous_x, previous_lambda)
    return previous_x' * solver.Q * previous_x + solver.q' * previous_x - previous_lambda' * previous_x
end


println("Primal function value:$(primal_function(solver,solver.x))")

println("Dual function value:$(lagrangian_relaxation(solver,solver.x, solver.lambda))")

# println("Params: $x_params")
# println("Params: $lambda_params")

# # Compute derivative of lagrangian_relaxation w.r.t. to lambda using AutoGrad
# g = @diff sum([x_params' * Q * x_params, q' * x_params, - lambda_params' * x_params])

# println("Overall gradient: $g")

# val = grad(g, lambda_params)

# println("Gradient w.r.t. lambda: $val")

# Inline function for squaring number
square(n) = n * n 


# Loop function which implements customized ADAGRAD algorithm
function my_ADAGRAD(solver)

    iter = 0

    previous_x = solver.x

    while iter < solver.max_iter #&& check_condition(solver.previous_lambda, solver.lambda)
        # Compute function value
        f_val = primal_function(solver, previous_x)

        # Compute subgradient of \psi (check report for derivation)
        subgrad = previous_x

        # Store subgradient in matrix
        solver.grads = [solver.grads subgrad]
        
        # Solution of lagrangian_relaxation of problem 2.4 (see report)
        s_t = Vector{Float64}()

        for (i,row) in enumerate(eachrow(solver.grads))
            sum = 0
            for item in row
                sum += square(item)
            end
            # s_t[i,1] = sqrt(sum)
            # append!(s_t, sqrt(sum))
            push!(s_t, sqrt(sum))
        end

        println(typeof(s_t))
        println("s_t vector:")
        display(s_t)
        print("\n\n")

        # Create diagonal matrix H_t
        # First of all create the diagonal matrix from s_t solution

        mat = diagm(s_t)

        println("Diag matrix:")
        display(mat)
        print("\n\n")

        # Construct Identity matrix
        Iden = Matrix{Float64}(I, solver.n, solver.n)

        println("Identity:")
        display(Iden)
        print("\n\n")

        delta_Id = solver.delta .* Iden

        println("Delta I:")
        display(delta_Id)
        print("\n\n")

        # Finally sum them (hessian approximation)
        H_t = delta_Id + mat
       
        println("H_t:")
        display(H_t)
        print("\n\n")

        iter += 1

    end

    return 0

end


val = my_ADAGRAD(solver)