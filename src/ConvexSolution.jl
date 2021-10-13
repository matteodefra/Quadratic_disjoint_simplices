module ConvexSolution

using Convex
# import SCS
# import ECOS
import COSMO
using MAT

mutable struct ConvexSol
    n :: Int
    x :: Variable
    A :: Array{Int64}
    Q :: Matrix{Float64}
    q :: Array{Float64}
    opt_val :: Float64
    ConvexSol(n, x, A, Q, q) = new(n, x, A, Q, q)
end

function compute_solution(convex_sol)

    # optimizers = [SCS.Optimizer(verbose=true), ECOS.Optimizer, COSMO.Optimizer]
    optimizers = [ COSMO.Optimizer ]

    problem = minimize( quadform(convex_sol.x, convex_sol.Q) + dot(convex_sol.q, convex_sol.x) )

    # problem.constraints += [convex_sol.A * convex_sol.x == ones((convex_sol.n, 1))]
    for row in eachrow(convex_sol.A)
        problem.constraints += [row' * convex_sol.x == 1]
    end

    problem.constraints += [convex_sol.x >= 0]

    println(problem)

    results = []

    for optimizer in optimizers

        # solve!(problem, () -> SCS.Optimizer(verbose=true), verbose=false)
        solve!(problem, optimizer, verbose=false)

        println("problem status is ", problem.status) # :Optimal, :Infeasible, :Unbounded etc.
        println("optimal value is ", problem.optval)

        println("Primal variables of problem:")
        display(convex_sol.x)
        print("\n")

        println("Dual variables constraints")
        for constraint in problem.constraints
            display(constraint.dual)
            print("\n")
        end

        convex_sol.opt_val = problem.optval[1]

        push!(results, problem.optval[1])

    end

    println("Complete values")
    display(results)
    print("\n")

    matwrite("mat/convexsol.mat", Dict(
        "Q" => convex_sol.Q,
        "q" => convex_sol.q, 
        "x" => evaluate(convex_sol.x)
    ); compress = true)

    return convex_sol
end
    
end
