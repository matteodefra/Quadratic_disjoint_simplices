module ConvexSolution

using Convex, SCS

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
    problem = minimize( quadform(convex_sol.x, convex_sol.Q) + dot(convex_sol.q, convex_sol.x) )

    problem.constraints += [convex_sol.A * convex_sol.x == 1]
    # for row in eachrow(convex_sol.A)
    #     problem.constraints += [row' * convex_sol.x == 1]
    # end

    problem.constraints += [convex_sol.x >= 0]

    println(problem)

    solve!(problem, () -> SCS.Optimizer(verbose=true), verbose=false)

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

    return convex_sol
end
    
end