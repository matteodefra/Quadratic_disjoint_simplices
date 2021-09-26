module JuMPSolution

using JuMP
using LinearAlgebra
import Ipopt
import Test

mutable struct JuMPSol
    n :: Int
    K :: Int
    A :: Array{Int64}
    Q :: Matrix{Float64}
    q :: Array{Float64}
    opt_val :: Float64
    JuMPSol(n, x, A, Q, q) = new(n, x, A, Q, q)
end


function compute_solution(jump_sol)

    model = Model(Ipopt.Optimizer)

    @variable( model, x[1:jump_sol.n] >= 0 )
    @objective( model, Max, x' * jump_sol.Q * x + dot(jump_sol.q, x) )
    @constraint( model, jump_sol.A * x .== ones((jump_sol.K,1)) )
    optimize!(model)

    print(model)
    println("Objective value: ", objective_value(model))
    println("x = ", value.(x))

    jump_sol.opt_val = round( objective_value(model), digits = 8 )

    Test.@test termination_status(model) == MOI.LOCALLY_SOLVED
    Test.@test primal_status(model) == MOI.FEASIBLE_POINT
    # Test.@test objective_value(model) ≈ 0.32699 atol = 1e-5
    # Test.@test value.(x) ≈ 0.32699 atol = 1e-5

    return jump_sol

end


end