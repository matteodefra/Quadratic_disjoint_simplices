function [primal] = solve_prob(Q, q, A)
    
    n = length(q);

    display(n);

    x = sdpvar(n,1);

    K = size(A,1);

    display(K);

    b = ones(K,1);

    A = double(A);

    Constraints = [A*x == b];

    for i=1:1:n
        Constraints = [Constraints, x(i) >= 0];
    end

    % Define an objective
    Objective = x'*Q*x + q'*x;
    
    % Set some options for YALMIP and solver
    % options = sdpsettings('verbose',1,'solver','quadprogbb');%,'quadprog.maxiter',100);
    
    % Solve the problem
    sol = optimize(Constraints,Objective);
    
    % Analyze error flags
    if sol.problem == 0
     % Extract and display value
     solution = value(x)
     display(value(x))
    else
     display('Hmm, something went wrong!');
     sol.info
     yalmiperror(sol.problem)
    end

    primal = computeprimal(Q, q, value(x));

    display(primal);

end

