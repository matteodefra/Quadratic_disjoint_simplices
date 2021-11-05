function [x, primal] = solve_prob(Q, q, A)
    
    n = length(q);

    disp(n);

    K = size(A,1);

    A = double(A);

    disp(K);

    b = ones(K,1);

    lb = zeros(n,1);

    options = optimoptions('quadprog','Display','iter-detailed');

    x = quadprog(Q, q, [], [], A, b, lb, [], [], options);

    primal = x' * Q * x + q' * x;
    disp(primal);

end
