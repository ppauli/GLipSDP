function L = lipopt_l2_shor_d2(W1, W2)
% One-hidden-layer case:
%
%   f(x) = W2 sigma(W1 x)
%
%   W1 : n1 x n0
%   W2 : 1  x n1
%
% Variables:
%   y = [1; t; u1]
%
% where
%   ||t||_2 <= 1
%   -1 <= u1_i <= 1
%
% Objective:
%   max 1/2 * t' * W1' * diag(u1 + 1) * W2'

    [n1, n0] = size(W1);

    if size(W2, 1) ~= 1 || size(W2, 2) ~= n1
        error('Expected W2 to have size 1 x n1.');
    end

    % Variable indexing in y = [1; t; u1]
    idx_const = 1;
    idx_t  = 2:(1+n0);
    idx_u1 = (2+n0):(1+n0+n1);

    N = 1 + n0 + n1;

    % Build symmetric objective matrix C such that objective = trace(C X)
    C = zeros(N, N);

    for i = 1:n1
        a = W2(1, i);

        for k = 1:n0
            coeff = 0.5 * a * W1(i, k);

            % Linear term: coeff * t_k
            C(idx_const, idx_t(k)) = C(idx_const, idx_t(k)) + coeff / 2;
            C(idx_t(k), idx_const) = C(idx_t(k), idx_const) + coeff / 2;

            % Bilinear term: coeff * u1_i * t_k
            C(idx_u1(i), idx_t(k)) = C(idx_u1(i), idx_t(k)) + coeff / 2;
            C(idx_t(k), idx_u1(i)) = C(idx_t(k), idx_u1(i)) + coeff / 2;
        end
    end

    cvx_begin sdp quiet
        cvx_solver mosek
        variable X(N, N) symmetric

        maximize trace(C * X)

        subject to
            X >= 0;
            X(idx_const, idx_const) == 1;

            % l2 unit ball for t
            trace(X(idx_t, idx_t)) <= 1;

            % hidden derivative variables u1_i in [-1, 1]
            for i = 1:n1
                X(idx_u1(i), idx_u1(i)) <= 1;
                X(idx_const, idx_u1(i)) <= 1;
                X(idx_const, idx_u1(i)) >= -1;
            end
    cvx_end

    L = cvx_optval;
end