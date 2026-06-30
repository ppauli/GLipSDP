function L = lipopt_l2_shor_d3(W1, W2, W3)
% Two-hidden-layer Shor SDP relaxation for l2 Lipschitz upper bound.
%
% W1 : n1 x n0
% W2 : n2 x n1
% W3 : 1  x n2

    [n1, n0] = size(W1);
    [n2, n1_check] = size(W2);

    if n1_check ~= n1
        error('Dimension mismatch: W2 must have size n2 x n1.');
    end

    if size(W3, 1) ~= 1 || size(W3, 2) ~= n2
        error('Expected W3 to have size 1 x n2.');
    end

    % y = [1; t; u1; u2; r]
    idx_const = 1;
    idx_t  = 2:(1+n0);
    idx_u1 = (2+n0):(1+n0+n1);
    idx_u2 = (2+n0+n1):(1+n0+n1+n2);

    r_start = 2 + n0 + n1 + n2;
    num_r = n1 * n2;
    idx_r = r_start:(r_start + num_r - 1);

    N = 1 + n0 + n1 + n2 + num_r;

    r_index = @(i,j) idx_r((j-1)*n1 + i);

    C = zeros(N, N);

    for i = 1:n1
        for j = 1:n2
            aij = 0.25 * W2(j, i) * W3(1, j);

            for k = 1:n0
                coeff = aij * W1(i, k);

                % coeff * t_k
                a = idx_const;
                b = idx_t(k);
                C(a,b) = C(a,b) + coeff/2;
                C(b,a) = C(b,a) + coeff/2;

                % coeff * t_k * u1_i
                a = idx_t(k);
                b = idx_u1(i);
                C(a,b) = C(a,b) + coeff/2;
                C(b,a) = C(b,a) + coeff/2;

                % coeff * t_k * u2_j
                a = idx_t(k);
                b = idx_u2(j);
                C(a,b) = C(a,b) + coeff/2;
                C(b,a) = C(b,a) + coeff/2;

                % coeff * t_k * r_ij
                a = idx_t(k);
                b = r_index(i,j);
                C(a,b) = C(a,b) + coeff/2;
                C(b,a) = C(b,a) + coeff/2;
            end
        end
    end

    cvx_begin sdp quiet
        cvx_solver mosek
        variable X(N, N) symmetric

        maximize trace(C * X)

        subject to
            X >= 0;
            X(idx_const, idx_const) == 1;

            % l2 dual variable constraint: ||t||_2 <= 1
            trace(X(idx_t, idx_t)) <= 1;

            % u1_i in [-1, 1]
            for i = 1:n1
                X(idx_u1(i), idx_u1(i)) <= 1;
                X(idx_const, idx_u1(i)) <= 1;
                X(idx_const, idx_u1(i)) >= -1;
            end

            % u2_j in [-1, 1]
            for j = 1:n2
                X(idx_u2(j), idx_u2(j)) <= 1;
                X(idx_const, idx_u2(j)) <= 1;
                X(idx_const, idx_u2(j)) >= -1;
            end

            % r_ij = u1_i * u2_j relaxed through Shor lifting
            for i = 1:n1
                for j = 1:n2
                    ridx = r_index(i,j);

                    X(idx_const, ridx) == X(idx_u1(i), idx_u2(j));

                    X(ridx, ridx) <= 1;
                    X(idx_const, ridx) <= 1;
                    X(idx_const, ridx) >= -1;
                end
            end
    cvx_end

    L = cvx_optval;
end