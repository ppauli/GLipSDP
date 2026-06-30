function [alpha, beta] = neuronSlopeFromPreReluVector(L, U)
%NEURONSLOPEFROMPRERELUVECTOR One ReLU state per neuron.
%
% L, U:
%   n x 1
%
% alpha, beta:
%   n x 1

    L = L(:);
    U = U(:);

    n = length(L);

    alpha = zeros(n, 1);
    beta  = zeros(n, 1);

    for j = 1:n

        lo = L(j);
        up = U(j);

        if up <= 0
            % inactive ReLU
            alpha(j) = 0;
            beta(j)  = 0;

        elseif lo >= 0
            % active ReLU
            alpha(j) = 1;
            beta(j)  = 1;

        else
            % unstable ReLU
            alpha(j) = 0;
            beta(j)  = 1;
        end
    end

    alpha = double(alpha);
    beta  = double(beta);

end