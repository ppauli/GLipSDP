function [L_fgl, bestPattern, info] = relu_fgl_l2(W)
% RELU_FGL_L2_ALL_PATTERNS
%
% Formal Global Lipschitz bound for a fully connected ReLU network
% with respect to the l2 norm.
%
% This enumerates all activation patterns, including infeasible ones.
%
% Network:
%
%   x_1 = ReLU(W{1} x_0 + b{1})
%   ...
%   f   = W{L} x_{L-1} + b{L}
%
% Biases are not needed for FGL, because FGL ignores feasibility.
%
% Computes:
%
%   L_fgl = max_sigma || W{L} D_{L-1} W{L-1} ... D_1 W{1} ||_2.

validate_weight_network(W);

numLayers = numel(W);
numHiddenLayers = numLayers - 1;

hiddenSizes = zeros(numHiddenLayers, 1);
for ell = 1:numHiddenLayers
    hiddenSizes(ell) = size(W{ell}, 1);
end

numHiddenUnits = sum(hiddenSizes);
numPatterns = 2^numHiddenUnits;

if numHiddenUnits > 30
    warning(['Enumerating 2^%d = %.4e activation patterns. ', ...
             'This may be very expensive.'], ...
             numHiddenUnits, numPatterns);
end

L_fgl = -inf;
bestPattern = cell(numHiddenLayers, 1);

info.numHiddenUnits = numHiddenUnits;
info.numPatterns = numPatterns;
info.bestJacobian = [];

currentPattern = cell(numHiddenLayers, 1);
for ell = 1:numHiddenLayers
    currentPattern{ell} = zeros(hiddenSizes(ell), 1);
end

enumerate_layer(1);

    function enumerate_layer(ell)
        if ell > numHiddenLayers
            J = pattern_jacobian(W, currentPattern);
            val = norm(J, 2);

            if val > L_fgl
                L_fgl = val;
                bestPattern = currentPattern;
                info.bestJacobian = J;
            end

            return;
        end

        n = hiddenSizes(ell);

        for k = 0:(2^n - 1)
            currentPattern{ell} = integer_to_binary_vector(k, n);
            enumerate_layer(ell + 1);
        end
    end

end

function J = pattern_jacobian(W, pattern)
% Computes J = W{L} D_{L-1} W{L-1} ... D_1 W{1}.

numLayers = numel(W);
numHiddenLayers = numLayers - 1;

J = W{1};

for ell = 1:numHiddenLayers
    d = pattern{ell}(:);
    D = diag(d);

    if ell == 1
        J = D * W{1};
    else
        J = D * W{ell} * J;
    end
end

J = W{end} * J;

end

function v = integer_to_binary_vector(k, n)
% Converts integer k to an n-dimensional binary vector.
% Least significant bit becomes first entry.

v = zeros(n, 1);

for i = 1:n
    v(i) = bitget(k, i);
end

end

function validate_weight_network(W)
% Validates dimensions of weight matrices.

if ~iscell(W)
    error('W must be a cell array.');
end

if numel(W) < 2
    error('The network must have at least one hidden layer and one output layer.');
end

for ell = 2:numel(W)
    if size(W{ell}, 2) ~= size(W{ell-1}, 1)
        error('Layer dimension mismatch between W{%d} and W{%d}.', ell-1, ell);
    end
end

end