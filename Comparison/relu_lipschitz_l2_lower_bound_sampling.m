function [L_lb, bestX, bestPattern, info] = relu_lipschitz_l2_lower_bound_sampling(W, b, lb, ub, numSamples)
% RELU_LIPSCHITZ_L2_LOWER_BOUND_SAMPLING
%
% Computes a sampling-based lower bound on the l2 Lipschitz constant of a
% fully connected ReLU network.
%
% Network convention:
%
%   x_1 = ReLU(W{1} x_0 + b{1})
%   x_2 = ReLU(W{2} x_1 + b{2})
%   ...
%   f   = W{L} x_{L-1} + b{L}
%
% The last layer is linear.
%
% The lower bound is:
%
%   L_lb = max_{sampled x} ||J_f(x)||_2.
%
% INPUT:
%
%   W          : cell array of weight matrices
%   b          : cell array of biases
%   lb         : lower input bound, n_in x 1
%   ub         : upper input bound, n_in x 1
%   numSamples : number of random samples
%
% OUTPUT:
%
%   L_lb        : sampling-based lower bound
%   bestX       : sampled input attaining the largest observed Jacobian norm
%   bestPattern : activation pattern at bestX
%   info        : diagnostic information

validate_network(W, b);

nIn = size(W{1}, 2);

lb = lb(:);
ub = ub(:);

if length(lb) ~= nIn || length(ub) ~= nIn
    error('lb and ub must have length equal to the input dimension.');
end

if any(~isfinite(lb)) || any(~isfinite(ub))
    error('Sampling lower bound requires finite lb and ub.');
end

if any(lb > ub)
    error('Each lower bound must be <= the corresponding upper bound.');
end

if nargin < 5 || isempty(numSamples)
    numSamples = 10000;
end

L_lb = -inf;
bestX = [];
bestPattern = [];

numHiddenLayers = numel(W) - 1;

info.numSamples = numSamples;
info.values = zeros(numSamples, 1);
info.bestJacobian = [];

for s = 1:numSamples

    % Uniform sample from the input box.
    x = lb + rand(nIn, 1) .* (ub - lb);

    % Get activation pattern induced by x.
    pattern = relu_activation_pattern(W, b, x);

    % Compute corresponding Jacobian.
    J = pattern_jacobian(W, pattern);

    % l2-induced operator norm.
    val = norm(J, 2);

    info.values(s) = val;

    if val > L_lb
        L_lb = val;
        bestX = x;
        bestPattern = pattern;
        info.bestJacobian = J;
    end
end

info.meanValue = mean(info.values);
info.maxValue = max(info.values);
info.minValue = min(info.values);
info.stdValue = std(info.values);

end

function pattern = relu_activation_pattern(W, b, x)
% RELU_ACTIVATION_PATTERN
%
% Computes the binary activation pattern of a ReLU network at input x.
%
% pattern{ell}(i) = 1 if neuron i in hidden layer ell is active,
% pattern{ell}(i) = 0 otherwise.

numLayers = numel(W);
numHiddenLayers = numLayers - 1;

pattern = cell(numHiddenLayers, 1);

a = x(:);

for ell = 1:numHiddenLayers
    z = W{ell} * a + b{ell};

    d = double(z >= 0);
    pattern{ell} = d;

    a = max(z, 0);
end

end

function J = pattern_jacobian(W, pattern)
% PATTERN_JACOBIAN
%
% Computes:
%
%   J = W{L} D_{L-1} W{L-1} ... D_1 W{1}.
%
% The last layer is assumed linear.

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

function validate_network(W, b)
% VALIDATE_NETWORK
%
% Validates dimensions of a fully connected feedforward network.

if ~iscell(W) || ~iscell(b)
    error('W and b must be cell arrays.');
end

if numel(W) ~= numel(b)
    error('W and b must have the same number of layers.');
end

if numel(W) < 2
    error('The network must have at least one hidden layer and one output layer.');
end

for ell = 1:numel(W)
    if size(W{ell}, 1) ~= length(b{ell})
        error('Size mismatch between W{%d} and b{%d}.', ell, ell);
    end
end

for ell = 2:numel(W)
    if size(W{ell}, 2) ~= size(W{ell-1}, 1)
        error('Layer dimension mismatch between W{%d} and W{%d}.', ell-1, ell);
    end
end

end