function [L_ub, diagnostics, aux] = relu_lcep_l2_moment_1hidden(W, b, xCenter, radius, relaxOrder)
% RELU_LCEP_L2_MOMENT_1HIDDEN
%
% Moment/SOS relaxation of the semialgebraic LCEP for a scalar-output
% one-hidden-layer ReLU network:
%
%   f(x) = W2 * ReLU(W1*x + b1) + b2.
%
% Computes an SDP upper bound on
%
%   L = sup_{x in Omega} ||grad f(x)||_2.
%
% Equivalent variational form:
%
%   L = max_{x,u,t} t' * W1' * diag(u) * W2'
%
% subject to
%
%   ||t||_2 <= 1,
%   x in Omega,
%   z1 = W1*x + b1,
%   u_i(u_i - 1) = 0,
%   (u_i - 1/2) z_i >= 0.
%
% Here u_i is the generalized ReLU derivative variable.

validate_scalar_relu_network(W, b, 1);

W1 = W{1};
W2 = W{2};
b1 = b{1};

n  = size(W1,2);
p1 = size(W1,1);

if nargin < 5
    relaxOrder = 2;
end

xCenter = xCenter(:);

if length(xCenter) ~= n
    error('xCenter has incompatible dimension.');
end

% YALMIP decision variables.
x  = sdpvar(n,1);     % input
t  = sdpvar(n,1);     % l2 dual variable
z1 = sdpvar(p1,1);    % first-layer preactivation
u1 = sdpvar(p1,1);    % ReLU generalized derivative variables

F = [];

% Compact l2 input domain.
F = [F, (x - xCenter)'*(x - xCenter) <= radius^2];

% Dual l2 unit ball.
F = [F, t'*t <= 1];

% Preactivation.
F = [F, z1 == W1*x + b1];

% Exact semialgebraic derivative graph:
% u_i in {0,1}, and sign consistency with z_i.
F = [F, u1.*(u1 - 1) == 0];
F = [F, (u1 - 0.5).*z1 >= 0];

% Scalar-output gradient:
%
%   grad f(x) = W1' * diag(u1) * W2'
%             = W1' * (u1 .* W2')
%
c = W2(:);
grad = W1' * (u1 .* c);

% l2 norm via support function:
%
%   ||grad||_2 = max_{||t||_2 <= 1} t' grad.
%
obj = t' * grad;

% YALMIP minimizes, so minimize -obj.
ops = sdpsettings( ...
    'solver', 'mosek', ...
    'verbose', 1);

diagnostics = solvemoment(F, -obj, ops, relaxOrder);

% The relaxation minimizes -obj. Therefore the upper bound on obj is -value(-obj).
L_ub = -value(-obj);

aux.x = x;
aux.t = t;
aux.z1 = z1;
aux.u1 = u1;
aux.objective = obj;
aux.constraints = F;
aux.relaxOrder = relaxOrder;

end

function validate_scalar_relu_network(W, b, numHiddenLayers)
% VALIDATE_SCALAR_RELU_NETWORK
%
% Validates dimensions for scalar-output fully connected ReLU networks.

if ~iscell(W) || ~iscell(b)
    error('W and b must be cell arrays.');
end

if numel(W) ~= numHiddenLayers + 1
    error('Expected %d weight matrices for %d hidden layers.', ...
        numHiddenLayers + 1, numHiddenLayers);
end

if numel(b) ~= numel(W)
    error('W and b must have the same number of layers.');
end

for ell = 1:numel(W)
    if size(W{ell},1) ~= length(b{ell})
        error('Size mismatch between W{%d} and b{%d}.', ell, ell);
    end
end

for ell = 2:numel(W)
    if size(W{ell},2) ~= size(W{ell-1},1)
        error('Layer dimension mismatch between W{%d} and W{%d}.', ell-1, ell);
    end
end

if size(W{end},1) ~= 1
    error('The output must be scalar, so W{end} must have one row.');
end

end