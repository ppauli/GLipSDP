function [L_ub, diagnostics, aux] = relu_lcep_l2_moment_2hidden(W, b, xCenter, radius, relaxOrder)
% RELU_LCEP_L2_MOMENT_2HIDDEN
%
% Moment/SOS relaxation of the semialgebraic LCEP for a scalar-output
% two-hidden-layer ReLU network:
%
%   h1 = ReLU(W1*x  + b1)
%   h2 = ReLU(W2*h1 + b2)
%   f  = W3*h2 + b3
%
% Computes an SDP upper bound on
%
%   L = sup_{x in Omega} ||grad f(x)||_2.
%
% The formal gradient is
%
%   grad f(x) =
%       W1' * D1 * W2' * D2 * W3',
%
% where D1 = diag(u1), D2 = diag(u2).
%
% For the l2 norm:
%
%   L = max_{||t||_2 <= 1} t' * grad f(x).
%
% The objective is cubic in (t,u1,u2), so relaxOrder >= 2 is needed.

validate_scalar_relu_network(W, b, 2);

W1 = W{1};
W2 = W{2};
W3 = W{3};

b1 = b{1};
b2 = b{2};

n  = size(W1,2);
p1 = size(W1,1);
p2 = size(W2,1);

if nargin < 5
    relaxOrder = 2;
end

if relaxOrder < 2
    error('For two hidden layers, use relaxOrder >= 2 because the objective is cubic.');
end

xCenter = xCenter(:);

if length(xCenter) ~= n
    error('xCenter has incompatible dimension.');
end

% YALMIP decision variables.
x  = sdpvar(n,1);      % input
t  = sdpvar(n,1);      % l2 dual variable

z1 = sdpvar(p1,1);     % first-layer preactivation
h1 = sdpvar(p1,1);     % first-layer activation
u1 = sdpvar(p1,1);     % first-layer derivative variables

z2 = sdpvar(p2,1);     % second-layer preactivation
u2 = sdpvar(p2,1);     % second-layer derivative variables

F = [];

% Compact l2 input domain.
F = [F, (x - xCenter)'*(x - xCenter) <= radius^2];

% Dual l2 unit ball.
F = [F, t'*t <= 1];

% First layer:
%
% z1 = W1*x + b1
% h1 = ReLU(z1)
% u1 in G(z1)
%
F = [F, z1 == W1*x + b1];

% ReLU graph for h1 = max(0,z1):
F = [F, h1.*(h1 - z1) == 0];
F = [F, h1 >= z1];
F = [F, h1 >= 0];

% Exact semialgebraic derivative graph for first layer:
F = [F, u1.*(u1 - 1) == 0];
F = [F, (u1 - 0.5).*z1 >= 0];

% Second layer:
%
% z2 = W2*h1 + b2
% u2 in G(z2)
%
F = [F, z2 == W2*h1 + b2];

% Exact semialgebraic derivative graph for second layer:
F = [F, u2.*(u2 - 1) == 0];
F = [F, (u2 - 0.5).*z2 >= 0];

% Scalar-output gradient:
%
%   grad f = W1' * D1 * W2' * D2 * W3'
%
% In vectorized form:
%
%   c2     = u2 .* W3'
%   middle = W2' * c2
%   grad   = W1' * (u1 .* middle)
%
c = W3(:);
c2 = u2 .* c;
middle = W2' * c2;
grad = W1' * (u1 .* middle);

% l2 norm via support function.
obj = t' * grad;

ops = sdpsettings( ...
    'solver', 'mosek', ...
    'verbose', 1);

diagnostics = solvemoment(F, -obj, ops, relaxOrder);

L_ub = -value(-obj);

aux.x = x;
aux.t = t;
aux.z1 = z1;
aux.h1 = h1;
aux.u1 = u1;
aux.z2 = z2;
aux.u2 = u2;
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