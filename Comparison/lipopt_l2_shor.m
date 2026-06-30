function [L, time] = lipopt_l2_shor(W)
%LIPOPT_L2_SHOR  Shor SDP relaxation for an l2 Lipschitz upper bound.
%
%   L = lipopt_l2_shor(W)
%
%   INPUT:
%       W : cell array of weights
%           W{1} : n1 x n0
%           W{2} : n2 x n1      for d = 3
%           W{3} : 1  x n2      for d = 3
%
%           or
%
%           W{1} : n1 x n0
%           W{2} : 1  x n1      for d = 2
%
%   OUTPUT:
%       L : SDP upper bound on the l2 Lipschitz constant.
%
%   Requires CVX: http://cvxr.com/cvx/
%
%   The activation derivative is assumed to satisfy
%       0 <= sigma'(.) <= 1.
%
%   Internally, hidden-layer derivative variables are normalized:
%       u_i = 2 s_i - 1,
%   so that
%       u_i in [-1, 1],
%       s_i = (u_i + 1)/2.
%
%   This implements the Shor SDP relaxation of the QCQP described
%   in Section 5 of LiPopt.

    d = numel(W);
    tic
    if d == 2
        L = lipopt_l2_shor_d2(W{1}, W{2});
    elseif d == 3
        L = lipopt_l2_shor_d3(W{1}, W{2}, W{3});
    else
        error('This implementation supports d = 2 or d = 3. For deeper networks, recursively introduce product variables.');
    end
    time = toc;
end
