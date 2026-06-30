function [L, time, info] = liplt_mlp(W, alpha, beta, opts)
%LIPLT_MLP  LipLT l2-Lipschitz bound for a standard dense MLP.
%
% Network:
%
%   f(x) = W{K} phi(W{K-1} phi(... phi(W{1} x)))
%
% There are K linear layers and K-1 activation layers.
%
% INPUT:
%   W      : cell array of dense matrices
%   alpha  : activation lower slope bound
%   beta   : activation upper slope bound
%   opts   : optional struct
%
% Example for ReLU:
%
%   alpha = 0;
%   beta  = 1;
%
% OUTPUT:
%   L      : LipLT l2-Lipschitz upper bound
%   info   : diagnostic information

    if nargin < 4
        opts = struct();
    end

    K = numel(W);

    if K < 2
        error('Need at least two linear layers.');
    end

    % Check dimensions.
    for k = 1:(K-1)
        if size(W{k}, 1) ~= size(W{k+1}, 2)
            error('Dimension mismatch: size(W{%d},1) must equal size(W{%d+1},2).', k, k);
        end
    end

    Lnum = K - 1;

    H = cell(Lnum, 1);
    G = cell(Lnum, 1);

    for k = 1:Lnum
        n_in_block  = size(W{k}, 2);
        n_out_block = size(W{k}, 1);

        H{k} = zeros(n_out_block, n_in_block);
        G{k} = eye(n_out_block);
    end
    
    tic
    [L, info] = liplt_residual(W, H, G, alpha, beta, opts);
    time = toc;
end