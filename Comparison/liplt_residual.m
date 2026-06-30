function [L, info] = liplt_residual(W, H, G, alpha, beta, opts)
%LIPLT_RESIDUAL  LipLT l2-Lipschitz upper bound for residual-form networks.
%
% Implements the recursion from LipLT:
%
%   y_k     = W_k x_k
%   x_{k+1} = H_k x_k + G_k phi(y_k)
%
% with final output
%
%   y_L = W_L x_L.
%
% MATLAB indexing:
%
%   W{1}     = W_0
%   W{2}     = W_1
%   ...
%   W{L+1}   = W_L
%
%   H{1}     = H_0
%   ...
%   H{L}     = H_{L-1}
%
%   G{1}     = G_0
%   ...
%   G{L}     = G_{L-1}
%
% INPUT:
%   W      : cell array of length L+1
%   H      : cell array of length L
%   G      : cell array of length L
%   alpha  : lower slope bound of activation
%   beta   : upper slope bound of activation
%   opts   : optional struct
%
% OUTPUT:
%   L      : LipLT l2-Lipschitz upper bound
%   info   : diagnostic information
%
% Requires no CVX.

    if nargin < 6
        opts = struct();
    end
    opts = liplt_default_opts(opts);

    Lnum = numel(W) - 1;

    if numel(H) ~= Lnum || numel(G) ~= Lnum
        error('Expected numel(H) = numel(G) = numel(W)-1.');
    end

    c   = (alpha + beta) / 2;
    eta = (beta - alpha) / 2;

    % Construct loop-transformed linear parts:
    %
    %   Hhat_k = H_k + c G_k W_k.
    %
    Hhat = cell(Lnum, 1);

    for k = 1:Lnum
        Hhat{k} = H{k} + c * G{k} * W{k};
    end

    % m(1) stores m_0 = ||W_0||_2.
    m = zeros(Lnum + 1, 1);
    m(1) = liplt_spectral_norm(W{1}, opts);

    % Recursion:
    %
    %   m_{k+1}
    %       = ||W_{k+1} Hhat_k ... Hhat_0||
    %         + eta * sum_{j=0}^k
    %             ||W_{k+1} Hhat_k ... Hhat_{j+1} G_j|| m_j.
    %
    for k = 1:Lnum

        % First term: ||W_{k+1} Hhat_k ... Hhat_1||
        A = W{k+1};
        for r = k:-1:1
            A = A * Hhat{r};
        end

        mk = liplt_spectral_norm(A, opts);

        % Second term
        for j = 1:k
            B = W{k+1};

            % Product Hhat_k ... Hhat_{j+1}
            for r = k:-1:(j+1)
                B = B * Hhat{r};
            end

            B = B * G{j};

            mk = mk + eta * liplt_spectral_norm(B, opts) * m(j);
        end

        m(k+1) = mk;

        if opts.verbose
            fprintf('LipLT m_%d = %.6g\n', k, m(k+1));
        end
    end

    L = m(end);

    info = struct();
    info.m = m;
    info.Hhat = Hhat;
    info.alpha = alpha;
    info.beta = beta;
    info.c = c;
    info.eta = eta;
end


function opts = liplt_default_opts(opts)

    if ~isfield(opts, 'useSvds')
        opts.useSvds = false;
    end

    if ~isfield(opts, 'svdsThreshold')
        opts.svdsThreshold = 1000;
    end

    if ~isfield(opts, 'svdsTol')
        opts.svdsTol = 1e-6;
    end

    if ~isfield(opts, 'svdsMaxIter')
        opts.svdsMaxIter = 300;
    end

    if ~isfield(opts, 'verbose')
        opts.verbose = false;
    end
end


function s = liplt_spectral_norm(A, opts)
%LIPLT_SPECTRAL_NORM  Spectral norm helper.

    if isempty(A)
        s = 0;
        return;
    end

    if nnz(A) == 0
        s = 0;
        return;
    end

    [m, n] = size(A);

    if opts.useSvds && max(m, n) >= opts.svdsThreshold
        try
            svdOpts.tol = opts.svdsTol;
            svdOpts.maxit = opts.svdsMaxIter;
            s = svds(A, 1, 'largest', svdOpts);
        catch
            warning('svds failed. Falling back to norm(A,2).');
            s = norm(A, 2);
        end
    else
        s = norm(A, 2);
    end
end