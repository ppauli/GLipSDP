function [Lseq, time, info] = seqlip(W, opts)
%SEQLIP  SeqLip Lipschitz estimate for dense sequential neural networks.
%
%   [Lseq, info] = seqlip(W, opts)
%
%   INPUT:
%       W : cell array of weight matrices
%           W{1} : n1 x n0
%           W{2} : n2 x n1
%           ...
%           W{K} : nK x n_{K-1}
%
%       The network is assumed to be
%
%           f(x) = W{K} rho_{K-1}( ... rho_1(W{1} x) ... )
%
%       with activation derivatives satisfying
%
%           0 <= rho_i'(.) <= 1.
%
%   OPTIONS:
%       opts.bruteMaxDim : max hidden width for exact ReLU enumeration
%                          default: 20
%
%       opts.numRestarts : number of random restarts for greedy mode
%                          default: 20
%
%       opts.maxIter     : projected-gradient iterations
%                          default: 200
%
%       opts.step0       : initial gradient-ascent step size
%                          default: 1.0
%
%       opts.tol         : stopping tolerance
%                          default: 1e-8
%
%       opts.verbose     : true / false
%                          default: false
%
%   OUTPUT:
%       Lseq : SeqLip estimate
%
%       info.factor{i}       : SeqLip factor for activation i
%       info.sigma{i}        : optimizing gate sigma_i
%       info.method{i}       : 'bruteforce' or 'greedy'
%       info.autolip         : product of spectral norms
%       info.isCertified     : true only if all factors were exact
%
%   WARNING:
%       If greedy mode is used, the returned value is the Greedy SeqLip
%       heuristic, not a certified upper bound. The exact SeqLip bound
%       requires solving each max problem globally.

    if nargin < 2
        opts = struct();
    end

    opts = set_default_opts(opts);

    K = numel(W);

    if K < 2
        error('SeqLip requires at least two linear layers.');
    end

    % Check dimensions.
    for i = 1:(K-1)
        if size(W{i}, 1) ~= size(W{i+1}, 2)
            error('Dimension mismatch: size(W{%d},1) must equal size(W{%d},2).', i, i+1);
        end
    end

    % Thin SVDs.
    U = cell(K, 1);
    S = cell(K, 1);
    V = cell(K, 1);
    svals = cell(K, 1);

    for i = 1:K
        [U{i}, S{i}, V{i}] = svd(W{i}, 'econ');
        svals{i} = diag(S{i});
    end

    % AutoLip baseline: product of spectral norms.
    Lauto = 1;
    for i = 1:K
        if isempty(svals{i})
            Lauto = 0;
        else
            Lauto = Lauto * svals{i}(1);
        end
    end

    factors = cell(K-1, 1);
    sigmas  = cell(K-1, 1);
    methods = cell(K-1, 1);
    
    tic
    allExact = true;
    Lseq = 1;

    for i = 1:(K-1)

        nGate = size(W{i}, 1);

        St_i   = make_tilde_sigma(S{i},   i,   K);
        St_ip1 = make_tilde_sigma(S{i+1}, i+1, K);

        % Factor has form:
        %
        %   max_sigma || St_{i+1} V_{i+1}' diag(sigma) U_i St_i ||_2
        %
        % Let A(sigma) = Lmat * diag(sigma) * Rmat.
        %
        Lmat = St_ip1 * V{i+1}';
        Rmat = U{i} * St_i;

        if nGate <= opts.bruteMaxDim
            [fac, sigma] = seqlip_factor_bruteforce(Lmat, Rmat);
            method = 'bruteforce';
        else
            [fac, sigma] = seqlip_factor_greedy(Lmat, Rmat, opts);
            method = 'greedy';
            allExact = false;
        end

        factors{i} = fac;
        sigmas{i} = sigma;
        methods{i} = method;

        Lseq = Lseq * fac;

        if opts.verbose
            fprintf('Activation %d: factor = %.6g, method = %s\n', i, fac, method);
        end
    end
    time = toc;

    info = struct();
    info.factor = factors;
    info.sigma = sigmas;
    info.method = methods;
    info.autolip = Lauto;
    info.isCertified = allExact;

    if ~allExact && opts.verbose
        fprintf('Warning: greedy mode was used, so this is Greedy SeqLip, not a certified upper bound.\n');
        fprintf('AutoLip certified upper bound: %.6g\n', Lauto);
    end
end


function opts = set_default_opts(opts)

    if ~isfield(opts, 'bruteMaxDim')
        opts.bruteMaxDim = 20;
    end

    if ~isfield(opts, 'numRestarts')
        opts.numRestarts = 20;
    end

    if ~isfield(opts, 'maxIter')
        opts.maxIter = 200;
    end

    if ~isfield(opts, 'step0')
        opts.step0 = 1.0;
    end

    if ~isfield(opts, 'tol')
        opts.tol = 1e-8;
    end

    if ~isfield(opts, 'verbose')
        opts.verbose = false;
    end
end


function St = make_tilde_sigma(S, layerIndex, K)
%MAKE_TILDE_SIGMA  Implements the SeqLip tilde-Sigma convention.
%
%   Sigma_tilde_i = Sigma_i       for i = 1 or i = K
%                 = Sigma_i^(1/2) otherwise

    if layerIndex == 1 || layerIndex == K
        St = S;
    else
        St = sqrt(S);
    end
end


function [bestVal, bestSigma] = seqlip_factor_bruteforce(Lmat, Rmat)
% Exact enumeration for ReLU gates sigma in {0,1}^n.
%
% Since ||L diag(sigma) R||_2 is convex in sigma over [0,1]^n,
% the maximum over the box is attained at an extreme point.
% Thus, enumerating {0,1}^n gives the exact relaxed maximum,
% but only for small n.

    n = size(Lmat, 2);

    if n ~= size(Rmat, 1)
        error('Dimension mismatch in factor matrices.');
    end

    numMasks = 2^n;

    bestVal = -inf;
    bestSigma = zeros(n, 1);

    for mask = 0:(numMasks-1)
        sigma = zeros(n, 1);

        for j = 1:n
            sigma(j) = bitget(mask, j);
        end

        A = Lmat * diag(sigma) * Rmat;
        val = norm(A, 2);

        if val > bestVal
            bestVal = val;
            bestSigma = sigma;
        end
    end
end


function [bestVal, bestSigma] = seqlip_factor_greedy(Lmat, Rmat, opts)
% Greedy projected-gradient ascent for
%
%   max_{0 <= sigma <= 1} ||Lmat diag(sigma) Rmat||_2.
%
% This is the Greedy SeqLip heuristic. It is not globally certified.

    n = size(Lmat, 2);

    bestVal = -inf;
    bestSigma = zeros(n, 1);

    % Include deterministic starts.
    starts = cell(opts.numRestarts + 2, 1);
    starts{1} = ones(n, 1);
    starts{2} = 0.5 * ones(n, 1);

    for r = 1:opts.numRestarts
        starts{r+2} = rand(n, 1);
    end

    for r = 1:numel(starts)

        sigma = starts{r};
        oldVal = -inf;

        for it = 1:opts.maxIter

            [val, grad] = seqlip_value_grad(Lmat, Rmat, sigma);

            if abs(val - oldVal) <= opts.tol * max(1, abs(oldVal))
                break;
            end

            oldVal = val;

            % Normalize gradient for stability.
            gnorm = norm(grad, 2);
            if gnorm > 0
                grad = grad / gnorm;
            else
                break;
            end

            % Backtracking projected ascent.
            step = opts.step0;
            improved = false;

            while step > 1e-12
                sigmaNew = min(1, max(0, sigma + step * grad));
                valNew = norm(Lmat * diag(sigmaNew) * Rmat, 2);

                if valNew >= val
                    sigma = sigmaNew;
                    improved = true;
                    break;
                end

                step = step / 2;
            end

            if ~improved
                break;
            end
        end

        val = norm(Lmat * diag(sigma) * Rmat, 2);

        if val > bestVal
            bestVal = val;
            bestSigma = sigma;
        end
    end
end


function [val, grad] = seqlip_value_grad(Lmat, Rmat, sigma)
% Computes value and gradient of
%
%   phi(sigma) = ||Lmat diag(sigma) Rmat||_2.
%
% If the top singular value is simple, then
%
%   d phi / d sigma_j =
%       (Lmat' * u)_j * (Rmat * v)_j
%
% where u, v are top left/right singular vectors of A.

    A = Lmat * diag(sigma) * Rmat;

    [U, S, V] = svd(A, 'econ');

    val = S(1, 1);
    u = U(:, 1);
    v = V(:, 1);

    grad = (Lmat' * u) .* (Rmat * v);
end