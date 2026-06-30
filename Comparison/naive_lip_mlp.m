function [L, time] = naive_lip_mlp(W)
%NAIVE_LIP_MLP  Product-of-spectral-norms Lipschitz bound.
    tic
    L = 1;

    for k = 1:numel(W)
        L = L * norm(W{k}, 2);
    end
    time = toc;
end