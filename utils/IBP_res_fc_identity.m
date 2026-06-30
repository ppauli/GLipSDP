function [Lout, Uout, cache] = IBP_res_fc_identity(Lin, Uin, W1, b1, W2, b2)
%IBP_RES_FC_IDENTITY IBP through residual FC block:
%
%   y = x + W2 * ReLU(W1 * x + b1) + b2
%
% Inputs:
%   Lin, Uin : d x 1 input interval bounds
%   W1, b1   : first affine layer
%   W2, b2   : second affine layer
%
% Outputs:
%   Lout, Uout : d x 1 output interval bounds
%   cache      : intermediate bounds for debugging

    Lin = single(Lin);
    Uin = single(Uin);

    W1 = single(W1);
    b1 = single(b1);
    W2 = single(W2);
    b2 = single(b2);

    % Ensure column vectors
    Lin = Lin(:);
    Uin = Uin(:);
    b1  = b1(:);
    b2  = b2(:);

    % First affine: z1 = W1*x + b1
    [preL1, preU1] = IBP_fc(Lin, Uin, W1, b1);

    % ReLU: h = ReLU(z1)
    [postL1, postU1] = IBP_relu(preL1, preU1);

    % Second affine: r = W2*h + b2
    [resL, resU] = IBP_fc(postL1, postU1, W2, b2);

    % Identity skip: y = x + r
    if numel(resL) ~= numel(Lin)
        error(['Residual identity skip requires output dimension to match input dimension. ', ...
               'Got input dim %d and residual branch output dim %d.'], ...
               numel(Lin), numel(resL));
    end

    Lout = Lin + resL;
    Uout = Uin + resU;

    % Optional debugging cache
    cache = struct;
    cache.preL1  = preL1;
    cache.preU1  = preU1;
    cache.postL1 = postL1;
    cache.postU1 = postU1;
    cache.resL   = resL;
    cache.resU   = resU;
end