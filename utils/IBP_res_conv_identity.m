function [Lout, Uout, cache] = IBP_res_conv_identity( ...
    Lin, Uin, W1, b1, W2, b2, stride1, padding1, stride2, padding2)
%IBP_RES_CONV_IDENTITY IBP through residual conv block:
%
%   Y = X + Conv2(ReLU(Conv1(X)))
%
% Inputs:
%   Lin, Uin : input bounds
%              current convention should match IBP_conv2d
%              e.g. C x H x W if your conv code is CHW
%
%   W1, b1   : first convolution weights/bias
%   W2, b2   : second convolution weights/bias
%
%   stride1, padding1 : first conv stride/padding
%   stride2, padding2 : second conv stride/padding
%
% Outputs:
%   Lout, Uout : output bounds after residual addition
%   cache      : intermediate bounds for debugging / Alpha-Beta

    Lin = single(Lin);
    Uin = single(Uin);

    W1 = single(W1);
    W2 = single(W2);
    b1 = single(b1);
    b2 = single(b2);

    % First conv: z1 = Conv1(x)
    [preL1, preU1] = IBP_conv2d(Lin, Uin, W1, b1, stride1, padding1);

    % Internal ReLU
    [postL1, postU1] = IBP_relu(preL1, preU1);

    % Second conv: r = Conv2(ReLU(z1))
    [resL, resU] = IBP_conv2d(postL1, postU1, W2, b2, stride2, padding2);

    % Identity skip: y = x + r
    if ~isequal(size(Lin), size(resL))
        error(['Identity residual conv block requires input and residual ', ...
               'branch output to have the same size. Got input size [%s] ', ...
               'and residual size [%s].'], ...
               num2str(size(Lin)), num2str(size(resL)));
    end

    Lout = Lin + resL;
    Uout = Uin + resU;

    cache = struct();
    cache.preL1   = preL1;
    cache.preU1   = preU1;
    cache.postL1  = postL1;
    cache.postU1  = postU1;
    cache.resL    = resL;
    cache.resU    = resU;
end