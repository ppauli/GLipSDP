function [Lout, Uout] = IBP_maxpool2d(Lin, Uin, pool_kernel, pool_stride, pool_padding)
%IBP_MAXPOOL2D Interval bound propagation through max pooling.
%
% Lin, Uin       : C x H x W
% pool_kernel    : scalar, e.g. 3
% pool_stride    : scalar, e.g. 2
% pool_padding   : scalar, e.g. 1
%
% Lout, Uout     : C x outH x outW
%
% For max pooling:
%   lower bound = max over lower-bound patch
%   upper bound = max over upper-bound patch
%
% Padding is handled using -Inf, matching max-pooling semantics.

    if nargin < 5 || isempty(pool_padding)
        pool_padding = 0;
    end

    Lin = single(Lin);
    Uin = single(Uin);

    C = size(Lin, 1);
    H = size(Lin, 2);
    W = size(Lin, 3);

    k = pool_kernel;
    s = pool_stride;
    p = pool_padding;

    % Pad with -Inf for max pooling
    LinPad = -inf(C, H + 2*p, W + 2*p, "single");
    UinPad = -inf(C, H + 2*p, W + 2*p, "single");

    LinPad(:, p+1:p+H, p+1:p+W) = Lin;
    UinPad(:, p+1:p+H, p+1:p+W) = Uin;

    Hp = H + 2*p;
    Wp = W + 2*p;

    outH = floor((Hp - k) / s) + 1;
    outW = floor((Wp - k) / s) + 1;

    Lout = zeros(C, outH, outW, "single");
    Uout = zeros(C, outH, outW, "single");

    for c = 1:C
        for oh = 1:outH
            hStart = (oh - 1) * s + 1;
            hEnd   = hStart + k - 1;

            for ow = 1:outW
                wStart = (ow - 1) * s + 1;
                wEnd   = wStart + k - 1;

                patchL = LinPad(c, hStart:hEnd, wStart:wEnd);
                patchU = UinPad(c, hStart:hEnd, wStart:wEnd);

                Lout(c, oh, ow) = max(patchL(:));
                Uout(c, oh, ow) = max(patchU(:));
            end
        end
    end
end