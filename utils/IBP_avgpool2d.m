function [Lout, Uout] = IBP_avgpool2d(Lin, Uin, pool_kernel, pool_stride)
%IBP_AVGPOOL2D Interval bound propagation through average pooling.
%
% Lin, Uin      : H x W x C
% pool_kernel   : scalar, e.g. 2
% pool_stride   : scalar, e.g. 2
%
% Lout, Uout    : outH x outW x C

    Lin = single(Lin);
    Uin = single(Uin);

    C = size(Lin, 1);
    H = size(Lin, 2);
    W = size(Lin, 3);

    k = pool_kernel;
    s = pool_stride;

    outH = floor((H - k) / s) + 1;
    outW = floor((W - k) / s) + 1;

    Lout = zeros(C, outH, outW, "single");
    Uout = zeros(C, outH, outW, "single");

    for c = 1:C
        for oh = 1:outH
            hStart = (oh - 1) * s + 1;
            hEnd   = hStart + k - 1;

            for ow = 1:outW
                wStart = (ow - 1) * s + 1;
                wEnd   = wStart + k - 1;

                patchL = Lin(c, hStart:hEnd, wStart:wEnd);
                patchU = Uin(c, hStart:hEnd, wStart:wEnd);

                Lout(c, oh, ow) = mean(patchL(:));
                Uout(c, oh, ow) = mean(patchU(:));
            end
        end
    end
end