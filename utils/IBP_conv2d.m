function [Lout, Uout] = IBP_conv2d(L, U, Wgt, bias, stride, padding)

    if nargin < 6
        padding = 0;
    end

    Wpos = max(Wgt, 0);
    Wneg = min(Wgt, 0);

    Lout = conv_forward_direct(Wpos, L, stride, padding) + ...
           conv_forward_direct(Wneg, U, stride, padding) + reshape(bias, [], 1, 1);

    Uout = conv_forward_direct(Wpos, U, stride, padding) + ...
           conv_forward_direct(Wneg, L, stride, padding) + reshape(bias, [], 1, 1);

end