function [alpha, beta] = channelSlopeFromPreReluTensor(L, U, row, col)
%CHANNELSLOPEFROMPRERELUTENSOR One ReLU state per output channel.
%
% L, U layout:
%   C x H x W
%
% alpha, beta:
%   C x 1

    C = size(L, 1);
    H = size(L, 2);
    W = size(L, 3);

    if row < 1 || row > H || col < 1 || col > W
        error("Requested spatial index (%d,%d) is outside tensor size %d x %d.", ...
            row, col, H, W);
    end

    alpha = zeros(C, 1);
    beta  = zeros(C, 1);

    for c = 1:C

        lo = L(c, row, col);
        up = U(c, row, col);

        if up <= 0
            % inactive ReLU
            alpha(c) = 0;
            beta(c)  = 0;

        elseif lo >= 0
            % active ReLU
            alpha(c) = 1;
            beta(c)  = 1;

        else
            % unstable ReLU
            alpha(c) = 0;
            beta(c)  = 1;
        end
    end

    alpha = double(alpha);
    beta  = double(beta);

end