function [Lout, Uout] = IBP_fc(L, U, W, b)

    Wpos = max(W, 0);
    Wneg = min(W, 0);

    Lout = Wpos * L + Wneg * U + b(:);
    Uout = Wpos * U + Wneg * L + b(:);

end