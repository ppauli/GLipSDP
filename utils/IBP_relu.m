function [Lout, Uout] = IBP_relu(L, U)

    Lout = max(L, 0);
    Uout = max(U, 0);

end