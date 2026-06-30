function Y = conv_forward(X, Wgt, padding, stride)
%CONV_FORWARD Evaluate convolution/cross-correlation without bias.
%
% X   : H x W x Cin
% Wgt : Cout x Cin x kH x kW
% Y   : outH x outW x Cout

    X = X;
    Wgt = Wgt;

    Cout = size(Wgt, 1);
    CinW = size(Wgt, 2);
    kH   = size(Wgt, 3);
    kW   = size(Wgt, 4);

    H   = size(X, 1);
    W   = size(X, 2);
    Cin = size(X, 3);

    if Cin ~= CinW
        error("conv_forward input channel mismatch: X has %d channels, Wgt expects %d.", Cin, CinW);
    end

    outH = floor((H + 2*padding - kH) / stride) + 1;
    outW = floor((W + 2*padding - kW) / stride) + 1;

    Y = zeros(outH, outW, Cout, "single");

    for co = 1:Cout
        for oh = 1:outH
            for ow = 1:outW

                val = single(0);

                for ci = 1:Cin
                    for kh = 1:kH
                        ih = (oh - 1) * stride + kh - padding;

                        if ih < 1 || ih > H
                            continue;
                        end

                        for kw = 1:kW
                            iw = (ow - 1) * stride + kw - padding;

                            if iw < 1 || iw > W
                                continue;
                            end

                            val = val + Wgt(co, ci, kh, kw) * X(ih, iw, ci);
                        end
                    end
                end

                Y(oh, ow, co) = val;
            end
        end
    end
end