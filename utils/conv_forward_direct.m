function y = conv_forward(kernel, x, stride, padding)
%CONV_FORWARD_DIRECT Direct cross-correlation matching conv_layer_to_fc.
%
% kernel size: [c_out, c_in, k1, k2]
% x size:      [c_in, in1, in2]
% y size:      [c_out, out1, out2]

    if isscalar(stride)
        stride = [stride, stride];
    end

    if isscalar(padding)
        padding = [padding, padding];
    end

    [c_out, c_in, k1, k2] = size(kernel);
    [c_in_x, in1, in2] = size(x);

    if c_in_x ~= c_in
        error('Input channel mismatch.');
    end

    stride1 = stride(1);
    stride2 = stride(2);

    pad1 = padding(1);
    pad2 = padding(2);

    padded1 = in1 + 2 * pad1;
    padded2 = in2 + 2 * pad2;

    xpad = zeros(c_in, padded1, padded2);
    xpad(:, (1:in1) + pad1, (1:in2) + pad2) = x;

    out1 = floor((padded1 - k1) / stride1) + 1;
    out2 = floor((padded2 - k2) / stride2) + 1;

    y = zeros(c_out, out1, out2);

    for co = 1:c_out
        for o2 = 1:out2
            for o1 = 1:out1
                acc = 0;

                for ci = 1:c_in
                    for b = 1:k2
                        for a = 1:k1
                            p1 = (o1 - 1) * stride1 + a;
                            p2 = (o2 - 1) * stride2 + b;

                            acc = acc + kernel(co, ci, a, b) * xpad(ci, p1, p2);
                        end
                    end
                end

                y(co, o1, o2) = acc;
            end
        end
    end
end