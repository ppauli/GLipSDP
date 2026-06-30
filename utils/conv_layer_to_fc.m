function [Wfc, outSize] = conv_layer_to_fc(kernel, inputSize, stride, padding)
%CONV_LAYER_TO_FC Convert a 2D convolutional layer to an equivalent FC matrix.
%
%   [Wfc, outSize] = conv_layer_to_fc(kernel, inputSize, stride, padding)
%
% INPUTS:
%   kernel:
%       4D array of size [c_out, c_in, k1, k2]
%
%   inputSize:
%       [c_in, in1, in2]
%
%   stride:
%       scalar or [stride1, stride2]
%
%   padding:
%       scalar or [pad1, pad2]
%
% OUTPUTS:
%   Wfc:
%       sparse matrix such that
%
%           y(:) = Wfc * x(:)
%
%       where x has size [c_in, in1, in2]
%       and y has size [c_out, out1, out2].
%
%   outSize:
%       [c_out, out1, out2]
%
% CONVENTION:
%   This implements deep-learning cross-correlation, not mathematical
%   convolution. The kernel is NOT flipped.
%
%   Kernel convention:
%
%       kernel(co, ci, a, b)
%
%   y(co, o1, o2) =
%       sum_{ci,a,b} kernel(co,ci,a,b) * xpad(ci, i1, i2)
%
%   with
%       i1 = (o1-1)*stride1 + a
%       i2 = (o2-1)*stride2 + b

    % -----------------------------
    % Parse dimensions
    % -----------------------------
    if numel(inputSize) ~= 3
        error('inputSize must be [c_in, input_1, input_2].');
    end

    c_in = inputSize(1);
    in1  = inputSize(2);
    in2  = inputSize(3);

    % Kernel convention: [c_out, c_in, k1, k2]
    [c_out, c_in_kernel, k1, k2] = size(kernel);

    if c_in_kernel ~= c_in
       error('Kernel c_in does not match inputSize c_in.');
    end

    if isscalar(stride)
        stride = [stride, stride];
    end

    if isscalar(padding)
        padding = [padding, padding];
    end

    if numel(stride) ~= 2
        error('stride must be scalar or [stride1, stride2].');
    end

    if numel(padding) ~= 2
        error('padding must be scalar or [pad1, pad2].');
    end

    stride1 = stride(1);
    stride2 = stride(2);

    pad1 = padding(1);
    pad2 = padding(2);

    % -----------------------------
    % Output dimensions
    % -----------------------------
    padded1 = in1 + 2 * pad1;
    padded2 = in2 + 2 * pad2;

    out1 = floor((padded1 - k1) / stride1) + 1;
    out2 = floor((padded2 - k2) / stride2) + 1;

    if out1 <= 0 || out2 <= 0
        error('Invalid output size. Check inputSize, kernel size, stride, and padding.');
    end

    outSize = [c_out, out1, out2];

    nIn  = c_in  * in1  * in2;
    nOut = c_out * out1 * out2;

    nnzMax = nOut * c_in * k1 * k2;

    rows = zeros(nnzMax, 1);
    cols = zeros(nnzMax, 1);
    vals = zeros(nnzMax, 1);

    count = 0;

    % -----------------------------
    % Build sparse FC matrix
    % -----------------------------
    for co = 1:c_out
        for o2 = 1:out2
            for o1 = 1:out1

                row = sub2ind([c_out, out1, out2], co, o1, o2);

                for ci = 1:c_in
                    for b = 1:k2
                        for a = 1:k1

                            % Coordinates in padded input
                            p1 = (o1 - 1) * stride1 + a;
                            p2 = (o2 - 1) * stride2 + b;

                            % Convert padded coordinates to original input coordinates
                            i1 = p1 - pad1;
                            i2 = p2 - pad2;

                            % Skip entries falling in zero-padding region
                            if i1 >= 1 && i1 <= in1 && i2 >= 1 && i2 <= in2

                                col = sub2ind([c_in, in1, in2], ci, i1, i2);

                                count = count + 1;
                                rows(count) = row;
                                cols(count) = col;

                                % Kernel convention: [c_out, c_in, k1, k2]
                                vals(count) = kernel(co, ci, a, b);
                            end
                        end
                    end
                end
            end
        end
    end

    rows = rows(1:count);
    cols = cols(1:count);
    vals = vals(1:count);

    Wfc = sparse(rows, cols, vals, nOut, nIn);
end