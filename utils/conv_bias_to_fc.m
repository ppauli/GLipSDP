function bfc = conv_bias_to_fc(b, outSize)
%CONV_BIAS_TO_FC Convert convolutional bias to equivalent FC bias vector.
%
%   bfc = conv_bias_to_fc(b, outSize)
%
% INPUTS:
%   b:
%       Bias vector for the convolutional layer.
%       Must satisfy numel(b) = c_out.
%
%   outSize:
%       Output size of the convolutional layer:
%
%           [c_out, out1, out2]
%
% OUTPUT:
%   bfc:
%       Bias vector of size [c_out*out1*out2, 1], arranged consistently
%       with y(:), where y has size [c_out, out1, out2].
%
% CONVENTION:
%   If
%
%       y(co, o1, o2) = ... + b(co),
%
%   then
%
%       y(:) = ... + bfc.
%
%   MATLAB linearizes [c_out, out1, out2] with co varying fastest, so
%   the correct stacking is
%
%       [b; b; ...; b]
%
%   repeated out1*out2 times.

    % -----------------------------
    % Parse dimensions
    % -----------------------------
    if numel(outSize) ~= 3
        error('outSize must be [c_out, out1, out2].');
    end

    c_out = outSize(1);
    out1  = outSize(2);
    out2  = outSize(3);

    if numel(b) ~= c_out
        error('Bias b must satisfy numel(b) = c_out.');
    end

    % Force column vector
    b = b(:);

    % -----------------------------
    % Stack bias
    % -----------------------------
    bfc = repmat(b, out1 * out2, 1);
end