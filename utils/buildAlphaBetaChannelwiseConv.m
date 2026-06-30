function [Alpha, Beta, slopeVectors] = buildAlphaBetaChannelwiseConv(ibp, row, col)
%BUILDALPHABETACHANNELWISECONV Build ReLU state Alpha/Beta from IBP bounds.
%
% For convolutional ReLU layers:
%   one Alpha/Beta value per output channel, extracted at spatial index
%   (row, col).
%
% For fully connected ReLU layers:
%   one Alpha/Beta value per neuron.
%
% ReLU state encoding:
%
%   inactive:  U <= 0      -> alpha = 0, beta = 0
%   active:    L >= 0      -> alpha = 1, beta = 1
%   unstable:  L < 0 < U   -> alpha = 0, beta = 1
%
% Inputs:
%   ibp.preL{i}, ibp.preU{i} : pre-ReLU bounds
%   ibp.hasReLU{i}           : true if layer i has ReLU
%   ibp.layerType{i}         : 'conv' or 'fc'
%
% Optional:
%   row, col : spatial index for channelwise conv slopes
%
% Outputs:
%   Alpha{k}, Beta{k} : sparse diagonal matrices
%   slopeVectors      : struct with raw alpha/beta vectors

if nargin < 2
    row = 2;
end

if nargin < 3
    col = 2;
end

Alpha = {};
Beta = {};

alphaVectors = {};
betaVectors = {};
sourceLayers = [];

reluCounter = 0;
numLayers = length(ibp.preL);

for i = 1:numLayers

    if isempty(ibp.preL{i})
        continue;
    end

    if isfield(ibp, "hasReLU")
        if ~ibp.hasReLU{i}
            continue;
        end
    end

    reluCounter = reluCounter + 1;

    L = ibp.preL{i};
    U = ibp.preU{i};

    switch ibp.layerType{i}

        case 'conv'
            [a, b] = channelSlopeFromPreReluTensor(L, U, row, col);

        case 'fc'
            [a, b] = neuronSlopeFromPreReluVector(L, U);

        case 'res_fc2'
            % For residual FC block:
            %
            %   y = x + W2 * ReLU(W1*x + b1) + b2
            %
            % ibp.preL{i}, ibp.preU{i} should store the bounds before
            % the internal ReLU, i.e. W1*x + b1.
            %
            % Therefore this is vector-valued, like an FC ReLU.
            [a, b] = neuronSlopeFromPreReluVector(L, U);

        case 'res_conv2'
            % Internal ReLU of:
            %   X + Conv2(ReLU(Conv1(X)))
            %
            % ibp.preL{i}, ibp.preU{i} store Conv1 pre-ReLU bounds.
            [a, b] = channelSlopeFromPreReluTensor(L, U, row, col);

        otherwise
            error("Unsupported layer type: %s", ibp.layerType{i});
    end

    Alpha{reluCounter, 1} = spdiags(a, 0, numel(a), numel(a));
    Beta{reluCounter, 1}  = spdiags(b, 0, numel(b), numel(b));

    alphaVectors{reluCounter, 1} = a;
    betaVectors{reluCounter, 1} = b;
    sourceLayers(reluCounter, 1) = i;
end

slopeVectors = struct();
slopeVectors.alpha = alphaVectors;
slopeVectors.beta = betaVectors;
slopeVectors.sourceLayers = sourceLayers;
slopeVectors.mode = "channelwise-conv-at-fixed-spatial-index";
slopeVectors.spatialIndex = [row, col];

end