function [L, U, ibp] = IBP_NN(NN, xL, xU)
%IBP_NN Interval bound propagation through a CNN/FC network.
%
% Returns:
%   L, U : final output lower and upper bounds
%   ibp  : struct containing intermediate bounds
%
% ibp.preL{i}, ibp.preU{i}   : bounds before ReLU at layer i
% ibp.postL{i}, ibp.postU{i} : bounds after ReLU/pooling at layer i
% ibp.hasReLU{i}             : true if layer i has ReLU
% ibp.layerType{i}           : layer type

L = single(xL);
U = single(xU);

flattened = false;

numLayers = length(NN.layers);

ibp = struct;
ibp.preL = cell(numLayers, 1);
ibp.preU = cell(numLayers, 1);
ibp.postL = cell(numLayers, 1);
ibp.postU = cell(numLayers, 1);
ibp.hasReLU = cell(numLayers, 1);
ibp.layerType = cell(numLayers, 1);

for i = 1:numLayers

    layer_type = NN.layers{i};
    Wi = NN.weights{i};
    bi = NN.biases{i};

    ibp.layerType{i} = layer_type;

    switch layer_type

        case 'conv'

            stride  = NN.strides(i);
            padding = NN.padding(i);

            % Affine convolutional bounds before ReLU
            [preL, preU] = IBP_conv2d(L, U, Wi, bi, stride, padding);

            ibp.preL{i} = preL;
            ibp.preU{i} = preU;

            % ReLU after convolution
            [L, U] = IBP_relu(preL, preU);

            ibp.hasReLU{i} = true;

            % Optional pooling after ReLU
            if isfield(NN, 'pool') && ~strcmp(NN.pool{i}, 'none')

                pool_type = NN.pool{i};

                switch pool_type

                    case 'max'
                        [L, U] = IBP_maxpool2d( ...
                            L, U, ...
                            NN.pool_kernel(i), ...
                            NN.pool_strides(i), ...
                            NN.pool_padding(i));

                    case {'avg', 'av'}
                        [L, U] = IBP_avgpool2d( ...
                            L, U, ...
                            NN.pool_kernel(i), ...
                            NN.pool_strides(i));

                    otherwise
                        error('Unknown pooling type: %s', pool_type);
                end
            end

            ibp.postL{i} = L;
            ibp.postU{i} = U;

        case 'fc'

            if ~flattened
                L = permute(L, [2 3 1]);   % CHW -> HWC
                U = permute(U, [2 3 1]);
                L = L(:);
                U = U(:);
                flattened = true;
            end

            % Affine fully connected bounds before ReLU
            [preL, preU] = IBP_fc(L, U, Wi, bi);

            ibp.preL{i} = preL;
            ibp.preU{i} = preU;

            % ReLU after all FC layers except final output layer
            if i < numLayers
                [L, U] = IBP_relu(preL, preU);
                ibp.hasReLU{i} = true;
            else
                L = preL;
                U = preU;
                ibp.hasReLU{i} = false;
            end

            ibp.postL{i} = L;
            ibp.postU{i} = U;

        case 'res_fc2'

            % Residual FC block:
            %
            %   y = x + W2 * ReLU(W1*x + b1) + b2

            W1 = NN.weights{i}{1};
            W2 = NN.weights{i}{2};

            b1 = NN.biases{i}{1};
            b2 = NN.biases{i}{2};

            [Lout, Uout, cache] = IBP_res_fc_identity(L, U, W1, b1, W2, b2);

            % Store internal pre-ReLU bounds of the residual branch
            ibp.preL{i} = cache.preL1;
            ibp.preU{i} = cache.preU1;

            % Store final residual-block output
            L = Lout;
            U = Uout;

            ibp.postL{i} = L;
            ibp.postU{i} = U;

            % The block contains an internal ReLU
            ibp.hasReLU{i} = true;
            ibp.resCache{i} = cache;

        case 'res_conv2'

            % Residual conv block:
            %
            %   Y = X + Conv2(ReLU(Conv1(X)))
            %
            % Optional pooling can be applied after the residual block:
            %
            %   Y = Pool(X + Conv2(ReLU(Conv1(X))))
            %
            % Expected:
            %
            %   NN.weights{i}{1} = W1
            %   NN.weights{i}{2} = W2
            %
            %   NN.biases{i}{1}  = b1
            %   NN.biases{i}{2}  = b2

            W1 = NN.weights{i}{1};
            W2 = NN.weights{i}{2};

            b1 = NN.biases{i}{1};
            b2 = NN.biases{i}{2};

            % Default: residual convs preserve spatial resolution
            stride1 = 1;
            stride2 = 1;
            padding1 = 1;
            padding2 = 1;

            if isfield(NN, 'res_strides') && numel(NN.res_strides) >= i ...
                    && ~isempty(NN.res_strides{i})
                stride1 = NN.res_strides{i}(1);
                stride2 = NN.res_strides{i}(2);
            end

            if isfield(NN, 'res_padding') && numel(NN.res_padding) >= i ...
                    && ~isempty(NN.res_padding{i})
                padding1 = NN.res_padding{i}(1);
                padding2 = NN.res_padding{i}(2);
            end

            [Lout, Uout, cache] = IBP_res_conv_identity( ...
                L, U, W1, b1, W2, b2, stride1, padding1, stride2, padding2);

            % Store bounds before the internal ReLU Conv1 -> ReLU -> Conv2
            ibp.preL{i} = cache.preL1;
            ibp.preU{i} = cache.preU1;

            % Final residual-block output after skip addition
            L = Lout;
            U = Uout;

            % Optional pooling after residual convolutional block
            if isfield(NN, 'pool') && numel(NN.pool) >= i ...
                    && ~isempty(NN.pool{i}) && ~strcmp(NN.pool{i}, 'none')

                pool_type = NN.pool{i};

                switch pool_type

                    case 'max'
                        [L, U] = IBP_maxpool2d( ...
                            L, U, ...
                            NN.pool_kernel(i), ...
                            NN.pool_strides(i), ...
                            NN.pool_padding(i));

                    case {'avg', 'av'}
                        [L, U] = IBP_avgpool2d( ...
                            L, U, ...
                            NN.pool_kernel(i), ...
                            NN.pool_strides(i));

                    otherwise
                        error('Unknown pooling type after res_conv2 layer %d: %s', ...
                            i, pool_type);
                end
            end

            % Store final output after optional pooling
            ibp.postL{i} = L;
            ibp.postU{i} = U;

            % The block contains one internal ReLU
            ibp.hasReLU{i} = true;

            if ~isfield(ibp, 'resCache')
                ibp.resCache = cell(numLayers, 1);
            end
            ibp.resCache{i} = cache;

        otherwise
            error('Unknown layer type: %s', layer_type);
    end
end
end