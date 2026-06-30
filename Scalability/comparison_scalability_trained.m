% This code compares LipSDP and LipSDP-conv
% using trained CNN weights instead of randomly generated / stored synthetic weights.

close all
clear all
clc

l1 = 3;      % kernel size
l2 = l1;     % kernel size
N1 = 14;     % input image size
N2 = 14;     % input image size

c_vec = [8, 16, 32];          % channel sizes
depth = [2, 4, 8, 16];        % number of convolutional weights used
model_depth = depth + 1;      % file names: C3, C5, C9, C17

for jj = 1:length(c_vec)
    jj

    channels = c_vec(jj);

    for kk = 1:length(depth)
        kk

        clear K W params

        d = depth(kk);              % number of convolutional weights
        C = model_depth(kk);        % model depth used in filename

        filename = sprintf('exported_mnist_models/cnn_C%d_CH%d.mat', C, channels);
        load(filename, 'params');

        %% Load trained convolutional weights
        K = cell(1, d);

        for ii = 1:d
            weight_name = sprintf('conv_%d_weight', ii);
            K{ii} = double(params.(weight_name));
        end

        %% Optional: inspect dimensions
        for ii = 1:d
            fprintf("K{%d}: ", ii);
            disp(size(K{ii}));
        end

        %% GLipSDP
        NN = init_NN;

        NN.layers = repmat({'conv'}, 1, d);
        NN.pool = repmat({'none'}, 1, d);
        NN.strides = ones(1, d);
        NN.pool_strides = zeros(1, d);
        NN.pool_kernel = zeros(1, d);
        NN.Alpha = cell(1, length(K));
        NN.Beta = cell(1, length(K));

        NN.weights = K;

        [L_SDP, info_SDP, time_SDP] = GLipSDP(NN);
        L_SDP_mat(jj,kk) = L_SDP
        info_SDP_mat{jj,kk} = info_SDP
        time_SDP_mat(jj,kk) = time_SDP

        save("results/results_all_CNN_trained.mat")

        %% CLipSDP
        [L_SDP_sparse, info_SDP_sparse, time_SDP_sparse] = CLipSDP(K);
        L_SDP_sparse_mat(jj,kk) = L_SDP_sparse
        info_SDP_sparse_mat{jj,kk} = info_SDP_sparse
        time_SDP_sparse_mat(jj,kk) = time_SDP_sparse

        save("results/results_all_CNN_trained.mat")

    end
end

save("results/results_all_CNN_trained.mat")