% This code compares LipSDP, LipLT and GLipSDP
% using trained MLP weights instead of randomly generated weights.

close all
clear all
clc

addpath('/Applications/mosek/11.0/toolbox/r2022bom')

c_vec = [64, 32, 16];              % widths
depth = [2, 4, 8, 16, 32, 64];     % number of weight matrices used
model_depth = depth + 1;           % file names: L3, L5, L9, L17, L33, L65

for jj = 1:length(c_vec)
    jj

    width = c_vec(jj);

    for kk = 1:length(depth)
        kk

        clear K W params

        d = depth(kk);             % number of weights to load
        L = model_depth(kk);       % model depth used in filename

        filename = sprintf('exported_mnist_models/mlp_L%d_W%d.mat', L, width);
        load(filename, 'params');

        % Load trained weights
        W = cell(1, d);

        for ii = 1:d
            weight_name = sprintf('linear_%d_weight', ii);
            W{ii} = double(params.(weight_name)');
        end

        %% GLipSDP
        NN = init_NN;
        NN.Alpha = cell(1, length(W));
        NN.Beta = cell(1, length(W));
        NN.layers = {};

        for ii = 1:d
            NN.layers{ii} = 'fc';
        end

        NN.weights = W;

        [L_GLipSDP, info_GlipSDP, time_GLipSDP] = GLipSDP(NN);
        L_GLipSDP_mat(jj,kk) = L_GLipSDP
        info_GLipSDP_mat{jj,kk} = info_GlipSDP
        time_GLipSDP_mat(jj,kk) = time_GLipSDP

        %% GLipSDP Subnetworks
        NN = init_NN;
        NN.layers{1} = 'subn_fc';
        NN.weights{1} = W;
        NN.Alpha = cell(1, length(W));
        NN.Beta = cell(1, length(W));

        [L_GLipSDP_2, info_GlipSDP_2, time_GLipSDP_2] = GLipSDP(NN);
        L_GLipSDP_2_mat(jj,kk) = L_GLipSDP_2
        info_GLipSDP_2_mat{jj,kk} = info_GlipSDP_2
        time_GLipSDP_2_mat(jj,kk) = time_GLipSDP_2


        %% LipSDP
        [L_LipSDP, info_LipSDP, time_LipSDP] = LipSDP(W);
        L_LipSDP_mat(jj,kk) = L_LipSDP
        info_LipSDP_mat{jj,kk} = info_LipSDP
        time_LipSDP_mat(jj,kk) = time_LipSDP;

    end
end

save("results/results_all_FC_trained.mat")