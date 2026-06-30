close all
clear all
clc

%{
Disclaimer:
This script compares several methods for estimating or bounding the Lipschitz
constant of neural networks, namely GLipSDP, LipSDP/CLipSDP, SeqLip,
LipOpt-SDP, and LipLT.
%}

addpath('/Applications/mosek/11.0/toolbox/r2022bom')

ind_load = 1;

%% Define neural network

if ind_load == 1
    load('exports_archive/cnn2_weights.mat')

    input_1 = 8;
    input_2 = 8;

    K = double(matlab_conv1_weight);
    W_all{2} = double(matlab_fc1_weight');
    b_K{1} = double(matlab_conv1_bias');
    b_all{2} = double(matlab_fc1_bias');

    W{2} = W_all{2}(8,:); % output for entry 8 selected
    b{2} = b_all{2}(8); % output for entry 8 selected

    [k1, k2, c1, c2] = size(K);
    [n3, n2] = size(W{2});

    stride = 1;
    padding = 1;

else
    input_1 = 10;
    input_2 = 10;

    c1 = 1;
    c2 = 8;
    n3 = 1;

    k1 = 3;
    k2 = 3;

    K = randn(k1, k2, c1, c2) / sqrt(c1);
    stride = 2;
    padding = 1;

end

inputSize = [c1, input_1, input_2];

[W_sparse, outSize] = conv_layer_to_fc(K, inputSize, stride, padding);
W{1} = full(W_sparse);
b{1} = conv_bias_to_fc(b_K{1}, outSize);

[n2,n1] = size(W{1});

if ind_load == 1
    if size(W{1},1) ~= size(W{2},2)
        error('Dimension mismatch: size(W{1},1) must equal size(W{2},2). Got %d and %d.', ...
            size(W{1},1), size(W{2},2));
    end
else
    W{2} = randn(n3,n2) / sqrt(n2);
end

%%
[Lip_MP, time_MP] = naive_lip_mlp(W)

%%
lb = zeros(n1,1);
ub = ones(n1,1);

numSamples = 100000;

[Lip_lb, bestX, bestPattern, info] = relu_lipschitz_l2_lower_bound_sampling(W, b, lb, ub, numSamples)

%%
opts = struct();
opts.bruteMaxDim = 20;
opts.verbose = true;

[Lip_SepLip, time_SepLip, info_SeqLip] = seqlip(W, opts)

%%

alpha = 0;
beta = 1;

opts = struct();
opts.verbose = true;

[Lip_LT, time_LT, info_LT] = liplt_mlp(W, alpha, beta, opts)

%%
[Lip_LipSDP, info_LipSDP, time_LipSDP] = LipSDP(W)

%%
%NN = init_NN;
%NN.layers = {'fc','fc'};
%NN.weights = W;
%NN.Alpha = {0,0};
%NN.Beta = {1,1};

%[Lip_GLipSDP, info_GLipSDP, time_GLipSDP] = GLipSDP(NN)

%%
NN = init_NN;
NN.layers = {'conv','fc'};
NN.weights = {permute(K, [4, 3, 1, 2]),W{2}};
NN.Alpha = {0,0};
NN.Beta = {1,1};
NN.pool = {'none','none'};
NN.strides      = stride;
NN.pool_strides = [0];
NN.pool_kernel  = [0];
NN.glob_loc = 'global';
NN.padding = padding;

[Lip_GLipSDP_c, info_GLipSDP_c, time_GLipSDP_c] = GLipSDP(NN)

%%

[Lip_LiPopt, time_LiPopt] = lipopt_l2_shor(W);
