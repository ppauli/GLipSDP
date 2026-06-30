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
    load('exports/cnn3_weights.mat')

    input_1 = 8;
    input_2 = 8;

    K{1} = double(matlab_conv1_weight);
    K{2} = double(matlab_conv2_weight);
    W_all{3} = double(matlab_fc1_weight');
    b_K{1} = double(matlab_conv1_bias');
    b_K{2} = double(matlab_conv2_bias');
    b_all{3} = double(matlab_fc1_bias');

    W{3} = W_all{3}(8,:); % output for entry 8 selected
    b{3} = b_all{3}(8); % output for entry 8 selected
    
    [k1, k2, c1(1), c2(1)] = size(K{1});

    stride = [1,1];
    padding = [1,1];

else

input_1 = 8;
input_2 = 8;

c1 = [1, 4];
c2 = [4, 4];
n4 = 1;

k1 = [3, 3];
k2 = [3, 3];

K{1} = randn(c2(1), c1(1), k1(1), k2(1)) / sqrt(c1(1));
K{2} = randn(c2(2), c1(2), k1(2), k2(2)) / sqrt(c1(2));

stride = [2,2];
padding = [1,1];

end

inputSize = [c1(1), input_1, input_2];

[W_sparse{1}, outSize{1}] = conv_layer_to_fc(K{1}, inputSize, stride(1), padding(1));
W{1} = full(W_sparse{1});
b{1} = conv_bias_to_fc(b_K{1}, outSize{1});

[W_sparse{2}, outSize{2}] = conv_layer_to_fc(K{2}, outSize{1}, stride(2), padding(2));
W{2} = full(W_sparse{2});
b{2} = conv_bias_to_fc(b_K{2}, outSize{2});

[n2,n1] = size(W{1});
[n4, n3] = size(W{3});

if ind_load == 1
    if size(W{2},1) ~= size(W{3},2)
        error('Dimension mismatch: size(W{1},1) must equal size(W{2},2). Got %d and %d.', ...
          size(W{2},1), size(W{3},2));
    end
else
    W{3} = randn(n4,n3) / sqrt(n3);
end

%%
[Lip_MP, time_MP] = naive_lip_mlp(W)

%%
lb = zeros(n1,1);
ub = ones(n1,1);

numSamples = 10000;

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
% NN = init_NN;
% NN.layers = {'fc','fc','fc'};
% NN.weights = W;
% NN.Alpha = {0,0,0};
% NN.Beta = {1,1,1};
% 
% [Lip_GLipSDP, info_GLipSDP, time_GLipSDP] = GLipSDP(NN)

%%
NN = init_NN;
NN.layers = {'conv','conv','fc'};
NN.weights = {permute(K{1},[4, 3, 1, 2]),permute(K{2},[4, 3, 1, 2]),W{3}};
NN.Alpha = {0,0,0};
NN.Beta = {1,1,1};
NN.pool = {'none','none'};
NN.strides      = stride;
NN.pool_strides = [0,0];
NN.pool_kernel  = [0,0];
NN.glob_loc = 'global';
NN.padding = padding;

[Lip_GLipSDP_c, info_GLipSDP_c, time_GLipSDP_c] = GLipSDP(NN)

%%

[Lip_LiPopt, time_LiPopt] = lipopt_l2_shor(W);