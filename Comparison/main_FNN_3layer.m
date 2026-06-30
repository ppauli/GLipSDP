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
    load('exports/fnn3_weights.mat')

    W{1} = double(matlab_fc1_weight');
    W{2} = double(matlab_fc2_weight');
    W_all{3} = double(matlab_fc3_weight');
    b{1} = double(matlab_fc1_bias');
    b{2} = double(matlab_fc2_bias');
    b_all{3} = double(matlab_fc3_bias');

    W{3} = W_all{3}(8,:); % output for entry 8 selected
    b{3} = b_all{3}(8); % output for entry 8 selected

    [n2,n1] = size(W{1});
    [n4,n3] = size(W{3});

else

    n1 = 64;
    n2 = 32;
    n3 = 32;
    n4 = 1;

    W{1} = randn(n2,n1) / sqrt(n1);
    W{2} = randn(n3,n2) / sqrt(n2);
    W{3} = randn(n4,n3) / sqrt(n3);

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
[Lip_LipSDP, info_LipSDP , time_LipSDP] = LipSDP(W)

%%
NN = init_NN;
NN.layers = {'fc','fc','fc'};
NN.weights = W;
NN.Alpha = {0,0,0};
NN.Beta = {1,1,1};

[Lip_GLipSDP, info_GLipSDP, time_GLipSDP] = GLipSDP(NN)

%%

[Lip_LiPopt, time_LiPopt] = lipopt_l2_shor(W);



