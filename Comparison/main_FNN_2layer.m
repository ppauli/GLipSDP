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
    load('exports_archive/fnn2_weights.mat')

    W{1} = double(matlab_fc1_weight');
    W_all{2} = double(matlab_fc2_weight');
    b{1} = double(matlab_fc1_bias');
    b_all{2} = double(matlab_fc2_bias');

    W{2} = W_all{2}(8,:); % output for entry 8 selected
    b{2} = b_all{2}(8); % output for entry 8 selected

    [n2,n1] = size(W{1});
    [n3,~] = size(W{2});

else

    n1 = 64;
    n2 = 32;
    n3 = 1;

    seed = 42;
    rng(seed);

    W{1} = randn(n2,n1) / sqrt(n1);
    W{2} = randn(n3,n2) / sqrt(n2);
    b{1} = randn(n2,1) / sqrt(n1);
    b{2} = randn(n3,1) / sqrt(n2);

    rng('shuffle')
end

%%
[Lip_MP, time_MP] = naive_lip_mlp(W)

%%
lb = zeros(n1,1);
ub = ones(n1,1);

numSamples = 100000;

[Lip_lb, bestX, bestPattern, info] = relu_lipschitz_l2_lower_bound_sampling(W, b, lb, ub, numSamples)

%%

xCenter = zeros(n1,1);
radius = 10;
relaxOrder = 2;

[L_ub, diagnostics, aux] = relu_lcep_l2_moment_1hidden(W, b, xCenter, radius, relaxOrder)

%%

%[L_FGL, bestPattern, info] = relu_fgl_l2(W)

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
NN.layers = {'fc','fc'};
NN.weights = W;
NN.Alpha = {0,0};
NN.Beta = {1,1};

[Lip_GLipSDP, info_GLipSDP, time_GLipSDP] = GLipSDP(NN)

%%

[Lip_LiPopt, time_LiPopt] = lipopt_l2_shor(W);