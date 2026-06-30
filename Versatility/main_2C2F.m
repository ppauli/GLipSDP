close all
clear all
clc

addpath('/Applications/mosek/11.0/toolbox/r2022bom')

% Load model
%load('models_archive/weights_bias_mnist_2C2F.mat')
%savepath = 'results/mnist_2C2F_wd.mat';

% Load weights and biases
load('models/weights_mat/2C2F_instance1_2.mat');
savepath = 'results/mnist_2C2F_instance1_2.mat';

%%

W{1} = double(features_0_conv_weight);
W{2} = double(features_2_conv_weight);
W{3} = double(classifier_1_weight);
W{4} = double(classifier_3_weight);
b{1} = double(features_0_conv_bias');
b{2} = double(features_2_conv_bias');
b{3} = double(classifier_1_bias');
b{4} = double(classifier_3_bias');

%% GLipSDP

NN = init_NN;

NN.layers = {'conv','conv','fc','fc'};

NN.weights = W;
NN.biases = b;
NN.pool = {'none','none'};
NN.strides      = [2,2];
NN.pool_strides = [0,0];
NN.pool_kernel  = [0,0];
NN.glob_loc = 'global';
NN.padding = [1,1];

xL = zeros(1, 28, 28);
xU = ones(1, 28, 28);

[L, U, ibp] = IBP_NN(NN, xL, xU);
[Alpha, Beta, slopeVectors] = buildAlphaBetaChannelwiseConv(ibp);

NN.Alpha = Alpha;
NN.Beta = Beta;

[Lip_GLipSDP,info_GlipSDP,time_GLipSDP] = GLipSDP(NN);

Lip_GLipSDP

%%

NN.glob_loc = 'local';

[Lip_GLipSDP_l,info_GLipSDP_l,time_GLipSDP_l] = GLipSDP(NN);

Lip_GLipSDP_l

save(savepath)

%% S-GLipSDP

NNconv = init_NN;

NNconv.layers = {'conv','conv'};

NNconv.weights = W(1:2);
NNconv.pool = {'none','none'};
NNconv.strides      = [2,2];
NNconv.pool_strides = [0,0];
NNconv.pool_kernel  = [0,0];
NNconv.Alpha = Alpha(1:2);
NNconv.Beta = Beta(1:2);

NNfc = init_NN;

NNfc.layers = {'fc','fc'};

NNfc.weights = W(3:4);
NNfc.Alpha = Alpha(3);
NNfc.Beta = Beta(3);

[Lip_conv,info_conv,time_conv] = GLipSDP(NNconv);

[Lip_fc,info_fc,time_fc] = GLipSDP(NNfc);

Lip_S_GLipSDP = Lip_conv*Lip_fc;
time_S_GLipSDP = time_conv+time_fc;

Lip_S_GLipSDP

save(savepath)

%% S-LipSDP

inputSize = [1, 30, 30];
stride = 2;
padding = 1;

[W_sparse{1}, outSize{1}] = conv_layer_to_fc(W{1}, inputSize, NN.strides(1), NN.padding(1));
W_c2fc{1} = full(W_sparse{1}); 

[W_sparse{2}, outSize{2}] = conv_layer_to_fc(W{2}, outSize{1}, NN.strides(2), NN.padding(2));
W_c2fc{2} = full(W_sparse{2});

[Lip1_LipSDP,info1_LipSDP,time1_LipSDP] = LipSDP(W_c2fc);
[Lip2_LipSDP,info2_LipSDP,time2_LipSDP] = LipSDP(W(3:4));

Lip_S_LipSDP = Lip1_LipSDP*Lip2_LipSDP;
time_S_LipSDP = time1_LipSDP+time2_LipSDP;

Lip_S_LipSDP

save(savepath)

%% MP

Lip_MP_s(1) = norm(W_c2fc{1});
Lip_MP_s(2) = norm(W_c2fc{2});
Lip_MP_s(3) = norm(W{3});
Lip_MP_s(4) = norm(W{4});

Lip_MP = 1;
for ii = 1:4
    Lip_MP = Lip_MP*Lip_MP_s(ii);
end

Lip_MP

%% Save results
save(savepath)

