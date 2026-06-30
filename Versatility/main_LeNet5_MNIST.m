close all
clear all
clc

% Load model
%load(['models_archive/weights_bias_mnist_lenet5.mat'])
%savepath = 'results/mnist_LeNet5_wd.mat';

% Load weights and biases
load('models_LeNet5/weights_mat/LeNet5ReLU_instance1_4.mat');
savepath = 'results/mnist_LeNet5_instance1_4.mat';

%%

W{1} = double(net_0_weight);
W{2} = double(net_3_weight);
W{3} = double(net_7_weight);
W{4} = double(net_9_weight);
W{5} = double(net_11_weight);
b{1} = double(net_0_bias');
b{2} = double(net_3_bias');
b{3} = double(net_7_bias');
b{4} = double(net_9_bias');
b{5} = double(net_11_bias');


%% GLipSDP

NN = init_NN;

NN.layers = {'conv','conv','fc','fc','fc'};

NN.weights = W;
NN.biases = b;
NN.pool = {'av','av'};
NN.strides      = [1,1];
NN.pool_strides = [2,2];
NN.pool_kernel  = [2,2];
NN.glob_loc = 'global';
NN.padding = [1,1];

xL = zeros(1, 28, 28);
xU = ones(1, 28, 28);

[L, U, ibp] = IBP_NN(NN, xL, xU);
[Alpha, Beta, slopeVectors] = buildAlphaBetaChannelwiseConv(ibp);

NN.Alpha = Alpha;
NN.Beta = Beta;

[Lip_GLipSDP,info_GLipSDP,time_GLipSDP] = GLipSDP(NN);

Lip_GLipSDP

NN.glob_loc = 'local';

[Lip_GLipSDP_l,info_GLipSDP_l,time_GLipSDP_l] = GLipSDP(NN);

Lip_GLipSDP_l


save(savepath)

%% S-GLipSDP

NNconv = init_NN;

NNconv.layers = {'conv','conv'};

NNconv.weights = W(1:2);
NNconv.pool = {'av','av'};
NNconv.strides      = [1,1];
NNconv.pool_strides = [2,2];
NNconv.pool_kernel  = [2,2];
NNconv.Alpha = Alpha(1:2);
NNconv.Beta = Beta(1:2);

NNfc = init_NN;

NNfc.layers = {'fc','fc','fc'};
NNfc.weights = W(3:5);
NNfc.Alpha = Alpha(3:4);
NNfc.Beta = Beta(3:4);

[Lip_conv,info_conv,time_conv] = GLipSDP(NNconv);

[Lip_fc,info_fc,time_fc] = GLipSDP(NNfc);

Lip_S_GLipSDP = Lip_conv*Lip_fc;
time_S_GLipSDP = time_conv+time_fc;

Lip_S_GLipSDP

save(savepath)

%% S-LipSDP

W_sparse{1} = conv_layer_to_fc(W{1},[1,32,32],NN.strides(1), NN.padding(1));
W_c2fc{1} = full(W_sparse{1})
W_sparse{2} = conv_layer_to_fc(W{2},[6,14,14],NN.strides(2), NN.padding(2));
W_c2fc{2} = full(W_sparse{2})
tic
Lip1_LipSDP = norm(W_c2fc{1})
Lip1_LipSDP = Lip1_LipSDP*norm(W_c2fc{2})
time1_LipSDP = toc;

[Lip2_LipSDP,info2_LipSDP,time2_LipSDP] = LipSDP(W(3:5));

Lip_S_LipSDP = Lip1_LipSDP*Lip2_LipSDP;
time_S_LipSDP = time1_LipSDP+time2_LipSDP;

Lip_S_LipSDP= Lip_S_LipSDP*0.5*0.5

save(savepath)

%% MP

Lip_MP_s(1) = norm(W_c2fc{1});
Lip_MP_s(2) = norm(W_c2fc{2});
Lip_MP_s(3) = norm(W{3});
Lip_MP_s(4) = norm(W{4});
Lip_MP_s(5) = norm(W{5});

Lip_MP = 1;
for ii = 1:5
    Lip_MP = Lip_MP*Lip_MP_s(ii);
end

Lip_MP=Lip_MP*0.5*0.5


%% Save results
save(savepath)