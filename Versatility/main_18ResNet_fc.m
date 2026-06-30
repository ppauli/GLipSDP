close all
clear all
clc

%load('models/weights_bias_mnist_resnetfc.mat')
%savepath = 'results/mnist_ResNet_fc_wd.mat';

load('models/weights_mat/FC-R18_instance1.mat')

savepath = 'results/mnist_resnet_fc_instance1.mat';

W{1} = double(input_layer_weight);
W{2}{1} = double(blocks_0_fc1_weight);
W{2}{2} = double(blocks_0_fc2_weight);
W{3}{1} = double(blocks_1_fc1_weight);
W{3}{2} = double(blocks_1_fc2_weight);
W{4}{1} = double(blocks_2_fc1_weight);
W{4}{2} = double(blocks_2_fc2_weight);
W{5}{1} = double(blocks_3_fc1_weight);
W{5}{2} = double(blocks_3_fc2_weight);
W{6}{1} = double(blocks_4_fc1_weight);
W{6}{2} = double(blocks_4_fc2_weight);
W{7}{1} = double(blocks_5_fc1_weight);
W{7}{2} = double(blocks_5_fc2_weight);
W{8}{1} = double(blocks_6_fc1_weight);
W{8}{2} = double(blocks_6_fc2_weight);
W{9}{1} = double(blocks_7_fc1_weight);
W{9}{2} = double(blocks_7_fc2_weight);
W{10} = double(output_layer_weight);

b{1} = double(input_layer_bias);
b{2}{1} = double(blocks_0_fc1_bias);
b{2}{2} = double(blocks_0_fc2_bias);
b{3}{1} = double(blocks_1_fc1_bias);
b{3}{2} = double(blocks_1_fc2_bias);
b{4}{1} = double(blocks_2_fc1_bias);
b{4}{2} = double(blocks_2_fc2_bias);
b{5}{1} = double(blocks_3_fc1_bias);
b{5}{2} = double(blocks_3_fc2_bias);
b{6}{1} = double(blocks_4_fc1_bias);
b{6}{2} = double(blocks_4_fc2_bias);
b{7}{1} = double(blocks_5_fc1_bias);
b{7}{2} = double(blocks_5_fc2_bias);
b{8}{1} = double(blocks_6_fc1_bias);
b{8}{2} = double(blocks_6_fc2_bias);
b{9}{1} = double(blocks_7_fc1_bias);
b{9}{2} = double(blocks_7_fc2_bias);
b{10} = double(output_layer_bias);

%% GLipSDP

NN = init_NN;
NN.layers = {'fc','res_fc2','res_fc2','res_fc2','res_fc2','res_fc2',...
    'res_fc2','res_fc2','res_fc2','fc'};

NN.weights = W;
NN.biases = b;

xL = zeros(784,1); % Lower bounds for input
xU = ones(784,1); % Upper bounds for input

[L, U, ibp] = IBP_NN(NN, xL, xU);
[Alpha, Beta, slopeVectors] = buildAlphaBetaChannelwiseConv(ibp);

NN.Alpha = Alpha;
NN.Beta = Beta;

[Lip_GLipSDP,info_GLipSDP,time_GLipSDP] = GLipSDP(NN);

Lip_GLipSDP

%%

NN.glob_loc = 'local';

[Lip_GLipSDP_l,info_GLipSDP_l,time_GLipSDP_l] = GLipSDP(NN);

Lip_GLipSDP_l


%% S-GLipSDP

NNfirst = init_NN;
NNfirst.layers = {'fc'};
NNfirst.weights{1} = W{1};
NNfirst.Alpha = Alpha{1};
NNfirst.Beta = Beta{2};

[Lip1,info1,time1] = GLipSDP(NNfirst);
Lip_S_GLipSDP = Lip1;

NN2 = init_NN;
NN2.layers = {'res_fc2'};
for ii = 2:9
    NN2.weights{1} = W{ii};
    NN2.Alpha = Alpha{ii};
    NN2.Beta = Beta{ii};
    [Lip2(ii-1),info2{ii-1},time2(ii-1)] = GLipSDP(NN2);
    Lip_S_GLipSDP = Lip_S_GLipSDP*Lip2(ii-1);
end

NNlast = init_NN;
NNlast.layers = {'fc'};
NNlast.weights{1} = W{end};
NNlast.Alpha = {};
NNlast.Beta = {};
[Lip3,info3,time3] = GLipSDP(NNlast);
Lip_S_GLipSDP = Lip_S_GLipSDP*Lip3;
time_S_GLipSDP = time1+sum(time2)+time3;

Lip_S_GLipSDP

%% S-LipSDP

tic
Lip_S_LipSDP = norm(W{1});
time1 = toc; 

for ii = 2:9
    [Lip_LipSDP(ii-1),info_LipSDP{ii-1},time_LipSDP(ii-1)] = LipSDP(W{ii});
end

for ii = 1:8
    Lip_S_LipSDP = Lip_S_LipSDP*(Lip_LipSDP(ii)+1);
end

tic
Lip_S_LipSDP = Lip_S_LipSDP*norm(W{10});
time2 = toc;
time_S_LipSDP = time1 + time2 + sum(time_LipSDP);

Lip_S_LipSDP

%% MP

Lip_MP = norm(W{1});

for ii = 2:9
    Lip_MP_s(ii-1) = 1+norm(W{ii}{1})*norm(W{ii}{2});
    Lip_MP = Lip_MP*Lip_MP_s(ii-1);
end

Lip_MP = Lip_MP*norm(W{end})

%% Save results
save(savepath)