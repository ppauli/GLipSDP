close all
clear all
clc

% Load model
load('models/weights_mnist_4C3F_wd_8_con.mat');

savepath = 'results/mnist_4C3F_wd_8.mat';

%% GLipSDP
% 
% NN = init_NN;
% 
% NN.layers = {'conv','conv','conv','conv','fc','fc','fc'};
% 
% NN.weights = W;
% NN.pool = {'none','none','none','none'};
% NN.strides      = [1,2,1,2];
% NN.pool_strides = [0,0,0,0];
% NN.pool_kernel  = [0,0,0,0];
% 
% [Lip_GLipSDP,info_GlipSDP,time_GLipSDP] = LipEst(NN);
% 
% Lip_GLipSDP

%% S-GLipSDP

NNconv = init_NN;

NNconv.layers = {'conv','conv','conv','conv'};

NNconv.weights = W(1:3);
NNconv.pool = {'none','none','none','none'};
NNconv.strides      = [1,2,1,2];
NNconv.pool_strides = [0,0,0,0];
NNconv.pool_kernel  = [0,0,0,0];

[Lip_conv,info_conv,time_conv] = LipEst(NNconv);

%tic
%Lip_fc1 = norm(W{5});
%time_fc1 = toc;

NNfc = init_NN;
NNfc.layers = {'subn_fc'};

W2{1}{1} = W{5};
W2{1}{2} = W{6};
W2{1}{3} = W{7};
NNfc.weights = W2;

[Lip_fc,info_fc,time_fc] = LipEst(NNfc);

Lip_S_GLipSDP = Lip_conv*Lip_fc;
time_S_GLipSDP = time_conv+time_fc;

Lip_S_GLipSDP

save(savepath)

%% MP

W_fc{1} = conv2fc(W{1},30,30,1,1);
W_fc{2} = conv2fc(W{2},30,30,2,1);
W_fc{3} = conv2fc(W{3},16,16,1,1);
W_fc{4} = conv2fc(W{4},16,16,2,0);

tic
for ii = 1:length(W_fc)
    Lip_MP_s(ii) = norm(W_fc{ii});
end
for ii = length(W_fc)+1:length(W)
    Lip_MP_s(ii) = norm(W{ii});
end
time_norms = toc;

Lip_MP = 1;
for ii = 1:length(W)
    Lip_MP = Lip_MP*Lip_MP_s(ii);
end

Lip_MP

save(savepath)

%% S-LipSDP

[Lip2_LipSDP,info2_LipSDP,time2_LipSDP] = LipschitzEstimationFazlyab(W(6:7));

Lip_S_LipSDP = 1;
for ii = 1:length(W)-2
    Lip_S_LipSDP = Lip_S_LipSDP*Lip_MP_s(ii);
end

Lip_S_LipSDP = Lip_S_LipSDP*Lip2_LipSDP;
time_S_LipSDP = time2_LipSDP+time_norms;

Lip_S_LipSDP

%% Save results
save(savepath)
