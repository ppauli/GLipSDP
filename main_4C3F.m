close all
clear all
clc

% Load model
load('models/weights_mnist_4C3F_con.mat')

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

tic
Lip2 = norm(W{5});
Lip3 = norm(W{6});
Lip4 = norm(W{7});
time_fc = toc;

Lip_S_GLipSDP = Lip_conv*Lip2*Lip3*Lip4;
time_S_GLipSDP = time_conv+time_fc;

Lip_S_GLipSDP

save('results/mnist_main_4C3F.mat')

%% MP

W_fc{1} = conv2fc(W{1},30,30,1,1);
W_fc{2} = conv2fc(W{2},30,30,2,1);
W_fc{3} = conv2fc(W{3},16,16,1,1);
W_fc{4} = conv2fc(W{4},16,16,2,0);

tic
for ii = 1:length(W_fc)
    Lip_MP_s(ii) = norm(W_c2fc{ii});
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
save('results/mnist_main_4C3F.mat')
