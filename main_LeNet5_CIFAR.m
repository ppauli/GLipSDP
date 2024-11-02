close all
clear all
clc

% Load model
load(['models/weights_cifar_LeNet5_wd_con.mat'])

savepath = 'results/cifar_LeNet5_wd.mat';

%% GLipSDP

NN = init_NN;

NN.layers = {'conv','conv','fc','fc','fc'};

NN.weights = W;
NN.pool = {'max','max'};
NN.strides      = [1,1];
NN.pool_strides = [2,2];
NN.pool_kernel  = [2,2];

[Lip_GLipSDP,info_GLipSDP,time_GLipSDP] = LipEst(NN);

Lip_GLipSDP

save(savepath)

%% S-GLipSDP

NNconv = init_NN;

NNconv.layers = {'conv','conv'};

NNconv.weights = W(1:2);
NNconv.pool = {'max','max'};
NNconv.strides      = [1,1];
NNconv.pool_strides = [2,2];
NNconv.pool_kernel  = [2,2];

NNfc = init_NN;

NNfc.layers = {'fc','fc','fc'};
NNfc.weights = W(3:5);

[Lip_conv,info_conv,time_conv] = LipEst(NNconv);

[Lip_fc,info_fc,time_fc] = LipEst(NNfc);

Lip_S_GLipSDP = Lip_conv*Lip_fc;
time_S_GLipSDP = time_conv+time_fc;

Lip_S_GLipSDP

save(savepath)

%% S-LipSDP

W_c2fc{1} = conv2fc(W{1},32,32,1,0);
W_c2fc{2} = conv2fc(W{2},14,14,1,0);
tic
Lip1_LipSDP = norm(W_c2fc{1})
Lip1_LipSDP = Lip1_LipSDP*norm(W_c2fc{2})
time1_LipSDP = toc;

[Lip2_LipSDP,info2_LipSDP,time2_LipSDP] = LipschitzEstimationFazlyab(W(3:5));

Lip_S_LipSDP = Lip1_LipSDP*Lip2_LipSDP;
time_S_LipSDP = time1_LipSDP+time2_LipSDP;

Lip_S_LipSDP

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

Lip_MP


%% Save results
save(savepath)