close all
clear all
clc

% Load model
load('models/weights_mnist_2C2F_con.mat')

%% GLipSDP

NN = init_NN;

NN.layers = {'conv','conv','fc','fc'};

NN.weights = W;
NN.pool = {'none','none'};
NN.strides      = [2,2];
NN.pool_strides = [0,0];
NN.pool_kernel  = [0,0];

[Lip_GLipSDP,info_GlipSDP,time_GLipSDP] = LipEst(NN);

Lip_GLipSDP

%% S-GLipSDP

NNconv = init_NN;

NNconv.layers = {'conv','conv'};

NNconv.weights = W(1:2);
NNconv.pool = {'none','none'};
NNconv.strides      = [2,2];
NNconv.pool_strides = [0,0];
NNconv.pool_kernel  = [0,0];

NNfc = init_NN;

NNfc.layers = {'fc','fc'};

NNfc.weights = W(3:4);

[Lip_conv,info_conv,time_conv] = LipEst(NNconv);

[Lip_fc,info_fc,time_fc] = LipEst(NNfc);

Lip_S_GLipSDP = Lip_conv*Lip_fc;
time_S_GLipSDP = time_conv+time_fc;

Lip_S_GLipSDP

%% S-LipSDP

W_c2fc{1} = conv2fc(W{1},30,30,2,1);
W_c2fc{2} = conv2fc(W{2},16,16,2,0);

[Lip1_LipSDP,info1_LipSDP,time1_LipSDP] = LipschitzEstimationFazlyab(W_c2fc);
[Lip2_LipSDP,info2_LipSDP,time2_LipSDP] = LipschitzEstimationFazlyab(W(3:4));

Lip_S_LipSDP = Lip1_LipSDP*Lip2_LipSDP;
time_S_LipSDP = time1_LipSDP+time2_LipSDP;

Lip_S_LipSDP

save('results/mnist_main_2C2F.mat')

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
save('results/mnist_main_2C2F.mat')

