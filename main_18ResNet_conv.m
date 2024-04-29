close all
clear all
clc

% Load model
load('models/weights_res_con.mat')

%% GLipSDP

NN = init_NN;

NN.layers = {'conv','res_conv','res_conv','res_conv','res_conv','res_conv',...
    'res_conv','res_conv','res_conv','fc'};

NN.weights = W;
NN.pool = {'max','none','none','none','none','none','none','none','av'};
NN.strides      = [2,1,1,1,1,1,1,1,1];
NN.pool_strides = [2,0,0,0,0,0,0,0,2];
NN.pool_kernel  = [3,0,0,0,0,0,0,0,2];
NN.cond = 0.5;

[Lip_GLipSDP,info_GLipSDP,time_GLipSDP] = LipEst(NN);

%% S-GLipSDP

NNfirst = init_NN;
NNfirst.layers = {'conv'};
NNfirst.weights{1} = W{1};
NNfirst.pool = {'max'};
NNfirst.strides      = [2];
NNfirst.pool_strides = [2];
NNfirst.pool_kernel  = [3];
NNfirst.cond = 1;

[Lip1,info1,time1] = LipEst(NNfirst);

Lip_S_GLipSDP = Lip1;

NNres = init_NN;
NNres.layers = {'res_conv'};
for ii = 2:9
    NNres.weights{1} = W{ii};
    NNres.strides      = [1];
    if ii == 9
        NNres.pool = {'av'};
        NNres.pool_strides = [2];
        NNres.pool_kernel  = [2];
    else
        NNres.pool = {'none'};
        NNres.pool_strides = [0];
        NNres.pool_kernel  = [0];
    end
    [Lip2(ii-1),info2{ii-1},time2(ii-1)] = LipEst(NNres);
    Lip_S_GLipSDP = Lip_S_GLipSDP*Lip2(ii-1);
end

NNlast = init_NN;
NNlast.layers = {'fc'};
NNlast.weights{1} = W{end};
[Lip3,info3,time3] = LipEst(NNlast);

Lip_S_GLipSDP = Lip_S_GLipSDP*Lip3;
time_S_GLipSDP = time1+sum(time2)+time3;

Lip_S_GLipSDP

%% MP
W3{1} = conv2fc(W{1},28,28,2,1);
Lip_MP = norm(W3{1})*4; %4 is the Lipschitz constant of the strided max pooling layer

for ii = 2:9
   Lip_MP_s(ii-1) = norm(conv2fc(W{ii}{1},14,14,1,1))*norm(conv2fc(W{ii}{2},14,14,1,1)) + 1;
   Lip_MP = Lip_MP * Lip_MP_s(ii-1);
end

Lip_MP = Lip_MP*norm(W{10})*0.25; %0.25 is the Lipschitz constant of the average pooling layer

%% Save results
save('results/mnist_ResNet_conv.mat')