close all
clear all
clc

% Load model
load('models/weights_mnist_res_wd_7_con.mat')

savepath = 'results/mnist_ResNet_conv_wd_7.mat';

W{1}=0.1*W{1}; % This trick changes the conditioning of the problem
W{10}=0.1*W{10}; % It is less likely to run into numerical problems
% We multiply by this factor after solving the SDP

%% GLipSDP

NN = init_NN;

NN.layers = {'conv','res_conv2','res_conv2','res_conv2','res_conv2','res_conv2',...
    'res_conv2','res_conv2','res_conv2','fc'};

NN.weights = W;
NN.pool = {'max','none','none','none','none','none','none','none','av'};
NN.strides      = [1,1,1,1,1,1,1,1,1];
NN.pool_strides = [2,0,0,0,0,0,0,0,2];
NN.pool_kernel  = [2,0,0,0,0,0,0,0,2];

[Lip_GLipSDP,info_GLipSDP,time_GLipSDP] = LipEst(NN);

Lip_GLipSDP = Lip_GLipSDP/0.1^2

save(savepath)

%% S-GLipSDP

NNfirst = init_NN;
NNfirst.layers = {'conv'};
NNfirst.weights{1} = W{1};
NNfirst.pool = {'max'};
NNfirst.strides      = [1];
NNfirst.pool_strides = [2];
NNfirst.pool_kernel  = [2];

[Lip1,info1,time1] = LipEst(NNfirst);

Lip_S_GLipSDP = Lip1;

NNres = init_NN;
NNres.layers = {'res_conv2'};
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

Lip_S_GLipSDP = Lip_S_GLipSDP/0.1^2

save(savepath)

%% S-LipSDP
W3{1} = conv2fc(W{1},28+6,28+6,1,0);

tic
Lip_S_LipSDP = norm(W3{1});
time1 = toc; 

for ii = 2:9
    W_fc{1} = conv2fc(W{ii}{1},14+2,14+2,1,1);
    W_fc{2} = conv2fc(W{ii}{2},14+2,14+2,1,1);
    [Lip_LipSDP(ii-1),info_LipSDP{ii-1},time_LipSDP(ii-1)] = LipschitzEstimationFazlyab(W_fc);
end

for ii = 1:8
    Lip_S_LipSDP = Lip_S_LipSDP*(Lip_LipSDP(ii)+1);
end

tic
Lip_S_LipSDP = Lip_S_LipSDP*norm(W{10})*0.5 %0.5 is the Lipschitz constant of the average pooling layer
time2 = toc;
time_S_LipSDP = time1 + time2 + sum(time_LipSDP);

Lip_S_LipSDP = Lip_S_LipSDP/0.1^2

save(savepath)

%% MP

Lip_MP = norm(W3{1})/0.1^2;

for ii = 2:9
   Lip_MP_s(ii-1) = norm(conv2fc(W{ii}{1},14+2,14+2,1,1))*norm(conv2fc(W{ii}{2},14+2,14+2,1,1)) + 1;
   Lip_MP = Lip_MP * Lip_MP_s(ii-1);
end

Lip_MP = Lip_MP*norm(W{10})*0.5 %0.5 is the Lipschitz constant of the average pooling layer

% Save results
save(savepath)