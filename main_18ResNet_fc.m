close all
clear all
clc

load('models/weights_mnist_res_fc_wd_con.mat')

savepath = 'results/mnist_ResNet_fc_wd.mat';

%% GLipSDP

NN = init_NN;
NN.layers = {'fc','res_fc2','res_fc2','res_fc2','res_fc2','res_fc2',...
    'res_fc2','res_fc2','res_fc2','fc'};

NN.weights = W;

[Lip_GLipSDP,info_GLipSDP,time_GLipSDP] = LipEst(NN);

Lip_GLipSDP

%% S-GLipSDP

NNfirst = init_NN;
NNfirst.layers = {'fc'};
NNfirst.weights{1} = W{1};
[Lip1,info1,time1] = LipEst(NNfirst);
Lip_S_GLipSDP = Lip1;

NN2 = init_NN;
NN2.layers = {'res_fc2'};
for ii = 2:9
    NN2.weights{1} = W{ii};
    [Lip2(ii-1),info2{ii-1},time2(ii-1)] = LipEst(NN2);
    Lip_S_GLipSDP = Lip_S_GLipSDP*Lip2(ii-1);
end

NNlast = init_NN;
NNlast.layers = {'fc'};
NNlast.weights{1} = W{end};
[Lip3,info3,time3] = LipEst(NNlast);
Lip_S_GLipSDP = Lip_S_GLipSDP*Lip3;
time_S_GLipSDP = time1+sum(time2)+time3;

Lip_S_GLipSDP

%% S-LipSDP

tic
Lip_S_LipSDP = norm(W{1});
time1 = toc; 

for ii = 2:9
    [Lip_LipSDP(ii-1),info_LipSDP{ii-1},time_LipSDP(ii-1)] = LipschitzEstimationFazlyab(W{ii});
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