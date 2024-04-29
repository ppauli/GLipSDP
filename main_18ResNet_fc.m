close all
clear all
clc

load('models/weights_res_fc_con')

%% GLipSDP

NN = init_NN;

NN.layers = {'fc','res_fc','res_fc','res_fc','res_fc','res_fc',...
    'res_fc','res_fc','res_fc','fc'};

NN.weights = W;
NN.cond = 0.5;

[Lip_GLipSDP,info_GLipSDP,time_GLipSDP] = LipEst(NN);

%% S-GLipSDP

NNfirst = init_NN;
NNfirst.layers = {'fc'};
NNfirst.weights{1} = W{1};
[Lip1,info1,time1] = LipEst(NNfirst);
Lip_S_GLipSDP = Lip1;

NN2 = init_NN;
NN2.layers = {'res_fc'};
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

%% MP

Lip_MP = norm(W{1});

for ii = 2:9
    Lip_MP_s(ii-1) = 1+norm(W{ii}{1})*norm(W{ii}{2});
    Lip_MP = Lip_MP*Lip_MP_s(ii-1);
end

Lip_MP = Lip_MP*norm(W{end})

%% Save results
save('results/mnist_ResNet_fc.mat')