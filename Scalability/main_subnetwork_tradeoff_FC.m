close all
clear all
clc

%% Subnetwork tradeoff

load('exported_mnist_models/mlp_L33_W16.mat')

%% Load trained weights
K = cell(1, 32);

for ii = 1:32
    weight_name = sprintf('linear_%d_weight', ii);
    W{ii} = double(params.(weight_name)');
end

%% Layer-by-layer

NN = init_NN;
NN.weights = W;
NN.Alpha = cell(32,1);
NN.Beta = cell(32,1);

for ii = 1:32
    NN.layers{ii} = 'fc';
end

[Lip,info{1},time(1)] = GLipSDP(NN);

%% 2-layer subnetworks

NN2 = init_NN;
NN2.Alpha = cell(32,1);
NN2.Beta = cell(32,1);

for ii = 1:16
    NN2.layers{ii} = 'subn_fc';
    NN2.weights{ii} = W(2*ii-1:2*ii);
end

[Lip2,info{2},time(2)] = GLipSDP(NN2);

%% 4-layer subnetworks

NN3 = init_NN;
NN3.Alpha = cell(32,1);
NN3.Beta = cell(32,1);

for ii = 1:8
    NN3.layers{ii} = 'subn_fc';
    NN3.weights{ii} = W(4*ii-3:4*ii);
end

[Lip3,info{3},time(3)] = GLipSDP(NN3);

%% 8-layer subnetworks

NN4 = init_NN;
NN4.Alpha = cell(32,1);
NN4.Beta = cell(32,1);

for ii = 1:4
    NN4.layers{ii} = 'subn_fc';
    NN4.weights{ii} = W(8*ii-7:8*ii);
end

[Lip4,info{4},time(4)] = GLipSDP(NN4);

%% 16-layer subnetworks

NN5 = init_NN;
NN5.Alpha = cell(32,1);
NN5.Beta = cell(32,1);

for ii = 1:2
    NN5.layers{ii} = 'subn_fc';
    NN5.weights{ii} = W(16*ii-15:16*ii);
end

[Lip5,info{5},time(5)] = GLipSDP(NN5);

%% 32-layer subnetwork

NN6 = init_NN;
NN6.Alpha = cell(32,1);
NN6.Beta = cell(32,1);

NN6.layers{1} = 'subn_fc';
NN6.weights{1} = W;

[Lip6,info{6},time(6)] = GLipSDP(NN6);

%% Save results
save('results/subnetwork_tradeoff_FC.mat')
