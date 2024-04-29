close all
clear all
clc

%% Subnetwork tradeoff

load('models/weights_l16_c8.mat')

K = convert_weights(weights,16);



%% Layer-by-layer


NN = init_NN;
NN.weights = K;

for ii = 1:16
    NN.layers{ii} = 'conv';
    NN.pool{ii} = 'none';
    NN.strides(ii) = 1;
    NN.pool_strides(ii) = 0;
    NN.pool_kernel(ii)  = 0;
end

[Lip,info{1},time(1)] = LipEst(NN);

%% 2-layer subnetworks

NN2 = init_NN;

for ii = 1:8
    NN2.layers{ii} = 'subn_conv';
    NN2.weights{ii} = K(2*ii-1:2*ii);
    NN2.pool{ii} = 'none';
    NN2.strides(ii) = 1;
    NN2.pool_strides(ii) = 0;
    NN2.pool_kernel(ii)  = 0;
end

[Lip2,info{2},time(2)] = LipEst(NN2);

%% 4-layer subnetworks

NN3 = init_NN;

for ii = 1:4
    NN3.layers{ii} = 'subn_conv';
    NN3.weights{ii} = K(4*ii-3:4*ii);
    NN3.pool{ii} = 'none';
    NN3.strides(ii) = 1;
    NN3.pool_strides(ii) = 0;
    NN3.pool_kernel(ii)  = 0;
end

[Lip3,info{3},time(3)] = LipEst(NN3);

%% 8-layer subnetworks

NN4 = init_NN;

for ii = 1:2
    NN4.layers{ii} = 'subn_conv';
    NN4.weights{ii} = K(8*ii-7:8*ii);
    NN4.pool{ii} = 'none';
    NN4.strides(ii) = 1;
    NN4.pool_strides(ii) = 0;
    NN4.pool_kernel(ii)  = 0;
end

[Lip4,info{4},time(4)] = LipEst(NN4);

%% 16-layer subnetwork

NN5 = init_NN;

NN5.layers{1} = 'subn_conv';
NN5.weights{1} = K;
NN5.pool{1} = 'none';
NN5.strides = 1;
NN5.pool_strides = 0;
NN5.pool_kernel  = 0;

[Lip5,info{5},time(5)] = LipEst(NN5);

%% Save results
save('results/subnetwork_tradeoff_8.mat')
