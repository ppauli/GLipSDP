close all
clear all
clc

NN = init_NN;

NN.layers = {'conv','conv'};

NN.weights{1} = randn(4,1,3,3);
NN.weights{2} = randn(4,4,3,3);
NN.pool = {'av','av'};
NN.strides      = [1,1];
NN.pool_strides = [2,2];
NN.pool_kernel  = [2,2];

[Lip,info,time] = LipEst(NN);

NN2 = NN;

NN2.pool = {'none','none'};
NN2.strides      = [1,1];
NN2.pool_strides = [0,0];
NN2.pool_kernel  = [0,0];

[Lip2,info2,time2] = LipEst(NN2);
