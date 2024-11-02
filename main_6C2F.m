close all
clear all
clc

load('models/weights_cifar_6C2F_wd_con.mat')

savepath = 'results/cifar_6C2F_wd.mat';

%% S-GLipSDP

NN1 = init_NN;

NN1.layers = {'conv','conv','conv'};

NN1.weights = W(1:3);
NN1.pool = {'none','none','none'};
NN1.strides      = [1,1,2];
NN1.pool_strides = [0,0,0];
NN1.pool_kernel  = [0,0,0];

[Lip1,info1,time1] = LipEst(NN1);

NN2 = init_NN;

NN2.weights = W(4:5);
NN2.layers = {'conv','conv'};
NN2.pool = {'none','none'};
NN2.strides      = [1,1];
NN2.pool_strides = [0,0];
NN2.pool_kernel  = [0,0];

[Lip2,info2,time2] = LipEst(NN2);

W_fc{6} = conv2fc(W{6},16,16,2,0);

tic
Lip3 = norm(W_fc{6});
time3 = toc

NN4 = init_NN;

NN4.weights = W(7:8);
NN4.layers = {'fc','fc'};
[Lip4,info4,time4] = LipEst(NN4);

%Lip4 = norm(W_fc{6});
%Lip5 = norm(W{7});
%Lip6 = norm(W{8});

Lip_S_GLipSDP = Lip1*Lip2*Lip3*Lip4;;
time_S_GLipSDP = time1+time2+time3+time4

Lip_S_GLipSDP

save(savepath)

%% S-LipSDP

W_fc{1} = conv2fc(W{1},30,30,1,1);
W_fc{2} = conv2fc(W{2},30,30,1,1);
W_fc{3} = conv2fc(W{3},30,30,2,1);
W_fc{4} = conv2fc(W{1},16,16,1,1);
W_fc{5} = conv2fc(W{2},16,16,1,1);
%W_fc{6} = conv2fc(W{4},16,16,2,0);

[Lip1_LipSDP,info1_LipSDP,time1_LipSDP] = LipschitzEstimationFazlyab(W_c2fc(1:2));
[Lip2_LipSDP,info2_LipSDP,time2_LipSDP] = LipschitzEstimationFazlyab(W_c2fc(3:4));
[Lip3_LipSDP,info3_LipSDP,time3_LipSDP] = LipschitzEstimationFazlyab(W_c2fc(5:6));
[Lip4_LipSDP,info4_LipSDP,time4_LipSDP] = LipschitzEstimationFazlyab(W(7:8));

Lip_S_LipSDP = Lip1_LipSDP*Lip2_LipSDP*Lip3_LipSDP*Lip4_LipSDP;
time_S_LipSDP = time1_LipSDP+time2_LipSDP+time3_LipSDP+time4_LipSDP;

Lip_S_LipSDP

save(savepath)

%% MP

for ii = 1:length(W_fc)
    Lip_MP_s(ii) = norm(W_c2fc{ii});
end
for ii = length(W_fc)+1:length(W)
    Lip_MP_s(ii) = norm(W{ii});
end

Lip_MP = 1;
for ii = 1:length(W)
    Lip_MP = Lip_MP*Lip_MP_s(ii);
end

Lip_MP

%% Save results
save(savepath)
