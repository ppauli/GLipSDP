% This code compares LipSDP, LipLT and GLipSDP

close all
clear all
clc

c_vec = [8,16,32,64,128]; % channel sizes
depth = [4,8,16,32];

for jj = 1:length(c_vec)
    jj
    for kk = 1:length(depth)
        kk
        clear K W
        NN = init_NN;
        NN.layers = {};
        for ii = 1:depth(kk)
            W{ii} = randn(c_vec(jj),c_vec(jj));
            W{ii} = 1/norm(W{ii})*W{ii};
            NN.layers{ii} = 'fc';
        end
        NN.weights = W;
        
        %% GLipSDP
        [L_SDP, info_SDP, time_SDP] = LipEst(NN);
        L_SDP_mat(jj,kk) = L_SDP
        info_SDP_mat{jj,kk} = info_SDP
        time_SDP_mat(jj,kk) = time_SDP
        
        %% MP
        L_triv = 1;
        for ii = 1:depth(kk)
            L_triv = L_triv * norm(W{ii});
        end
        L_triv_mat(jj,kk) = L_triv
        %% LipSDP
        [L_LipSDP, info_LipSDP, time_LipSDP] = LipschitzEstimationFazlyab(W);
        L_LipSDP_mat(jj,kk) = L_LipSDP
        info_LipSDP_mat{jj,kk} = info_LipSDP
        time_LipSDP_mat(jj,kk) = time_LipSDP;
        %% LipLT
        [L_LipLT, time_LipLT] = LipLT(W);
        L_LipLT_mat(jj,kk) = L_LipLT
        time_LipLT_mat(jj,kk) = time_LipLT;
    end
end

save("../results/results_all_FC.mat")
