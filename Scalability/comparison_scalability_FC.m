% This code compares LipSDP, LipLT and GLipSDP

close all
clear all
clc

c_vec = [16,32,64]; % channel sizes
depth = [2,4,8,16,32,64];

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
        NN.weights = W;
        [L_GLipSDP,info_GlipSDP,time_GLipSDP] = LipEst(NN);
        L_GLipSDP_mat(jj,kk) = L_GLipSDP
        info_GLipSDP_mat{jj,kk} = info_GlipSDP
        time_GLipSDP_mat(jj,kk) = time_GLipSDP

        %% GLipSDP Subnetworks
        NN = init_NN;
        NN.layers{1} = 'subn_fc';
        NN.weights{1} = W;
        
        [L_GLipSDP_2,info_GlipSDP_2,time_GLipSDP_2] = LipEst(NN);
        L_GLipSDP_2_mat(jj,kk) = L_GLipSDP_2
        info_GLipSDP_2_mat{jj,kk} = info_GlipSDP_2
        time_GLipSDP_2_mat(jj,kk) = time_GLipSDP_2
        
        %% MP
        L_triv = 1;
        for ii = 1:depth(kk)
            L_triv = L_triv * norm(W{ii});
        end
        L_triv_mat(jj,kk) = L_triv
        %% LipLT
        [L_LipLT, time_LipLT] = LipLT(W);
        L_LipLT_mat(jj,kk) = L_LipLT
        time_LipLT_mat(jj,kk) = time_LipLT;
        %% LipSDP
        [L_LipSDP, info_LipSDP, time_LipSDP] = LipschitzEstimationFazlyab(W);
        L_LipSDP_mat(jj,kk) = L_LipSDP
        info_LipSDP_mat{jj,kk} = info_LipSDP
        time_LipSDP_mat(jj,kk) = time_LipSDP;

    end
end

save("../results/results_all_FC.mat")
