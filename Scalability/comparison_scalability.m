% This code compares LipSDP and LipSDP-conv

close all
clear all
clc

l1 = 3; % kernel size
l2 = l1; % kernel size
N1 = 14; % input image size
N2 = 14; % input image size

c_vec = [8,16,32]; % channel sizes
depth = [2,4,8,16];

for jj = 1:length(c_vec)
    jj
    for kk = 1:length(depth)
        kk
        clear K W
        load(['weights/weights_l',num2str(depth(kk)),'_c',num2str(c_vec(jj)),'.mat'])
        K = convert_weights(weights,depth(kk));
        
        %% GLipSDP
        NN = init_NN;
        NN.layers = {};
        for ii = 1:depth(kk)
            NN.layers{ii} = 'conv';
            NN.pool{ii} = 'none';
            NN.strides(ii)      = 1;
            NN.pool_strides(ii) = 0;
            NN.pool_kernel(ii)  = 0;
        end
        NN.weights = K;

        [L_SDP, info_SDP, time_SDP] = LipEst(NN);
        L_SDP_mat(jj,kk) = L_SDP
        info_SDP_mat{jj,kk} = info_SDP
        time_SDP_mat(jj,kk) = time_SDP
        
        %% CLipSDP
        [L_SDP_sparse,info_SDP_sparse,time_SDP_sparse] = solve_SDP_sparse(K);
        L_SDP_sparse_mat(jj,kk) = L_SDP_sparse
        info_SDP_sparse_mat{jj,kk} = info_SDP_sparse
        time_SDP_sparse_mat(jj,kk) = time_SDP_sparse
       
        L_triv = 1;
        for ii = 1:depth(kk)
            W{ii} = conv2fc(K{ii},N1+2,N2+2,1,1);
            L_triv = L_triv * norm(W{ii});
        end
        L_triv_mat(jj,kk) = L_triv

        %% LipLT
        [L_LipLT, time_LipLT] = LipLT(W);
        L_LipLT_mat(jj,kk) = L_LipLT
        time_LipLT_mat(jj,kk) = time_LipLT

        %if c_vec(jj) <= 8 && depth(kk) <= 2
        %    [L_LipSDP, info_LipSDP, time_LipSDP] = LipschitzEstimationFazlyab(W);
        %    L_LipSDP_mat(jj,kk) = L_LipSDP
        %    info_LipSDP_mat{jj,kk} = info_LipSDP
        %    time_LipSDP_mat(jj,kk) = time_LipSDP
        %end
    end
end

save("../results/results_all.mat")