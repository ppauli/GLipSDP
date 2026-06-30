function lip_values = evaluate_single_model_lipschitz(weight_file, model_name)

% Load one trained model instance.
weights = load(weight_file);

addpath('/Applications/mosek/11.0/toolbox/r2022bom')

load(weight_file)

switch model_name

    case "fnn2"

        W{1} = double(fc1_weight');
        W_all{2} = double(fc2_weight');
        b{1} = double(fc1_bias');
        b_all{2} = double(fc2_bias');

        W{2} = W_all{2}(8,:); % output for entry 8 selected
        b{2} = b_all{2}(8); % output for entry 8 selected

        [n2,n1] = size(W{1});
        [n3,~] = size(W{2});

        %%
        [Lip_MP, time_MP] = naive_lip_mlp(W)

        %%
        lb = zeros(n1,1);
        ub = ones(n1,1);

        numSamples = 100000;

        [Lip_lb, bestX, bestPattern, info] = relu_lipschitz_l2_lower_bound_sampling(W, b, lb, ub, numSamples)

        %%

        %xCenter = zeros(n1,1);
        %radius = 10;
        %relaxOrder = 2;

        %[L_ub, diagnostics, aux] = relu_lcep_l2_moment_1hidden(W, b, xCenter, radius, relaxOrder)

        %%

        %[L_FGL, bestPattern, info] = relu_fgl_l2(W)

        %%

        opts = struct();
        opts.bruteMaxDim = 20;
        opts.verbose = true;

        [Lip_SepLip, time_SepLip, info_SeqLip] = seqlip(W, opts)

        %%

        alpha = 0;
        beta = 1;

        opts = struct();
        opts.verbose = true;

        [Lip_LT, time_LT, info_LT] = liplt_mlp(W, alpha, beta, opts)

        %%
        [Lip_LipSDP, info_LipSDP , time_LipSDP] = LipSDP(W)

        %%
        NN = init_NN;
        NN.layers = {'fc','fc'};
        NN.weights = W;
        NN.Alpha = {0,0};
        NN.Beta = {1,1};

        [Lip_GLipSDP, info_GLipSDP, time_GLipSDP] = GLipSDP(NN)

        %%

        [Lip_LiPopt, time_LiPopt] = lipopt_l2_shor(W);

    case "fnn3"

        W{1} = double(fc1_weight');
        W{2} = double(fc2_weight');
        W_all{3} = double(fc3_weight');
        b{1} = double(fc1_bias');
        b{2} = double(fc2_bias');
        b_all{3} = double(fc3_bias');

        W{3} = W_all{3}(8,:); % output for entry 8 selected
        b{3} = b_all{3}(8); % output for entry 8 selected

        [n2,n1] = size(W{1});
        [n4,n3] = size(W{3});


        %%
        [Lip_MP, time_MP] = naive_lip_mlp(W)

        %%
        lb = zeros(n1,1);
        ub = ones(n1,1);

        numSamples = 100000;

        [Lip_lb, bestX, bestPattern, info] = relu_lipschitz_l2_lower_bound_sampling(W, b, lb, ub, numSamples)

        %%

        opts = struct();
        opts.bruteMaxDim = 20;
        opts.verbose = true;

        [Lip_SepLip, time_SepLip, info_SeqLip] = seqlip(W, opts)

        %%

        alpha = 0;
        beta = 1;

        opts = struct();
        opts.verbose = true;

        [Lip_LT, time_LT, info_LT] = liplt_mlp(W, alpha, beta, opts)

        %%
        [Lip_LipSDP, info_LipSDP , time_LipSDP] = LipSDP(W)

        %%
        NN = init_NN;
        NN.layers = {'fc','fc','fc'};
        NN.weights = W;
        NN.Alpha = {0,0,0};
        NN.Beta = {1,1,1};

        [Lip_GLipSDP, info_GLipSDP, time_GLipSDP] = GLipSDP(NN)

        %%

        [Lip_LiPopt, time_LiPopt] = lipopt_l2_shor(W);


    case "cnn2"

        input_1 = 8;
        input_2 = 8;

        K = permute(double(conv1_weight),[4,3,1,2]);
        W_all{2} = double(fc1_weight');
        b_K{1} = double(conv1_bias');
        b_all{2} = double(fc1_bias');

        W{2} = W_all{2}(8,:); % output for entry 8 selected
        b{2} = b_all{2}(8); % output for entry 8 selected

        [c2, c1, k1, k2] = size(K);
        [n3, n2] = size(W{2});

        stride = 1;
        padding = 1;


        inputSize = [c1, input_1, input_2];

        [W_sparse, outSize] = conv_layer_to_fc(K, inputSize, stride, padding);
        W{1} = full(W_sparse);
        b{1} = conv_bias_to_fc(b_K{1}, outSize);

        [n2,n1] = size(W{1});

        if size(W{1},1) ~= size(W{2},2)
            error('Dimension mismatch: size(W{1},1) must equal size(W{2},2). Got %d and %d.', ...
                size(W{1},1), size(W{2},2));
        end

        %%
        [Lip_MP, time_MP] = naive_lip_mlp(W)

        %%
        lb = zeros(n1,1);
        ub = ones(n1,1);

        numSamples = 100000;

        [Lip_lb, bestX, bestPattern, info] = relu_lipschitz_l2_lower_bound_sampling(W, b, lb, ub, numSamples)

        %%
        opts = struct();
        opts.bruteMaxDim = 20;
        opts.verbose = true;

        [Lip_SepLip, time_SepLip, info_SeqLip] = seqlip(W, opts)

        %%

        alpha = 0;
        beta = 1;

        opts = struct();
        opts.verbose = true;

        [Lip_LT, time_LT, info_LT] = liplt_mlp(W, alpha, beta, opts)

        %%
        [Lip_LipSDP, info_LipSDP, time_LipSDP] = LipSDP(W)

        %%
        %NN = init_NN;
        %NN.layers = {'fc','fc'};
        %NN.weights = W;
        %NN.Alpha = {0,0};
        %NN.Beta = {1,1};

        %[Lip_GLipSDP, info_GLipSDP, time_GLipSDP] = GLipSDP(NN)

        %%
        NN = init_NN;
        NN.layers = {'conv','fc'};
        NN.weights = {K,W{2}};
        NN.Alpha = {0,0};
        NN.Beta = {1,1};
        NN.pool = {'none','none'};
        NN.strides      = stride;
        NN.pool_strides = [0];
        NN.pool_kernel  = [0];
        NN.glob_loc = 'global';
        NN.padding = padding;

        [Lip_GLipSDP_c, info_GLipSDP_c, time_GLipSDP_c] = GLipSDP(NN)

        %%

        [Lip_LiPopt, time_LiPopt] = lipopt_l2_shor(W);

    case "cnn3"

        input_1 = 8;
        input_2 = 8;

        K{1} = double(conv1_weight);
        K{2} = double(conv2_weight);
        W_all{3} = double(fc1_weight');
        b_K{1} = double(conv1_bias');
        b_K{2} = double(conv2_bias');
        b_all{3} = double(fc1_bias');

        W{3} = W_all{3}(8,:); % output for entry 8 selected
        b{3} = b_all{3}(8); % output for entry 8 selected

        [c2(1), c1(1), k1, k2] = size(K{1});

        stride = [1,1];
        padding = [1,1];

        inputSize = [c1(1), input_1, input_2];

        [W_sparse{1}, outSize{1}] = conv_layer_to_fc(K{1}, inputSize, stride(1), padding(1));
        W{1} = full(W_sparse{1});
        b{1} = conv_bias_to_fc(b_K{1}, outSize{1});

        [W_sparse{2}, outSize{2}] = conv_layer_to_fc(K{2}, outSize{1}, stride(2), padding(2));
        W{2} = full(W_sparse{2});
        b{2} = conv_bias_to_fc(b_K{2}, outSize{2});

        [n2,n1] = size(W{1});
        [n4, n3] = size(W{3});

        if size(W{2},1) ~= size(W{3},2)
            error('Dimension mismatch: size(W{1},1) must equal size(W{2},2). Got %d and %d.', ...
                size(W{2},1), size(W{3},2));
        end

        %%
        [Lip_MP, time_MP] = naive_lip_mlp(W)

        %%
        lb = zeros(n1,1);
        ub = ones(n1,1);

        numSamples = 10000;

        [Lip_lb, bestX, bestPattern, info] = relu_lipschitz_l2_lower_bound_sampling(W, b, lb, ub, numSamples)

        %%

        opts = struct();
        opts.bruteMaxDim = 20;
        opts.verbose = true;

        [Lip_SepLip, time_SepLip, info_SeqLip] = seqlip(W, opts)

        %%

        alpha = 0;
        beta = 1;

        opts = struct();
        opts.verbose = true;

        [Lip_LT, time_LT, info_LT] = liplt_mlp(W, alpha, beta, opts)

        %%
        [Lip_LipSDP, info_LipSDP, time_LipSDP] = LipSDP(W)

        %%
        % NN = init_NN;
        % NN.layers = {'fc','fc','fc'};
        % NN.weights = W;
        % NN.Alpha = {0,0,0};
        % NN.Beta = {1,1,1};
        %
        % [Lip_GLipSDP, info_GLipSDP, time_GLipSDP] = GLipSDP(NN)

        %%
        NN = init_NN;
        NN.layers = {'conv','conv','fc'};
        NN.weights = {permute(K{1},[4, 3, 1, 2]),permute(K{2},[4, 3, 1, 2]),W{3}};
        NN.Alpha = {0,0,0};
        NN.Beta = {1,1,1};
        NN.pool = {'none','none'};
        NN.strides      = stride;
        NN.pool_strides = [0,0];
        NN.pool_kernel  = [0,0];
        NN.glob_loc = 'global';
        NN.padding = padding;

        [Lip_GLipSDP_c, info_GLipSDP_c, time_GLipSDP_c] = GLipSDP(NN)

        %%

        %[Lip_LiPopt, time_LiPopt] = lipopt_l2_shor(W);

    otherwise

        error("Unknown model name: %s", model_name);
end

% -------------------------------------------------------------
% Return every variable starting with "Lip"
% -------------------------------------------------------------
vars = whos("Lip*");

lip_values = struct();

for i = 1:numel(vars)
    var_name = vars(i).name;
    lip_values.(var_name) = eval(var_name);
end
end