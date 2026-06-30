function y = conv_forward_roesser(M, x, padding, outSize, kernelSize, stride)
%CONV_FORWARD_ROESSER Evaluate CNN-style strided convolution using Roesser model.
%
% Assumes channel-first tensors:
%   x : cin x height x width
%   y : cout x out_height x out_width
%
% For CNN-style cross-correlation, construct M as
%
%   M = getRoesser(kernel(:, :, end:-1:1, end:-1:1), stride);
%
% because getRoesser flips the kernel internally.

    [cin, height, width] = size(x);

    k1 = kernelSize(1);
    k2 = kernelSize(2);
    s  = stride;

    cout = size(M.D, 1);

    out_height = outSize(2);
    out_width  = outSize(3);

    if s < 1 || floor(s) ~= s
        error('stride must be a positive integer.');
    end

    if s > k1 || s > k2
        error('This getRoesser construction assumes stride <= kernel dimensions.');
    end

    expected_u_dim = cin * s^2;

    if size(M.D, 2) ~= expected_u_dim
        error('M.D has %d columns, but expected cin * stride^2 = %d.', ...
              size(M.D, 2), expected_u_dim);
    end

    if size(M.B, 2) ~= expected_u_dim
        error('M.B has %d columns, but expected cin * stride^2 = %d.', ...
              size(M.B, 2), expected_u_dim);
    end

    % ------------------------------------------------------------
    % 1. Zero-pad input.
    % ------------------------------------------------------------
    x_pad = zeros(cin, height + 2 * padding, width + 2 * padding);

    x_pad(:, padding + 1 : padding + height, ...
             padding + 1 : padding + width) = x;

    [~, height_pad, width_pad] = size(x_pad);

    % ------------------------------------------------------------
    % 2. Correct phase alignment.
    %
    % The Roesser system is causal and produces an output when the
    % bottom-right corner of the convolution window has arrived.
    %
    % For a strided system, we need phase + kernel_size to be divisible
    % by stride.
    % ------------------------------------------------------------
    row_phase = mod(s - mod(k1, s), s);
    col_phase = mod(s - mod(k2, s), s);

    row_start = (row_phase + k1) / s;
    col_start = (col_phase + k2) / s;

    if abs(row_start - round(row_start)) > 1e-12 || ...
       abs(col_start - round(col_start)) > 1e-12
        error('Internal phase error: crop indices are not integers.');
    end

    row_start = round(row_start);
    col_start = round(col_start);

    % Number of coarse Roesser steps needed.
    n_block_rows = row_start + out_height - 1;
    n_block_cols = col_start + out_width  - 1;

    % Extend input so every stride-by-stride block exists.
    height_ext = max(row_phase + height_pad, n_block_rows * s);
    width_ext  = max(col_phase + width_pad,  n_block_cols * s);

    x_ext = zeros(cin, height_ext, width_ext);

    x_ext(:, row_phase + 1 : row_phase + height_pad, ...
             col_phase + 1 : col_phase + width_pad) = x_pad;

    % ------------------------------------------------------------
    % 3. Roesser state dimensions.
    % ------------------------------------------------------------
    n1 = size(M.A11, 1);
    n2 = size(M.A22, 1);

    X1 = zeros(n1, n_block_cols);
    y_full = zeros(cout, n_block_rows, n_block_cols);

    % ------------------------------------------------------------
    % 4. Coarse-grid Roesser scan.
    % ------------------------------------------------------------
    for ii = 1:n_block_rows

        x2_state = zeros(n2, 1);
        X1_next = zeros(n1, n_block_cols);

        for jj = 1:n_block_cols

            x1_state = X1(:, jj);

            row_idx = (ii - 1) * s + 1 : ii * s;
            col_idx = (jj - 1) * s + 1 : jj * s;

            block = x_ext(:, row_idx, col_idx);

            % Important:
            % Do NOT use u = block(:), because MATLAB's default linear
            % indexing gives a different order.
            %
            % The ordering here matches getRoesser:
            %
            %   [ block(:,1,1);
            %     block(:,1,2);
            %     ...
            %     block(:,1,s);
            %     block(:,2,1);
            %     ...
            %     block(:,s,s) ]
            %
            u = zeros(cin * s^2, 1);
            idx = 1;

            for aa = 1:s
                for bb = 1:s
                    u(idx : idx + cin - 1) = block(:, aa, bb);
                    idx = idx + cin;
                end
            end

            % Output equation.
            y_full(:, ii, jj) = M.C1 * x1_state + ...
                                M.C2 * x2_state + ...
                                M.D  * u;

            % State update.
            x1_next = M.A11 * x1_state + ...
                      M.A12 * x2_state + ...
                      M.B1  * u;

            x2_next = M.A21 * x1_state + ...
                      M.A22 * x2_state + ...
                      M.B2  * u;

            if n1 > 0
                X1_next(:, jj) = x1_next;
            end

            x2_state = x2_next;
        end

        X1 = X1_next;
    end

    % ------------------------------------------------------------
    % 5. Crop causal delay.
    % ------------------------------------------------------------
    y = y_full(:, row_start : row_start + out_height - 1, ...
                  col_start : col_start + out_width  - 1);
end