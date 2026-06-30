close all
clear all
clc

% This is a sanity check for the matrix conversion function

c_in = 1;
c_out = 1;

input_1 = 3;
input_2 = 3;

kernel_1 = 3;
kernel_2 = 3;

kernel = randn(c_out, c_in, kernel_1, kernel_2);

inputSize = [c_in, input_1, input_2];
stride = 2;
padding = 1;

[Wfc, outSize] = conv_layer_to_fc(kernel, inputSize, stride, padding);

x = randn(inputSize);

y_direct = conv_forward_direct(kernel, x, stride, padding);
y_fc = reshape(Wfc * x(:), outSize);

err_fc = norm(y_direct(:) - y_fc(:));

fprintf('Matrix conversion error: %.3e\n', err_fc);

%% Roesser sanity check

M_roesser = getRoesser(kernel, stride);
y_roesser = conv_forward_roesser(M_roesser, x, padding, outSize, [kernel_1, kernel_2], stride);

err_roesser = norm(y_direct(:) - y_roesser(:));

fprintf('Roesser error, kernel directly: %.3e\n', err_roesser);