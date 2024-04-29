close all
clear all
clc

rng(1)
cin = 2;
stride = 2;
conv1 = randn(2,cin,3,3);
M = getRoesser(conv1,stride);
image = randn(1,cin,5,5); % first dimension = batches

%%
W = conv2fc(conv1,5,5,stride,0);

uflat = flatten(image);

y_flat = W*uflat;

%%
%image_tmp = zeros(1,cin,7,7);
%image_tmp(:,:,2:6,2:6) = image;
%image = image_tmp;

y = simulate2DSysRoesser(M,image,stride);

image = reshape(image,[],5,5);

y_Filter = conv2D_multi(conv1,image,stride);

error = sum(sum(sum(y-y_Filter).^2));

