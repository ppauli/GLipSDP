close all
clear all
clc

load('example_image.mat')

load('models\weights_mnist_2C2F_con.mat')

cin = 1;
conv1 = W{1};
M = getRoesser(conv1,2);


%%
W = conv2fc(conv1,28,28,2,2);

uflat = flatten(image);

y_flat = W*uflat;

%%
image_tmp = zeros(1,cin,32,32);
image_tmp(1,:,3:30,3:30) = image;
image = image_tmp;

y = simulate2DSysRoesser(M,image,2);

image = reshape(double(image),[1,32,32]);

y_Filter = conv2D_multi(conv1,image,2);

error = sum(sum(sum(y-y_Filter).^2));

