close all
clear all
clc

load('New_weights_with_weight_decay/weights_mnist_4C3F_wd_4.mat')

W{1} = double(weights.('conv1.weight'));
W{2} = double(weights.('conv2.weight'));
W{3} = double(weights.('conv3.weight'));
W{4} = double(weights.('conv4.weight'));
W{5} = double(weights.('fc1.weight'));
W{6} = double(weights.('fc2.weight'));
W{7} = double(weights.('fc3.weight'));