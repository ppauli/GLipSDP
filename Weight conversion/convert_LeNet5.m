close all
clear all
clc

load('New_weights_with_weight_decay/weights_mnist_LeNet5_wd.mat')

W{1} = double(weights.('layer1.0.weight'));
W{2} = double(weights.('layer2.0.weight'));
W{3} = double(weights.('fc.weight'));
W{4} = double(weights.('fc1.weight'));
W{5} = double(weights.('fc2.weight'));
