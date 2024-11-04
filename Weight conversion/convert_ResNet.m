close all
clear all
clc

load('New_weights_with_weight_decay/weights_mnist_res_wd_7.mat')

W{1} = double(weights.('conv1.weight'));
W{2}{1} = double(weights.('layer1.0.conv1.weight'));
W{2}{2} = double(weights.('layer1.0.conv2.weight'));
W{3}{1} = double(weights.('layer1.1.conv1.weight'));
W{3}{2} = double(weights.('layer1.1.conv2.weight'));
W{4}{1} = double(weights.('layer2.0.conv1.weight'));
W{4}{2} = double(weights.('layer2.0.conv2.weight'));
W{5}{1} = double(weights.('layer2.1.conv1.weight'));
W{5}{2} = double(weights.('layer2.1.conv2.weight'));
W{6}{1} = double(weights.('layer3.0.conv1.weight'));
W{6}{2} = double(weights.('layer3.0.conv2.weight'));
W{7}{1} = double(weights.('layer3.1.conv1.weight'));
W{7}{2} = double(weights.('layer3.1.conv2.weight'));
W{8}{1} = double(weights.('layer4.0.conv1.weight'));
W{8}{2} = double(weights.('layer4.0.conv2.weight'));
W{9}{1} = double(weights.('layer4.1.conv1.weight'));
W{9}{2} = double(weights.('layer4.1.conv2.weight'));
W{10} = double(weights.('fc.weight'));
