close all
clear all
clc

%%
load('../results/subnetwork_tradeoff.mat')

figure
semilogy(time)
xticks([1,2,3,4,5])
xticklabels([16,8,4,2,1])

load('../results/subnetwork_tradeoff_8.mat')
hold on
semilogy(time)

matlab2tikz('scalability_subnetworks.tex');