close all
clear all
clc

%%
load('results/subnetwork_tradeoff_ch8.mat')

figure
semilogy(time)
xticks([1,2,3,4,5])
xticklabels([16,8,4,2,1])

load('results/subnetwork_tradeoff_ch16.mat')
hold on
semilogy(time)

load('results/subnetwork_tradeoff_ch32.mat')
hold on
semilogy(time)

matlab2tikz('scalability_subnetworks.tex');