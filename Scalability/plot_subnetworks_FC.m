close all
clear all
clc

%%
load('results/subnetwork_tradeoff_FC_w16.mat')

figure
semilogy(time)
xticks([1,2,3,4,5,6])
xticklabels([32,16,8,4,2,1])

load('results/subnetwork_tradeoff_FC_w32.mat')
hold on
semilogy(time)

load('results/subnetwork_tradeoff_FC_w64.mat')
hold on
semilogy(time)

matlab2tikz('scalability_subnetworks_FC.tex');