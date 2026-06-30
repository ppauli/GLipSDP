close all
clear all
clc

%%
load('results/results_all_FC_trained.mat')

figure('position',[100,100,310,200])
subplot(1,1,1)
loglog(depth,time_GLipSDP_mat(1,:)','-','Color',[0.5 0 0.5])
hold on
loglog(depth,time_GLipSDP_mat(2,:)','-','Color',[0 0.5 0.5])
loglog(depth,time_GLipSDP_mat(3,:)','-','Color',[0.5 0.5 0])
loglog(depth,time_LipSDP_mat(1,:)','--','Color',[0.5 0 0.5])
loglog(depth,time_LipSDP_mat(2,:)','--','Color',[0 0.5 0.5])
loglog(depth,time_LipSDP_mat(3,:)','--','Color',[0.5 0.5 0])
xticks([2,4,8,16,32,64,128])
xlabel('depth')
ylabel('time')
legend('$c=16$','$c=32$','$c=64$','interpreter','latex','Location','northoutside','orientation','horizontal')
%grid on
xlim([min(depth),max(depth)])
%title('Computation Time')

matlab2tikz('scalability_comp_times_FC.tex');
