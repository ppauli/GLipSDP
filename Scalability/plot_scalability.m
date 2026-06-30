close all
clear all
clc

%%
load('results/results_all_CNN_trained.mat')

figure('position',[100,100,310,200])
subplot(1,1,1)
loglog(depth,time_SDP_mat(1,:)','-','Color',[0.5 0 0.5])
hold on
loglog(depth,time_SDP_mat(2,:)','-','Color',[0 0.5 0.5])
loglog(depth,time_SDP_mat(3,:)','-','Color',[0.5 0.5 0])
loglog(depth,time_SDP_sparse_mat(1,:)','--','Color',[0.5 0 0.5])
loglog(depth,time_SDP_sparse_mat(2,:)','--','Color',[0 0.5 0.5])
loglog(depth,time_SDP_sparse_mat(3,:)','--','Color',[0.5 0.5 0])
%loglog(2,time_LipSDP,'x','Color',[0.5 0 0.5])
xticks([2,4,8,16])
xlabel('depth')
ylabel('time')
legend('$c=8$','$c=16$','$c=32$','interpreter','latex','Location','northoutside','orientation','horizontal')
%grid on
xlim([min(depth),max(depth)])
%title('Computation Time')

matlab2tikz('scalability_comp_times.tex');
