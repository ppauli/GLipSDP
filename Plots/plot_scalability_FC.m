close all
clear all
clc

%%
load('results/results_all_FC_3.mat')

figure('position',[100,100,310,200])
subplot(1,1,1)
loglog(depth,time_GLipSDP_mat(1,:)','-','Color',[0.5 0 0.5])
hold on
loglog(depth,time_GLipSDP_mat(2,:)','-','Color',[0 0.5 0.5])
loglog(depth,time_GLipSDP_mat(3,:)','-','Color',[0.5 0.5 0])
loglog(depth,time_GLipSDP_mat(4,:)','-','Color',[0.5 0.5 0.5])
loglog(depth,time_LipSDP_mat(1,:)','--','Color',[0.5 0 0.5])
loglog(depth,time_LipSDP_mat(2,:)','--','Color',[0 0.5 0.5])
loglog(depth,time_LipSDP_mat(3,:)','--','Color',[0.5 0.5 0])
loglog(depth,time_LipSDP_mat(4,:)','--','Color',[0.5 0.5 0.5])
%loglog(depth,time_LipLT_mat(1,:)','.-','Color',[0.5 0 0.5])
%loglog(depth,time_LipLT_mat(2,:)','.-','Color',[0 0.5 0.5])
%loglog(depth,time_LipLT_mat(3,:)','.-','Color',[0.5 0.5 0])
%loglog(depth,time_LipLT_mat(4,:)','.-','Color',[0.5 0.5 0.5])
xticks([2,4,8,16,32,64,128])
xlabel('depth')
ylabel('time')
legend('$c=8$','$c=16$','$c=32$','$c=64$','interpreter','latex','Location','northoutside','orientation','horizontal')
%grid on
xlim([min(depth),max(depth)])
%title('Computation Time')

matlab2tikz('scalability_comp_times_FC.tex');

%%
figure('position',[100,100,350,350])
subplot(2,2,1)
bar([L_LipSDP_mat(1,1:3)', L_GLipSDP_mat(1,1:3)', L_LipLT_mat(1,1:3)', L_triv_mat(1,1:3)'])
set(gca,'xticklabel',{4,8,16})
xlabel('$c=8$','interpreter','latex')
ylabel('$\rho$','interpreter','latex')
legend('LipSDP','GLipSDP','LipLT','MP','interpreter','latex','Location','north','orientation','horizontal')
subplot(2,2,2)
bar([L_LipSDP_mat(2,1:3)', L_GLipSDP_mat(2,1:3)', L_LipLT_mat(2,1:3)', L_triv_mat(2,1:3)'])
set(gca,'xticklabel',{4,8,16})
xlabel('$c=16$','interpreter','latex')
subplot(2,2,3)
bar([L_LipSDP_mat(3,1:3)', L_GLipSDP_mat(3,1:3)', L_LipLT_mat(3,1:3)', L_triv_mat(3,1:3)'])
set(gca,'xticklabel',{4,8,16})
xlabel('$c=32$','interpreter','latex')
subplot(2,2,4)
bar([L_LipSDP_mat(4,1:3)', L_GLipSDP_mat(4,1:3)', L_LipLT_mat(4,1:3)', L_triv_mat(4,1:3)'])
set(gca,'xticklabel',{4,8,16})
xlabel('$c=64$','interpreter','latex')

matlab2tikz('scalability_Lip_bounds_FC.tex');
