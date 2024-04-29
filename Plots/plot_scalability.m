close all
clear all
clc

%%
load('../results/results_all.mat')

figure('position',[100,100,310,200])
subplot(1,1,1)
loglog(depth,time_SDP_mat(1,:)','-','Color',[0.5 0 0.5])
hold on
loglog(depth,time_SDP_mat(2,:)','-','Color',[0 0.5 0.5])
loglog(depth,time_SDP_mat(3,:)','-','Color',[0.5 0.5 0])
loglog(depth,time_SDP_sparse_mat(1,:)','--','Color',[0.5 0 0.5])
loglog(depth,time_SDP_sparse_mat(2,:)','--','Color',[0 0.5 0.5])
loglog(depth,time_SDP_sparse_mat(3,:)','--','Color',[0.5 0.5 0])
loglog(2,time_LipSDP,'x','Color',[0.5 0 0.5])
xticks([2,4,8,16])
xlabel('depth')
ylabel('time')
legend('$c=8$','$c=16$','$c=32$','interpreter','latex','Location','northoutside','orientation','horizontal')
%grid on
xlim([min(depth),max(depth)])
%title('Computation Time')

matlab2tikz('scalability_comp_times.tex');

% figure
% subplot(1,1,1)
% bar([log(L_SDP_mat(1,:))',log(L_triv_mat(1,:))';...
%     log(L_SDP_mat(2,:))',log(L_triv_mat(2,:))';...
%     log(L_SDP_mat(3,:))',log(L_triv_mat(3,:))'])
% xlabel('depth')
% ylabel('time')
% legend('$c=8$','$c=16$','$c=32$','interpreter','latex')
% %grid on
% %xlim([min(depth),max(depth)])
% title('Computation Time')
%%
load('results/L_triv_mat.mat')

figure('position',[100,100,300,200])
subplot(1,1,1)
loglog(depth,time_SDP_mat(1,:)','-','Color',[0.5 0 0.5])
hold on
loglog(depth,time_SDP_mat(2,:)','-','Color',[0 0.5 0.5])
loglog(depth,time_SDP_mat(3,:)','-','Color',[0.5 0.5 0])
loglog(depth,time_SDP_sparse_mat(1,:)','--','Color',[0.5 0 0.5])
loglog(depth,time_SDP_sparse_mat(2,:)','--','Color',[0 0.5 0.5])
loglog(depth,time_SDP_sparse_mat(3,:)','--','Color',[0.5 0.5 0])
xlabel('depth')
ylabel('time')
legend('$c=8$','$c=16$','$c=32$','interpreter','latex')
grid on
xlim([min(depth),max(depth)])
title('Computation Time')

figure('position',[100,100,350,110])
subplot(1,3,1)
bar([log([L_LipSDP; NaN; NaN; NaN]), log(L_SDP_mat(1,:))',log(L_triv_mat(1,:))'])
set(gca,'xticklabel',{2,4,8,16})
xlabel('$c=8$','interpreter','latex')
ylabel('$\rho$','interpreter','latex')
legend('LipSDP','GLipSDP','MP','interpreter','latex','Location','north','orientation','horizontal')
subplot(1,3,2)
bar([log([NaN; NaN; NaN; NaN]), log(L_SDP_mat(2,:))',log(L_triv_mat(2,:))'])
set(gca,'xticklabel',{2,4,8,16})
xlabel('$c=16$','interpreter','latex')
subplot(1,3,3)
bar([log([NaN; NaN; NaN; NaN]), log(L_SDP_mat(3,:))',log(L_triv_mat(3,:))'])
set(gca,'xticklabel',{2,4,8,16})
xlabel('$c=32$','interpreter','latex')
%legend('GLipSDP','MP','interpreter','latex','Location','northwest')
%grid on
%xlim([min(depth),max(depth)])
%title('Lipschitz bound')

%matlab2tikz('scalability_Lip_bounds.tex');