function [L,info,time] = LipschitzEstimationFazlyab(W)

n_tot = length(W);

AC = [];
n_vec = [];
for ii = 1:n_tot
    AC = blkdiag(AC,W{ii});
    n_vec = [n_vec size(W{ii},2)];
    if ii<n_tot && size(W{ii},1)~=size(W{ii+1},2)
        error('Dimensions do not match.')
    end
end
n_vec = [n_vec size(W{end},1)];

n = sum(n_vec(2:end-1));

A = AC(1:n,:);
B = [zeros(n,n_vec(1)) eye(n)];

rho = sdpvar;
Lam = diag(sdpvar(n,1));

alpha = 0;
beta = 1;

M = [A;B]'*[-2*alpha*beta*Lam (alpha+beta)*Lam; (alpha+beta)*Lam -2*Lam]*[A;B];

P = blkdiag(-rho*eye(n_vec(1)),zeros(sum(n_vec(2:end-2))),W{end}'*W{end});

sol = optimize([M+P<=-1E-10,Lam>=0],rho)
info = sol.info;
time = sol.solvertime;

L = sqrt(value(rho));