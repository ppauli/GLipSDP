function [L,info,time] = solve_SDP_sparse(K)

n_conv = length(K);

% Construct parallel system for the layers
A11 = [];
A12 = [];
A21 = [];
A22 = [];

B1 = [];
B2 = [];

C1 = [];
C2 = [];

D = [];

P_1 = [];
P_2 = [];

for ii = 1:n_conv
    M = getRoesser(K{ii},1);
    A11 = blkdiag(A11,M.A11);
    A12 = blkdiag(A12,M.A12);
    A21 = blkdiag(A21,M.A21);
    A22 = blkdiag(A22,M.A22);
    
    B1 = blkdiag(B1,M.B1);
    B2 = blkdiag(B2,M.B2);
    
    C1 = blkdiag(C1,M.C1);
    C2 = blkdiag(C2,M.C2);
    
    D = blkdiag(D,M.D);
    [nx1,nx2] = size(M.A12);
    P_1 = blkdiag(P_1,sdpvar(nx1));
    P_2 = blkdiag(P_2,sdpvar(nx2));
end

i_ch1 = size(K{1},2);
o_chl = size(K{end},1);

B11 = B1(:,1:i_ch1);
B21 = B2(:,1:i_ch1);
B12 = [B1(:,i_ch1+1:end),zeros(size(B1,1),o_chl)];
B22 = [B2(:,i_ch1+1:end),zeros(size(B2,1),o_chl)];

C11 = C1;
C12 = C2;
C21 = zeros(o_chl,size(A11,2));
C22 = zeros(o_chl,size(A22,2));

D11 = D(:,1:i_ch1);
D21 = zeros(o_chl,i_ch1);
D12 = [D(:,i_ch1+1:end),zeros(size(D,1),o_chl)];
D22 = [zeros(o_chl,size(D,1)-o_chl),eye(o_chl,o_chl)];

[n_1,n_2] = size(A12);
[l_1,m_1] = size(D11);
[l_2,m_2] = size(D22);

%% Define 2-D system LMI
%P_1 = sdpvar(n_1,n_1);
%P_2 = sdpvar(n_2,n_2);
Lambda_2D = sdpvar(l_1,1);
constraints = [Lambda_2D >= 0,P_1 >=0, P_2 >= 0];
Lambda_2D = diag(Lambda_2D);
gamma = sdpvar;

projection = [A11,A12,B11,B12;
              A21,A22,B21,B22;
              eye(n_1+n_2),zeros(n_1+n_2,m_1+m_2);
              C11,C12,D11,D12;
              zeros(l_1,n_1+n_2+m_1),eye(m_2);
              C21,C22,D21,D22;
              zeros(m_1,n_1+n_2),eye(m_1),zeros(m_1,m_2)];
center = blkdiag(-P_1,-P_2,P_1,P_2,[zeros(m_2,m_2),-Lambda_2D;-Lambda_2D,2*Lambda_2D],-eye(l_2),gamma*eye(m_1));
LMI1 = projection'*center*projection;

ops = sdpsettings('solver','mosek','verbose',1,'debug',1,'dualize',0);

sol = optimize([LMI1 >= 10*eps,constraints],gamma,ops);
L = sqrt(value(gamma));
info = sol.info;
time = sol.solvertime;
end