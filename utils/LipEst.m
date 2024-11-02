function [Lip,info,time] = LipEst(NN)

layers = NN.layers;
W = NN.weights;
depth = length(W);
strides = NN.strides;
pool = NN.pool;
pool_kernel = NN.pool_kernel;
pool_strides = NN.pool_strides;
cond = NN.cond;

rho = sdpvar;
if iscell(W{1})
    nu = size(W{1}{1},2);
else
    nu = size(W{1},2);
end

Qm = rho * speye(nu);
cons = [rho>=0];

depth2 = 0;
if cond ~= 1
    for ii = 1:depth
        if iscell(W{ii})
            for jj = 1:length(W{ii})
                W{ii}{jj} = cond*W{ii}{jj};
                depth2 = depth2 +1;
            end
        else
            W{ii} = cond*W{ii};
            depth2 = depth2 +1;
        end
    end
end

for ii = 1:depth-1
    layer = layers{ii};
    switch layer
        case 'conv'
            strides_ii = [strides(ii),pool_kernel(ii),pool_strides(ii)];
            [con, Qm] = get_con_conv(W{ii}, strides_ii, Qm, 0, pool{ii});
        case 'fc'
            if ii>=2
                if strcmp(layers{ii-1},'conv') || strcmp(layers{ii-1},'subn_conv') || strcmp(layers{ii-1},'res_conv') || strcmp(layers{ii-1},'res_conv2')
                    nu_fc = size(W{ii},2);
                    nu_conv = size(Qm,2);
                    Qm = kron(Qm,eye(nu_fc/nu_conv)); % order reverse because of the default flattening operation
                end
            end
            [con, Qm] = get_con_fc(W{ii}, Qm, 0);
        case 'subn_conv'
            strides_ii = [strides(ii),pool_kernel(ii),pool_strides(ii)];
            [con, Qm] = get_con_subn_conv(W{ii}, strides_ii, Qm, 0, 0, pool{ii},cond);
        case 'subn_fc'
            [con, Qm] = get_con_subn_fc(W{ii}, Qm, 0, 0, cond);
        case 'res_conv'
            strides_ii = [strides(ii),pool_kernel(ii),pool_strides(ii)];
            [con, Qm] = get_con_subn_conv(W{ii}, strides_ii, Qm, 0, 1, pool{ii},cond);
        case 'res_fc'
            [con, Qm] = get_con_subn_fc(W{ii}, Qm, 0, 1, cond);
        case 'res_conv2'
            strides_ii = [strides(ii),pool_kernel(ii),pool_strides(ii)];
            [con, Qm] = get_con_res_conv2(W{ii}, strides_ii, Qm, pool{ii}, 0);
        case 'res_fc2'
            [con, Qm] = get_con_res_fc2(W{ii}, Qm, 0);
    end
    cons = [cons,con];
end

layer = layers{depth};
switch layer
    case 'conv'
        strides_depth = [strides(depth),pool_kernel(depth),pool_strides(depth)];
        con = get_con_conv(W{depth}, strides_depth, Qm, 1, pool{depth});
    case 'fc'
        if depth>1
            if strcmp(layers{depth-1},'conv') || strcmp(layers{depth-1},'subn_conv') || strcmp(layers{depth-1},'res_conv') || strcmp(layers{depth-1},'res_conv2')
                nu_fc = size(W{depth},2);
                nu_conv = size(Qm,2);
                Qm = kron(Qm,eye(nu_fc/nu_conv)); % order reverse because of the default flattening operation
            end
        end
        con = get_con_fc(W{depth}, Qm, 1);
    case 'subn_conv'
        strides_depth = [strides(depth),pool_kernel(depth),pool_strides(depth)];
        con = get_con_subn_conv(W{depth}, strides_depth, Qm, 1, 0, pool{depth}, cond);
    case 'subn_fc'
        con = get_con_subn_fc(W{depth}, Qm, 1, 0);
    case 'res_conv'
        strides_depth = [strides(depth),pool_kernel(depth),pool_strides(depth)];
        con = get_con_subn_conv(W{depth}, strides_depth, Qm, 1, 1, pool{depth}, cond);
    case 'res_fc'
        con = get_con_subn_fc(W{depth}, Qm, 1, 1, cond);
    case 'res_conv2'
        strides_ii = [strides(depth),pool_kernel(depth),pool_strides(depth)];
        con = get_con_res_conv2(W{depth}, strides_ii, Qm, pool, 1);
    case 'res_fc2'
        con = get_con_res_fc2(W{depth}, Qm, 1);
end
cons = [cons,con];

sol = optimize(cons,rho);

Lip = sqrt(value(rho))/cond^depth2;
info = sol.info;
time = sol.solvertime;
end

function [con,Q] = get_con_conv(K, strides, Qm, last, pool)
sys = getRoesser(K,strides(1));
A = sys.A;
B = sys.B;
C = sys.C;
D = sys.D;

[nx1,nx2] = size(sys.A12);
[ny,~] = size(D);

P1 = sdpvar(nx1,nx1);
P2 = sdpvar(nx2,nx2);
P = blkdiag(P1,P2);

if strides(1)>1
    Qm = kron(eye(strides(1)^2),Qm);
end

if last == 1
    [~, Qm] = get_Q_conv(Qm,ny,pool,strides);
    con = [[P-A'*P*A-C'*C -A'*P*B-C'*D;...
        -B'*P*A-D'*C Qm-B'*P*B-D'*D]>=1E-10, P>=1E-10];
else
    [Q, Qm] = get_Q_conv(Qm,ny,pool,strides);
    Lambda = diag(sdpvar(1,ny));
    con = [[P-A'*P*A -A'*P*B -C'*Lambda;...
        -B'*P*A Qm-B'*P*B -D'*Lambda;...
        -Lambda*C -Lambda*D 2*Lambda-Q]>=1E-10, P>=1E-10, Lambda>=0];
end
end

function [con,Q] = get_con_fc(W,Qm,last)
[ny,nu] = size(W);

if last == 1
    con = [Qm-W'*W>=0];
    Q = [];
else
    Q = sdpvar(ny,ny);
    lam = sdpvar(ny,1);
    Lam = diag(lam);
    con = [[Qm -W'*Lam; -Lam*W 2*Lam-Q]>=0,lam>=0];
end
end

function [cons,Q] = get_con_subn_fc_alt(W,Qm,last)

depth = length(W);

Lower = [];
Middle = [Qm];
cons = [];

for ii = 1:depth-1
    [ny(ii),nu(ii)] = size(W{ii});
    lam{ii} = sdpvar(ny(ii),1);
    Lam = diag(lam{ii});
    Lower = blkdiag(Lower,-Lam*W{ii});
    Middle = blkdiag(Middle,2*Lam);
    cons = [cons,lam{ii}>=0];
end

if last == 1
    Q = [];
    Middle = Middle - blkdiag(zeros(sum(nu),sum(nu)),W{depth}'*W{depth});
else
    [ny(depth),nu(depth)] = size(W{depth});
    Q = sdpvar(ny(depth),ny(depth));
    lam{depth} = sdpvar(ny(depth),1);
    Lam = diag(lam{depth});
    Lower = blkdiag(Lower,-Lam*W{ii});
    Middle = blkdiag(Middle,2*Lam-Q);
    cons = [cons,lam{ii}>=0];
end

tmp = zeros(size(Middle));
tmp(nu(1)+1:end,1:sum(nu)) = Lower;
Lower = tmp;

M = Middle + Lower + Lower';

cons = [cons, M >= 10e-10];
end

function [cons,Q] = get_con_subn_fc(W, Qm, last, res, cond)
depth = length(W);

cons = [];

for ii = 1:depth
    [ny,nu] = size(W{ii});

    if (last == 1) && (ii == depth) && (res == 0)
        M{ii} = -W{ii}'*W{ii};
    elseif (last == 1) && (ii == depth) && (res == 1)
        M11 = -eye(ny)*cond^(2*depth);
        M12 = -W{ii}*cond^depth;
        M22 = -W{ii}'*W{ii};
    else
        lam = sdpvar(1,ny);
        Lambda = diag(lam);
        cons = [cons, lam >= 0];
        M{ii} = [zeros(nu,nu) -W{ii}'*Lambda;...
            -Lambda*W{ii} 2*Lambda];
    end
    n_vec(ii) = nu;
end

if last == 1
else
    n_vec = [n_vec, ny];
end

if res == 1 && n_vec(1) ~= ny
    error('Input and output dimensions must match in a ResNet!')
end

n_cum_vec(1) = 0;
for ii = 2:length(n_vec)+1
    n_cum_vec(ii) = n_cum_vec(ii-1) + n_vec(ii-1);
end

Mtot = zeros(n_cum_vec(end));
for ii = 1:depth
    tmp = zeros(n_cum_vec(end),'like',sdpvar);
    if (last == 1) && (ii == depth) && (res == 0)
        tmp(n_cum_vec(ii)+(1:n_vec(ii)),...
            n_cum_vec(ii)+(1:n_vec(ii))) = M{ii};
    elseif (last == 1) && (ii == depth) && (res == 1)
        tmp(1:n_vec(1),1:n_vec(1)) = M11;
        tmp(1:n_vec(1),n_cum_vec(ii)+(1:n_vec(ii))) = M12;
        tmp(n_cum_vec(ii)+(1:n_vec(ii)),1:n_vec(1)) = M12';
        tmp(n_cum_vec(ii)+(1:n_vec(ii)),n_cum_vec(ii)+(1:n_vec(ii))) = M22;
    else
        tmp(n_cum_vec(ii)+(1:sum(n_vec(ii-1+(1:2)))),...
            n_cum_vec(ii)+(1:sum(n_vec(ii-1+(1:2))))) = M{ii};
    end
    M{ii} = tmp;
    Mtot = Mtot + M{ii};
end

if res == 1 && last == 0
    tmp = zeros(n_cum_vec(end),'like',sdpvar);
    tmp(n_cum_vec(end-1)+1:end,1:ny) = -Lambda*cond^depth;
    tmp(1:ny,n_cum_vec(end-1)+1:end) = -Lambda*cond^depth;
    Mtot = Mtot + tmp;
end

if last == 1
    Q = [];
    Mtot = Mtot + blkdiag(Qm,zeros(n_cum_vec(end)-n_cum_vec(2)));%,-W{ii}'*W{ii});
else
    Q = sdpvar(ny,ny);
    Mtot = Mtot + blkdiag(Qm,zeros(n_cum_vec(end-1)-n_cum_vec(2)),-Q);
end

cons = [cons, Mtot >= 10e-10];
end

function [cons,Q] = get_con_res_fc2(W, Qm, last)
[ny,nu] = size(W{1});

lam = sdpvar(1,ny);
Lambda = diag(lam);

if last == 1
    Q = [];
    M = [Qm-eye(nu) -W{1}'*Lambda-W{2};...
        -Lambda*W{1}-W{2}' 2*Lambda-W{2}'*W{2}];
else
    Q = sdpvar(ny,ny);
    M = [Qm-Q -W{1}'*Lambda-Q*W{2};...
        -Lambda*W{1}-W{2}'*Q 2*Lambda-W{2}'*Q*W{2}];
end

cons = [M >= 10e-10];
end

function [cons,Q] = get_con_res_conv2(W, strides, Qm, pool, last)
sys1 = getRoesser(W{1},1);
A1 = sys1.A;
B1 = sys1.B;
C1 = sys1.C;
D1 = sys1.D;

nx1 = size(A1,strides(1));
[nx1_1,nx1_2] = size(sys1.A12);
[ny1,nu1] = size(D1);

sys2 = getRoesser(W{2},1);
A2 = sys2.A;
B2 = sys2.B;
C2 = sys2.C;
D2 = sys2.D;

nx2 = size(A2,strides(1));
[nx2_1,nx2_2] = size(sys2.A12);
[ny2,nu2] = size(D2);

lam = sdpvar(1,ny1);
Lambda = diag(lam);
P1_1 = sdpvar(nx1_1,nx1_1);
P1_2 = sdpvar(nx1_2,nx1_2);
P2_1 = sdpvar(nx2_1,nx2_1);
P2_2 = sdpvar(nx2_2,nx2_2);
P1 = blkdiag(P1_1,P1_2);
P2 = blkdiag(P2_1,P2_2);

if last == 1
    Q = [];
    [~,Qm] = get_Q_conv(Qm,ny2,pool,strides);
    M = [Qm-B1'*P1*B1-eye(nu1) -B1'*P1*A1     -D1'*Lambda-D2               -C2;...
        -A1'*P1*B1             P1-A1'*P1*A1   -C1'*Lambda                  zeros(nx1,nx2);...
        -Lambda*D1-D2'         -Lambda*C1     2*Lambda-B2'*P2*B2-D2'*D2    -B2'*P2*A2-D2'*C2;...
        -C2'                   zeros(nx2,nx1) -A2'*P2*B2-C2'*D2            P2-A2'*P2*A2-C2'*C2];
else
    [Q,Qm] = get_Q_conv(Qm,ny2,pool,strides);
    M = [Qm-B1'*P1*B1-Q  -B1'*P1*A1     -D1'*Lambda-Q*D2             -Q*C2;...
        -A1'*P1*B1       P1-A1'*P1*A1   -C1'*Lambda                  zeros(nx1,nx2);...
        -Lambda*D1-D2'*Q -Lambda*C1     2*Lambda-B2'*P2*B2-D2'*Q*D2  -B2'*P2*A2-D2'*Q*C2;...
        -C2'*Q           zeros(nx2,nx1) -A2'*P2*B2-C2'*Q*D2          P2-A2'*P2*A2-C2'*Q*C2];
end

cons = [M >= 10e-10];
end

function [cons,Q] = get_con_subn_conv(W, strides, Qm, last, res, pool, cond)
depth = length(W);

cons = [];

for ii = 1:depth
    sys = getRoesser(W{ii},1);
    A = sys.A;
    B = sys.B;
    C = sys.C;
    D = sys.D;

    [nx1,nx2] = size(sys.A12);
    [ny,nu] = size(D);

    P1 = sdpvar(nx1,nx1);
    P2 = sdpvar(nx2,nx2);
    P = blkdiag(P1,P2);
    cons = [cons, P >= 10e-10];
    if (last == 1) && (ii == depth) && (res == 0)
        M{ii} = [-B'*P*B-D'*D  -B'*P*A-D'*C;...
            -A'*P*B-C'*D P-A'*P*A-C'*C];
    elseif (last == 1) && (ii == depth) && (res == 1)
        M11 = -eye(ny)*cond^(2*depth);
        M22 = [-B'*P*B-D'*D -B'*P*A-D'*C;...
            -A'*P*B-C'*D P-A'*P*A-C'*C];
        M12 = [-D -C]*cond^depth;
    else
        lam = sdpvar(1,ny);
        Lambda = diag(lam);
        cons = [cons,lam>=0];
        M{ii} = [-B'*P*B -B'*P*A -D'*Lambda;...
            -A'*P*B P-A'*P*A -C'*Lambda;...
            -Lambda*D -Lambda*C  2*Lambda];
    end

    n_vec(2*ii-1) = nu;
    n_vec(2*ii) = nx1+nx2;
end

if last == 1
else
    n_vec = [n_vec, ny];
end

if res == 1 && n_vec(1) ~= ny
    error('Input and output dimensions must match in a ResNet!')
end

n_cum_vec(1) = 0;
for ii = 2:length(n_vec)+1
    n_cum_vec(ii) = n_cum_vec(ii-1) + n_vec(ii-1);
end

Mtot = zeros(n_cum_vec(end));
for ii = 1:depth
    tmp = zeros(n_cum_vec(end),'like',sdpvar);
    if (last == 1) && (ii == depth) && (res == 0)
        tmp(n_cum_vec(2*(ii-1)+1)+(1:sum(n_vec(2*(ii-1)+(1:2)))),...
            n_cum_vec(2*(ii-1)+1)+(1:sum(n_vec(2*(ii-1)+(1:2))))) = M{ii};
    elseif (last == 1) && (ii == depth) && (res == 1)
        tmp(n_cum_vec(2*ii-1)+(1:sum(n_vec(ii-1:ii))),n_cum_vec(2*ii-1)+(1:sum(n_vec(ii-1:ii)))) = M22;
        tmp(1:n_vec(1),1:n_vec(1)) = M11;
        tmp(1:n_vec(1),n_cum_vec(2*ii-1)+(1:sum(n_vec(2*ii-1:2*ii)))) = M12;
        tmp(n_cum_vec(2*ii-1)+(1:sum(n_vec(2*ii-1:2*ii))),1:n_vec(1)) = M12';
    else
        tmp(n_cum_vec(2*(ii-1)+1)+(1:sum(n_vec(2*(ii-1)+(1:3)))),...
            n_cum_vec(2*(ii-1)+1)+(1:sum(n_vec(2*(ii-1)+(1:3))))) = M{ii};
    end
    M{ii} = tmp;
    Mtot = Mtot + M{ii};
end

if res == 1 && last == 0
    tmp = zeros(n_cum_vec(end),'like',sdpvar);
    tmp(n_cum_vec(end-1)+1:end,1:ny) = -Lambda*cond^depth;
    tmp(1:ny,n_cum_vec(end-1)+1:end) = -Lambda*cond^depth;
    Mtot = Mtot + tmp;
end

if last == 1
    [~,Qm] = get_Q_conv(Qm,ny,pool,strides);
    Mtot = Mtot + blkdiag(Qm,zeros(n_cum_vec(end)-n_cum_vec(2)));
else
    [Q,Qm] = get_Q_conv(Qm,ny,pool,strides);
    Mtot = Mtot + blkdiag(Qm,zeros(n_cum_vec(end-1)-n_cum_vec(2)),-Q);
end

cons = [cons, Mtot>=10e-10];
end

function mu = get_mu_max(r,s)
mu = ceil(r/s)^2;
end

function mu = get_mu_av(r,s)
K = 1/r^2*ones(1,1,r,r);
sys = getRoesser(K,s);
A = sys.A;
B = sys.B;
C = sys.C;
D = sys.D;

[nx1,nx2] = size(sys.A12);

P1 = sdpvar(nx1,nx1);
P2 = sdpvar(nx2,nx2);
P = blkdiag(P1,P2);

mu = sdpvar;

con = [[P-A'*P*A-C'*C -A'*P*B-C'*D;...
    -B'*P*A-D'*C mu*eye(s^2)-B'*P*B-D'*D]>=10e-10,P>=10e-10];

optimize(con,mu)
mu = value(mu);
end

function [Q,Qm] = get_Q_conv(Qm,ny,pool,strides)
if strcmp(pool,'max')
    mu = get_mu_max(strides(2),strides(3));
    Qm = 1/mu*Qm;
    Q = diag(sdpvar(1,ny));
elseif strcmp(pool,'av')
    mu = get_mu_av(strides(2),strides(3));
    Qm = 1/mu*Qm;
    Q = sdpvar(ny,ny);
else
    Q = sdpvar(ny,ny);
end
end