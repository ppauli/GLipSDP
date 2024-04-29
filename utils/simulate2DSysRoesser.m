function y = simulate2DSysRoesser(M,u,s)
A11 = M.A11;
A12 = M.A12;
A21 = M.A21;
A22 = M.A22;
B1 = M.B1;
B2 = M.B2;
C1 = M.C1;
C2 = M.C2;
D = M.D;

[nx1,nx2] = size(A12);
[ny,nu] = size(D);


[~,cin,n_1,n_2] = size(u);
 
x1 = zeros(nx1,ceil(n_1/s),ceil(n_2/s));
x2 = zeros(nx2,ceil(n_1/s),ceil(n_2/s));
y = zeros(ny,ceil(n_1/s),ceil(n_2/s));

for ii = 1:n_1/s
    for jj = 1:n_2/s
        u_s = u(:,:,(ii-1)*s+1:ii*s,(jj-1)*s+1:jj*s);
        uflat = flip_flatten(u_s);
        x1(:,ii+1,jj) = A11*x1(:,ii,jj) + A12*x2(:,ii,jj) + B1*uflat;
        x2(:,ii,jj+1) = A21*x1(:,ii,jj) + A22*x2(:,ii,jj) + B2*uflat;
        y(:,ii,jj) = C1*x1(:,ii,jj) + C2*x2(:,ii,jj) + D*uflat;
    end
end
end

function uflat = flip_flatten(u_s)
uflat = reshape(permute(u_s,[2,1,4,3]),[],1);
end
