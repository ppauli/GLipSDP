function M = getRoesser(K,s)

[cout,cin,l1,l2] = size(K);

mat = reshape(permute(K(:, :, end:-1:1, end:-1:1), [1, 3, 2, 4]), [], l2*cin);
mat1 = [zeros(cout,cout*ceil((l1-s)/s)); eye(cout*ceil((l1-s)/s))];
mat2 = [zeros(cin*(l2-s),cin*s),eye(cin*(l2-s))];

A12 = mat(1:cout*(l1-s),1:cin*(l2-s));
B1 = mat(1:cout*(l1-s),cin*(l2-s)+1:cin*l2);
C2 = mat(cout*(l1-s)+1:end,1:cin*(l2-s));
D = mat(cout*(l1-s)+1:end,(l2-s)*cin+1:l2*cin);

A11 = mat1(1:cout*ceil((l1-s)/s),1:cout*ceil((l1-s)/s));
C1 = mat1(cout*ceil((l1-s)/s)+1:end,1:cout*ceil((l1-s)/s));

A22mat = mat2(1:cin*(l2-s),1:cin*(l2-s));
B2mat = mat2(1:cin*(l2-s),cin*(l2-s)+1:end);
A22 = A22mat;
B2 = B2mat;
for ii = 1:s-1
    A22 = blkdiag(A22,A22mat);
    B2 = blkdiag(B2,B2mat);
end

A12flat = flatten(A12,cout);
B1flat = flatten(B1,cout);
C2 = flatten(C2,cout);
D = flatten(D,cout);

r = s-mod(l1-s,s);
if r~=s
    A12flat = [zeros(cout,r*cin*(l2-s)) A12flat];
    B1flat = [zeros(cout,r*cin*s) B1flat];
end

A12 = [];
B1 = [];
for ii = 1:ceil((l1-s)/s)
    A12 = [A12; A12flat(:,(ii-1)*cin*(l2-s)*s+1:ii*cin*(l2-s)*s)];
    B1 = [B1; B1flat(:,(ii-1)*cin*s^2+1:ii*cin*s^2)];
end

A21 = zeros(cin*(l2-s)*s,cout*ceil((l1-s)/s));

A = [A11, A12; A21 A22];
B = [B1; B2];
C = [C1 C2];

M = struct();
M.A11 = A11;
M.A12 = A12;
M.A21 = A21;
M.A22 = A22;
M.B1 = B1;
M.B2 = B2;
M.C1 = C1;
M.C2 = C2;
M.D = D;
M.A = A;
M.B = B;
M.C = C;
M.D = D;
end

function mat2 = flatten(mat,cout)
    len = size(mat,1)/cout;
    mat2 = [];
    for ii = 1:len
        mat2 = [mat2, mat((ii-1)*cout+1:ii*cout,:)];
    end
end