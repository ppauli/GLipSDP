function y_mat = conv2D_multi(K,u,s)

[~,n1,n2] = size(u);
[cout,cin,l1,l2] = size(K);

u2 = zeros(cin,n1+l1-1,n2+l2-1);
u2(:,l1:n1+l1-1,l2:n2+l2-1) = u;

y_mat = zeros(cout,floor(n1/s),floor(n2/s));
for kk = 1:floor(n1/s)
    for ll = 1:floor(n2/s)
        for mm = 1:cout
            y = 0;
            for ii = 1:l1
                for jj = 1:l2
                    y = y + K(mm,1:cin,ii,jj)*u2(1:cin,s*kk+l1-ii,s*ll+l2-jj);
                end
            end
            y_mat(mm,kk,ll) = y;
        end
    end
end