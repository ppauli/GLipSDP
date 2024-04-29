function W3 = conv2fc(K,N1,N2,s,pad)

[cout, cin, l1,l2] = size(K);

W = reshape(K(:,:,end:-1:1,end:-1:1),cout,[]);

Z = zeros(cout,cin*(N2-l2));

W2 = W(:,1:l2*cin);
for ii = 2:l2
    W2 = [W2 Z W(:,l2*cin*(ii-1)+(1:l2*cin))];
end

if pad == 0
    W3 = zeros(cout*ceil(N1/s-(l1-s)/s)*ceil(N2/s-(l2-s)/s),cin*N1*N2);
    for jj = 1:ceil(N1/s-(l1-s)/s)
        for ii = 1:ceil(N2/s-(l2-s)/s)
            W3((jj-1)*cout*(N1/s-(l1-s)/s)+(ii-1)*cout+(1:cout),(jj-1)*N2*cin*s+(ii-1)*cin*s+(1:(l2-1)*N2*cin+l2*cin)) = W2;
        end
    end
    
else
    W3 = zeros(cout*ceil((N1+l1-1)/s-(l1-s)/s)*ceil((N2+l2-1)/s-(l2-s)/s),cin*N1*N2);
    for jj = 1:ceil((N1+l1-1)/s-(l1-s)/s)
        for ii = 1:ceil((N2+l2-1)/s-(l2-s)/s)
            if ii == 1 || jj == 1 || ii == ceil((N1+l1-1)/s-(l1-s)/s) || jj == ceil((N2+l2-1)/s-(l2-s)/s)
            else
                W3((jj-1)*cout*(ceil((N1+l1-1)/s-(l1-s)/s))+(ii-1)*cout+(1:cout),(jj-2)*N2*cin*s+(ii-2)*cin*s+(1:(l2-1)*N2*cin+l2*cin)) = W2;
            end
        end
    end
end