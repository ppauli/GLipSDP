function [Lip, time] = LipLT(W)

%% LipLT for feedforward fully-connected NNs
tic

m(1) = norm(W{1});

for kk = 1:length(W)-1
    H = W{1};
    for jj = 1:kk
        H = W{jj+1}*H*0.5;
    end
    sum = 0;
    for jj = 1:kk
        H2 = W{jj+1};
        for ii = jj+2:kk+1
            H2 = W{ii}*H2*0.5;
        end
        sum = sum + norm(H2)*m(jj);
    end
    m(kk+1) = norm(H) + 0.5*sum;
end

time = toc
Lip = m(end);

end