function Wp = pool2fc(k,c,n,s) % kernel, channel, stride, input_size

Wp1 = conv2fc(1/(k*k)*ones(1,1,k,k),n,n,s,0);

Wp = kron(Wp1,eye(c));




% kernel = size(P,2);
% c = size(P,1);
% 
% P_flat=zeros(1,(kernel-1)*n+kernel);
% for ii=1:kernel
%     P_flat(1,(ii-1)*n+(1:kernel))=P(ii,1:kernel);
% end
% 
% W = zeros(n*n/kernel^2,n*n);
% for jj=1:n/kernel
%     for ii=1:n/kernel
%         W(ii+(jj-1)*(n/kernel),2*(ii-1)+2*(jj-1)*n+(1:length(P_flat))) = P_flat;
%     end
% end
% 
% Wp = [];
% 
% for ii = 1:c
%     Wp = blkdiag(Wp, W);
% end


