function y = ReLU(x);

for ii = 1:length(x)
    y(ii) = max(x(ii),0);
end

y = y';