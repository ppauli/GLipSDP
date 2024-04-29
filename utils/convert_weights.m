function W = convert_weights(weights,layers)

for ii = 1:layers
    str = ['layers.',num2str(ii),'.weight'];
    W{ii} = double(weights.(str));
end
