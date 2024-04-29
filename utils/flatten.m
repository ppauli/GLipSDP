function uflat = flatten(u)
uflat = reshape(permute(u,[1,2,3,4]),[],1);
end