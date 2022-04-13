function [ topkRate ] = topRate( Y_est, Y, k)
%TOPRATE Summary of this function goes here
%   Detailed explanation goes here
if min(Y(:))==-1
    Y = (Y+1)/2;
end
[N,L] = size(Y);
[maxV3,maxI3] = sort(Y_est,2,'descend');
count = 0;
for i = 1:N
        count = count+sum(Y(i,maxI3(i,1:k)));
end
topkRate = count/N/k;
end


