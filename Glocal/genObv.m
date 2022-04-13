function [J] = genObv( train_target, rho )
%GENERATEOBV Summary of this function goes here
%   Detailed explanation goes here
     if rho == 1
         J = ones(size(train_target));
         return;
     end
     J = zeros(size(train_target));
     for i=1:size(train_target,1)
         y = train_target(i,:);
         pos = find(y==1);
         neg = find(y~=1);
         pi = randperm(length(pos));
         pi = pi(1:ceil(rho*length(pos)));
         J(i,pos(pi)) = 1;
         ni = randperm(length(neg));
         ni = ni(1:ceil(rho*length(neg)));
         J(i,neg(ni)) = 1;
     end
end

