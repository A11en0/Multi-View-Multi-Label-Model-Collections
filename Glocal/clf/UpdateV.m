function [ V ] = UpdateV(V, J,Y,U,W,X,param )
   [k,n] = size(V);
   %manifold = stiefelfactory(n,k);
   manifold = euclideanfactory(n, k);
   problem.M = manifold;
   
  % problem.M.proj = @projection;
    % Define the problem cost function and its gradient.
%    problem.finalproj = @(x) Vproj(x);
    problem.cost = @(x) Vcost(x,J,Y,U,W,X,param);
    problem.grad = @(x) Vgrad(x,J,Y,U,W,X,param);
    % Numerically check gradient consistency.
    %checkgradient(problem);

    options = param.tooloptions;
    
    %[x xcost info] = conjugategradient(problem,V',options);
    [x xcost info] = steepestdescent(problem,V',options);
    %[x xcost info] = trustregions(problem,V',options);
    V = x';
end

function proj = Vproj(V)
   V(V<0) = 0;
   proj = V;
end

% function cost = Vcost(V,Y,U,W,X)
%      cost = 0.5*norm(Y-U*V)^2 + 0.5*norm(V-W'*X)^2;
% end
function cost = Vcost(V,J,Y,U,W,X,param)
     cost = 0.5*norm(J.*(Y-U*V'),'fro')^2 + 0.5*param.lambda3*norm(V,'fro')^2 + 0.5*param.lambda*norm(V'-W'*X,'fro')^2;
end
% function grad = Vgrad(V,Y,U,W,X)
%      grad = U'*(U*V-Y) + (V-W'*X);
% end
function grad = Vgrad(V,J,Y,U,W,X,param)
     grad = (U'*(J.*(U*V'-Y)))' + param.lambda*(V'-W'*X)' + param.lambda3*V;
end
