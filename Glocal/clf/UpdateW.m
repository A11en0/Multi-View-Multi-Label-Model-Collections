function [ W, WXXW,WXGXGWPool ] = UpdateW(W, U, V, X, betav, XX, XGXGPool, MgPool, param )

     
     k = param.k;
     [d,~] = size(X);
 
     manifold = euclideanfactory(d, k);
     problem.M = manifold;
     problem.cost = @(x) Wcost(x, U, V, X, betav, XX, XGXGPool, MgPool, param);
     problem.grad = @(x) Wgrad(x, U, V, X, betav, XX, XGXGPool, MgPool, param);
     %checkgradient(problem);
     
     options = param.tooloptions;
     %[x xcost info] = trustregions(problem,W,options);
     [x xcost info] = steepestdescent(problem,W,options);
     W = x;
     WXXW = W'*XX*W;
     WXGXGWPool = cell(size(MgPool));
     for i=1:length(MgPool)
         WXGXGWPool{i} = x'*XGXGPool{i}*x;
     end
end

function cost = Wcost(x,U, V, X, betav, XX, XGXGPool, MgPool, param)
 % cost = 0.5*param.lambda1*norm(V-W'*X,'fro')^2 + 0.5*param.lambda4*norm(W,'fro')^2+0.5*param.lambda5*trace(X'*W*U'*S*U*W'*X);
  lambda = param.lambda;
  lambda1 = param.lambda1;
  lambda2 = param.lambda2;
  lambdaw = param.lambda5;
  
  cost = lambda*norm(V-x'*X)^2 + lambdaw*norm(x)^2;
  UWXXWU = U*x'*XX*x*U';
  for i=1:length(MgPool)
      Mg = MgPool{i};
      UWXGXGWU = U*x'*XGXGPool{i}*x*U';
      cost = cost + lambda2*betav(i)*trace(UWXXWU*Mg) ...
          + lambda1*trace(UWXGXGWU*Mg);    
  end
end
function grad = Wgrad(x,U, V, X, betav, XX, XGXGPool, MgPool, param)
  lambda = param.lambda;
  lambda1 = param.lambda1;
  lambda2 = param.lambda2;
  lambdaw = param.lambda5;
  
  grad = lambda*XX*x - lambda*X*V' + lambdaw*x;
  for i=1:length(MgPool)
      Mg = MgPool{i};
      UMU = U'*Mg*U;
      grad = grad + (lambda2*betav(i)*XX + lambda1*XGXGPool{i})*x*UMU;
  end
end
