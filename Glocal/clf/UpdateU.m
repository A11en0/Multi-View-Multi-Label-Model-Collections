function [ U ] = UpdateU(U, J, Y, V, betav,MgPool,WXXW,WXGXGWPool, param )

     
     k = param.k;
     [l,~] = size(Y);
 
     manifold = euclideanfactory(l, k);
     problem.M = manifold;
     problem.cost = @(x) Ucost(x,J,Y,V,betav,MgPool,WXXW,WXGXGWPool,param);
     problem.grad = @(x) Ugrad(x,J,Y,V,betav,MgPool,WXXW,WXGXGWPool,param);
 %    problem.finalproj = @(x) Uproj(x);
 
     options = param.tooloptions;
     %[x xcost info] = trustregions(problem,U,options);
     [x xcost info] = steepestdescent(problem,U,options);
    % [x xcost info] = conjugategradient(problem,U,options);
     U = x;

end
function proj = Uproj(U)
   U(U<0) = 0;
   proj = U;
end
function cost = Ucost(x,J,Y,V,betav,MgPool,WXXW,WXGXGWPool,param)

  lambdau = param.lambda4;
  lambda1 = param.lambda1;
  lambda2 = param.lambda2;
  cost = norm(J.*(Y-x*V))^2 + lambdau*norm(x)^2;
  for i=1:length(MgPool)
      Mg = MgPool{i};
      cost = cost + lambda2*betav(i)*trace(x*WXXW*x'*Mg) ...
          + lambda1*trace(x*WXGXGWPool{i}*x'*Mg);    
  end
end
function grad = Ugrad(x,J,Y,V,betav,MgPool,WXXW,WXGXGWPool,param)
  
  lambdau = param.lambda4;
  lambda1 = param.lambda1;
  lambda2 = param.lambda2;
  grad = J.*(x*V-Y)*V' + lambdau*x;
  for i=1:length(MgPool)
      Mg = MgPool{i};      
      grad = grad + Mg*x*(lambda2*betav(i)*WXXW + lambda1*WXGXGWPool{i});
  end
end
