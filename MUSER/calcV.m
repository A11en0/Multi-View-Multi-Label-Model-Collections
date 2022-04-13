function [stepsize] = calcV(V,grad_V,X,Q,W,B,L,Y,alpha,beta,ita)
    c = 0.1;
    stepsize = 1;
    Vn = V-stepsize*grad_V;
    oldobj = 0.5*(V-X'*Q*W) + 0.5*alpha*norm(Y-V*B,'fro')^2 + 0.5*beta*trace(V'*L*V) + 0.5*ita*norm(V,'fro')^2;
    newobj = 0.5*(Vn-X'*Q*W) + 0.5*alpha*norm(Y-Vn*B,'fro')^2 + 0.5*beta*trace(Vn'*L*Vn)+ 0.5*ita*norm(Vn,'fro')^2;
    if newobj - oldobj > c*sum(sum(grad_V.*(Vn-V)))
        while true
            stepsize = stepsize*0.1;  % armijo rule
            Vn = V - stepsize*grad_V;
            newobj = 0.5*(Vn-X'*Q*W) + 0.5*alpha*norm(Y-Vn*B,'fro')^2 + 0.5*beta*trace(Vn'*L*Vn)+ 0.5*ita*norm(Vn,'fro')^2;
            if newobj - oldobj <= c*sum(sum(grad_V.*(Vn-V)))+eps
                break;
            end
        end
    else
        return;
    end