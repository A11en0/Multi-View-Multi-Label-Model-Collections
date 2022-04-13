function [stepsize] = calcW(W,X,Q,V,grad_W,mu)
    c = 0.1;
    stepsize = 1; 
    Wn = W-stepsize*grad_W; 
    oldobj = 0.5*norm(V-X'*Q*W,'fro')^2 + 0.5*mu*norm(W,'fro')^2;
    newobj = 0.5*norm(V-X'*Q*Wn,'fro')^2 + 0.5*mu*norm(Wn,'fro')^2;
    if newobj - oldobj > c*sum(sum(grad_W.*(Wn-W)))
        while true
            stepsize = stepsize*0.1;  % armijo rule
            Wn = W - stepsize*grad_W; 
            newobj = 0.5*norm(V-X'*Q*Wn,'fro')^2 + 0.5*mu*norm(W,'fro')^2;
            if newobj - oldobj <= c*sum(sum(grad_W.*(Wn-W)))+eps
                break;
            end
        end
    else
        return;
    end