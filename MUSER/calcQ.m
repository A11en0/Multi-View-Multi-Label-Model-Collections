function [stepsize] = calcQ(Q,grad_Q,X,V,W,ita)
    c = 0.1;
    stepsize = 1;
    Qn = Q - stepsize*grad_Q;
    oldobj = 0.5*norm(V-X'*Q*W,'fro')^2 + 0.5*ita*norm(Q,'fro');
    newobj = 0.5*norm(V-X'*Qn*W,'fro')^2 + 0.5*ita*norm(Qn,'fro');
    if newobj - oldobj > c*sum(sum(grad_Q.*(Qn-Q)))
        while true
            stepsize = stepsize*0.1;  % armijo rule
            Qn = Q - stepsize*grad_Q;
            newobj = 0.5*norm(V-X'*Qn*W,'fro')^2 + 0.5*ita*norm(Qn,'fro');
            if newobj - oldobj <= c*sum(sum(grad_Q.*(Qn-Q)))+eps
                break;
            end
        end
    else
        return;
    end