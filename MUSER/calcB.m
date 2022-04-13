function [stepsize] = calcB(B,V,Y,grad_B)
    c = 0.1;
    stepsize = 1;
    Bn = B - stepsize*grad_B;
    oldobj = 0.5*norm(Y-V*B,'fro')^2;
    newobj = 0.5*norm(Y-V*Bn,'fro')^2;
    if newobj - oldobj > c*sum(sum(grad_B.*(Bn-B)))
        while true
            stepsize = stepsize*0.1;  % armijo rule
            Bn = B - stepsize*grad_B;
            newobj = 0.5*norm(Y-V*Bn,'fro')^2;
            if newobj - oldobj <= c*sum(sum(grad_B.*(Bn-B)))+eps
                break;
            end
        end
    else
        return;
    end