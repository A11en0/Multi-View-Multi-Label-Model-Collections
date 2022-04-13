 function alpha = update_alpha( Y, L, v_num,r )
        alpha = ones(1,v_num)/v_num;
        den_a = 0;
        for v_i = 1:v_num
           den_a = den_a + (1/trace(Y*L{v_i,1}*transpose(Y)))^(1/(r-1)); 
        end
        if(~isreal(den_a))
            return;
        end
        for i=1:v_num 
          alpha(1,i) = ((1/trace(Y*L{i,1}*transpose(Y)))^(1/(r-1)))/den_a;
        end
    end
