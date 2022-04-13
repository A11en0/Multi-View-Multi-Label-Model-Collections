function alpha=get_view_weight(d,n,Laplacian, v_num ,r,max_iter)
% Description
%
%       get_view_weight takes,
%           n                   - the number of examples
%           d                   - Dimensions of low-dimensional space
%           vmum                - the number of views
%           Laplacian           - the Laplacian matrix for each view
%           max_iter            - the number of iterations
%          
%       and returns,
%           alpha               - the weight vector for each view

    alpha = ones(1,v_num) / v_num;
    Public_space=zeros(d,n);
    Previous_pb=ones(d,n);
    iteration = 0;
      
    while Previous_pb ~= Public_space
        
        %disp(sprintf('Iteration: %i', iteration));
        iteration = iteration +1;        
        Previous_pb = Public_space;
        w_L=zeros(n,n);
        for i=1:v_num
            w_L = w_L+Laplacian{i,1}*alpha(i);
        end
		
        [V,D] = eig(w_L);
        
        e = diag(D);
        [~ , I] = sort(e);
        Public_space = transpose(V(:,I(2:d+1)));
        isR=isreal(Public_space);
        if(~isR)
            break;
        end
        alpha = update_alpha( Public_space, Laplacian, v_num ,r );

        %disp(strcat('Alpha:',sprintf(' %f',alpha )));
          
        if iteration >= max_iter
           break; 
        end
    end
    