function [betav, XGXGPool, XX, param ] = InitGroup( Y, X, T,param )
    
     [l,n] = size(Y);

    
     gp = unique(T);
     g = length(gp);
     param.g = g;
     XX = X*X';    
     XGXGPool = cell(g,1);     
     
     betav = ones(g,1);
     for i=1:g
       ii = T==gp(i);
       XgXg = X(:,ii)*X(:,ii)';
       XGXGPool{i,1}=XgXg;
       betav(i) = betav(i)*sum(ii)/n;
     end   
end