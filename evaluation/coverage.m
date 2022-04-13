function [ loss ] = coverage( pred,true )
    pred=pred';
    true=true';
    
    [C,D] = sort(pred,2,'descend');
    Pos = zeros(size(pred,1) ,1);
    m = size(pred,1); d=  size(pred,2);
    for i=1:m
        for j=1:d
            if true(i,D(i,j)) == 1
                Pos(i,1) = j;
            end
        end
    end
    loss = mean((Pos-1)/d);

end

