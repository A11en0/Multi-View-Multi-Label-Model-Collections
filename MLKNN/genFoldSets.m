function [ train_x,train_y,test_x,test_y ] = genFoldSets( X,Y,kk,index,totalCV )
% X: n by d data matrix
% Y: n by l label matrix
% kk = randperm(size(X,1));
% for cv = 1 : totalCV
%     [ train_x,train_y,test_x,test_y ] = generateCVSet( X,Y,kk,cv, totalCV);

    assert(index <= 10);
    assert(totalCV <= 10);
    m = size(X,1);
    slice = ceil(m/totalCV);
    test_x = X(kk((index - 1) * slice + 1: min( index * slice , m ) ) ,:);
    test_y = Y(kk((index - 1) * slice +1: min( index * slice , m ) ) ,:);
    
    train_x = X(setdiff(kk,kk((index - 1) * slice + 1: min( index * slice , m ) )),:);
    train_y = Y(setdiff(kk,kk((index - 1) * slice + 1: min( index * slice , m ) )),:);

end