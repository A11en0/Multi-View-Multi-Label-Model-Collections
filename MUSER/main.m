tic;%保存当前时间
%V(nxc) B(cxm) Q(dxd') W(d'xc)依次求该四项
clear all;
clc;
load ('Pascal');

% [partial_labels] = transLabel(partial_labels, 0);
% [target] = transLabel(target, 0);
partial_labels = rand_noisy(target', 3, 0.7);
data = data';
data = double(cat(2, data{:}));
P = partial_labels'; 
Y = target;
X = data;
score = zeros(12,5);
[row,col]=size(X);
indices=crossvalind('Kfold',X(1:row,col),10);
X = X';
cvResult  = zeros(16, 5);
for i=1:5
    test=(indices==i);
    train_data =X (:,~test);
    test_data=X(:,test);
    train_p_target=P(~test,:);
    test_target=Y(test,:);
    trainIndex=1:size(train_data,2);
    options = [];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'KNN';
    options.k = 10; 
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    S = constructW(train_data',options);
    L = diag(sum(S,2))-S;
    
    [W,Q,B,V] =  trainResult(train_data,train_p_target,L);
    
    Y_predict_ori = test_data'*Q*W*B;
    
%     [test_target] = transLabel(test_target, 0);
    [Y_predict] = transLabel(Y_predict_ori, 0.1);
    fprintf('-- Evaluation\n');
    ResultAll = EvaluationAll(Y_predict', Y_predict_ori', test_target');
    cvResult(:, i) = cvResult(:, i) + ResultAll;
    Avg_Result      = zeros(16,2);
    Avg_Result(:,1) = mean(cvResult,2);
    Avg_Result(:,2) = std(cvResult,1,2);
    PrintResults(Avg_Result); 
    ranking = Ranking_loss(Y_predict_ori',test_target') %#ok<NOPTS>
    hamming = Hamming_loss(Y_predict',test_target') %#ok<NOPTS>
    Coverage = coverage(Y_predict_ori',test_target') %#ok<NOPTS>
    one_error = One_error(Y_predict_ori',test_target') %#ok<NOPTS>
    average = Average_precision(Y_predict_ori',test_target') %#ok<NOPTS>
    
    score(i,1) = ranking;
    score(i,2) = hamming;
    score(i,3) = Coverage;
    score(i,4) = one_error;
    score(i,5) = average;
end

score(11,1) = mean(score(1:5,1));
score(11,2) = mean(score(1:5,2));
score(11,3) = mean(score(1:5,3));
score(11,4) = mean(score(1:5,4));
score(11,5) = mean(score(1:5,5));

score(12,1) = std(score(1:5,1));
score(12,2) = std(score(1:5,2));
score(12,3) = std(score(1:5,3));
score(12,4) = std(score(1:5,4));
score(12,5) = std(score(1:5,5));

toc;



