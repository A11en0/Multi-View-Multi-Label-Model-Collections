%DEMO This is an examplar file on how the PARTICLE-journal version program could be used
%The main function is "PAR_train.m" , "PAR_predict.m" ,"PAR_VLS" and "PAR_MAP".
%
%
% Copyright: Jun-Peng Fang (fangjp@seu.edu.cn),
%   Min-Ling Zhang (mlzhang@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University,
%   Nanjing 211189, China
%

% Loading the file containing the necessary inputs for calling the PARTICLE function
load('Pascal');
target = target';
data = double(cat(2, data{:}));
% Set parameters for the PARTICLE-journal version algorithm
nfold = 5;                 %ten fold crossvalidation
k=10;                       % k-nearstneighbor
alpha=0.95;              % A balancing coefficient parameter
str=' -t 0 -c 1';         % cllibsvm parameter
o=0.9;                      % label confidence threshold parameter
mode=1;                  % 0 means PARTICLE-VLS, 1 means PARTICLE-MAP
partial_labels = rand_noisy(target, 3, 0.7);
[n_sample,~]= size(data);
result=zeros(nfold,5); %save evaluation result
n_test = round(n_sample/nfold);
prelab=[];  %save step 1 result, the most credible labels.
I = 1:n_sample;
for i=1:nfold%
    fprintf('data2 processing,Cross validation: %d\n', i);
    start_ind = (i-1)*n_test + 1;
    if start_ind+n_test-1 > n_sample
        test_ind = start_ind:n_sample;
    else
        test_ind = start_ind:start_ind+n_test-1;
    end
    train_ind = setdiff(I,test_ind);
    train_data = data(train_ind, :);
    train_p_target = partial_labels(:,train_ind);
    test_data = data(test_ind,:);
    test_p_target = partial_labels(:, test_ind);
    model = PAR_train(train_data,train_p_target,k,alpha);
    lab = PAR_predict(train_data,test_data,test_p_target,model,o);
    prelab=[prelab,lab];
end
prelab=[prelab,partial_labels(:,n_test*nfold+1:n_sample)];
cvResult  = zeros(16, 5);
for i=1:nfold%
    fprintf('result processing,Cross validation: %d\n', i);
    start_ind = (i-1)*n_test + 1;
    if start_ind+n_test-1 > n_sample
        test_ind = start_ind:n_sample;
    else
        test_ind = start_ind:start_ind+n_test-1;
    end
    train_ind = setdiff(I,test_ind);
    train_data = data(train_ind, :);
    train_target = prelab(:,train_ind);
    pre_target=partial_labels(:,train_ind);
    test_data = data(test_ind,:);
    test_target = target(:, test_ind);
    pre_test_target = prelab(:, test_ind);
    if mode==0
        [Pre_Labels, Outputs] = PAR_VLS( train_data,train_target,pre_target,test_data,test_target,str);
    elseif mode==1
        [Pre_Labels, Outputs] = PAR_MAP( train_data,train_target,pre_target,test_data,test_target,str);
    end
    ResultAll = EvaluationAll(Pre_Labels, Outputs, test_target);
    cvResult(:, i) = cvResult(:, i) + ResultAll;
    Avg_Result      = zeros(16,2);
    Avg_Result(:,1) = mean(cvResult,2);
    Avg_Result(:,2) = std(cvResult,1,2); 
end
fprintf('-- Evaluation\n');
PrintResults(Avg_Result);  

