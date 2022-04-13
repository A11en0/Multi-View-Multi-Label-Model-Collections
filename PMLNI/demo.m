
clear;

warning('off');

% load data
load('emotion_4.mat');
target(target==-1)=0;
pLabels = rand_noisy(target, 3, 0.3);
pLabels(pLabels==-1)=0;

% set the parameters
opt.lambda = 10;
opt.beta = 0.5;
opt.gamma = 0.5;
opt.max_iter = 500;

N = length(target);
cvResult  = zeros(16, 5);
for i=1:5
    indices = crossvalind('Kfold', 1:N ,5);
    test_idxs = (indices == 1);
    train_idxs = ~test_idxs;

    train_data=data(train_idxs,:);train_target=pLabels(train_idxs,:);true_target = target(train_idxs,:);
    test_data=data(test_idxs,:);test_target=target(test_idxs,:);

    % pre-processing
    [train_data, settings]=mapminmax(train_data');
    test_data=mapminmax('apply',test_data',settings);
    train_data(find(isnan(train_data)))=0;
    test_data(find(isnan(test_data)))=0;
    train_data=train_data';
    test_data=test_data';

    % training
    model = PMLNI_train(train_data, train_target, true_target, opt);

    % testing
    [Pre_Labels, Outputs] = PMLNI_test( test_data,test_target,model);
    ResultAll = EvaluationAll(Pre_Labels', Outputs', test_target');
    cvResult(:, i) = cvResult(:, i) + ResultAll;
    Avg_Result      = zeros(16,2);
    Avg_Result(:,1) = mean(cvResult,2);
    Avg_Result(:,2) = std(cvResult,1,2);
end
fprintf('-- Evaluation\n');
PrintResults(Avg_Result);
    


