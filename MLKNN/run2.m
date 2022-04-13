warning off 
clear all
clc
addpath(genpath('C:/Users/Administrator/Desktop/evaluation'));
%dataset_lists = {'Mirflickr'};
dataset_lists = {'emotions'};
%dataset_lists = {'Pascal'};
totalCV = 5; % Fold number

Num = 10;  % Number of neighbors used in the k-nearest neighbor algorithm
Smooth = 1; % moothing parameter
% tuneThreshold = 1;

for i = 1: length(dataset_lists)
    %% load the dataset
    load(dataset_lists{i});
    if exist('train_data','var')==1
        data    = [train_data;test_data]; % 1*num_view
        target  = [train_target,test_target]; % L*N
        if deleteData == 1
            clear train_data test_data train_target test_target
        end
    end
    
%     data = reshape(data, size(data,2), size(data,1));
    
   X = cell2mat(data); % view-data concat
%     X=cat(2,data{1},data{2});
%    
%    for i=3:length(data)
%        X=cat(2,X,data{i});
%    end
%   
   
    kk = randperm(size(X,1)); % shuffle
    cvResult  = zeros(16, totalCV);
    
    target(target == 0) = -1;
    [target,noisy_num]=rand_noisy(target,3,0.3);
    disp("finish")
    for cv = 1 : totalCV
        [ train_x,train_y,test_x,test_y ] = genFoldSets(X,target',kk,cv, totalCV);
        %[train_y2,noisy_num_trian]=rand_noisy(train_y',3,0.3);
        %[test_y2,noisy_num_test]=rand_noisy(test_y',3,0.3);
        out = MLKNN(train_x, train_y, test_x, test_y);
        [Prior, PriorN, Cond, CondN] = MLKNN_train(train_x, train_y', Num, Smooth);
        [HammingLoss1,RankingLoss1, OneError1, Coverage1, Average_Precision1, Outputs, Pre_Labels] = MLKNN_test(train_x, train_y', test_x, test_y', Num, Prior, PriorN, Cond, CondN); % Performing the test procedure
        
        %% Prediction and evaluation
    %     if tuneThreshold == 1
    %         fscore                 = (train_x*modelLSML.W)';
    %         [ tau,  currentResult] = TuneThreshold( fscore, train_y', 1, 2);
    %         Pre_Labels             = Predict(Outputs,tau);
    %     else
    %         Pre_Labels = double(Outputs>=0.5);
    %     end
        
        target(target == -1) = 0;
        Pre_Labels(Pre_Labels == -1) = 0;
        fprintf('-- Evaluation\n');
        ResultAll = EvaluationAll(Pre_Labels, Outputs, test_y');
        cvResult(:, cv) = cvResult(:, cv) + ResultAll;
        Avg_Result      = zeros(16,2);
        Avg_Result(:,1) = mean(cvResult,2);
        Avg_Result(:,2) = std(cvResult,1,2);
        PrintResults(Avg_Result);
    end
end