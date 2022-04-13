warning off 
clear all
clc

dataset_lists = {'emotions'};

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

    X = cell2mat(data); % view-data concat
    kk = randperm(size(X,1)); % shuffle
    cvResult  = zeros(16, totalCV);

    for cv = 1 : totalCV
        [ train_x,train_y,test_x,test_y ] = genFoldSets(X,target',kk,cv, totalCV);
        [Prior, PriorN, Cond, CondN] = MLKNN_train(train_x, train_y', Num, Smooth);
        [HammingLoss,RankingLoss, OneError, Coverage, Average_Precision, Outputs, Pre_Labels] = MLKNN_test(train_x, train_y', test_x, test_y', Num, Prior, PriorN, Cond, CondN); % Performing the test procedure

        %% Prediction and evaluation
    %     if tuneThreshold == 1
    %         fscore                 = (train_x*modelLSML.W)';
    %         [ tau,  currentResult] = TuneThreshold( fscore, train_y', 1, 2);
    %         Pre_Labels             = Predict(Outputs,tau);
    %     else
    %         Pre_Labels = double(Outputs>=0.5);
    %     end
    
        fprintf('-- Evaluation\n');
        ResultAll = EvaluationAll(Pre_Labels, Outputs, test_y');
        cvResult(:, cv) = cvResult(:, cv) + ResultAll;
        Avg_Result      = zeros(16,2);
        Avg_Result(:,1) = mean(cvResult,2);
        Avg_Result(:,2) = std(cvResult,1,2);
        PrintResults(Avg_Result);
    end
end
