function [ tau,  currentResult] = TuneThreshold( output, target, bAllOne, metricIndex)
% Tune the threshold for multi-label learning algorithms on the training
% data with one evaluation metric
    if nargin < 3
        bAllOne = 1;
    elseif nargin < 4
        metricIndex = 3; % F1 or label-based F1
    end
    fprintf('-- Tune threshold for multi-label classification\n');
    [num_class,num_train] = size(target);
    TotalNums = 50;
    %min_score = min(min(output));
    min_score = 0;
    max_score = max(max(output));
    step = (max_score - min_score)/TotalNums;
    tau_range = min_score:step:max_score;
    
    tau = zeros(1,num_class);
    currentResult = tau;
    for t = 1:length(tau_range)
        threshold = tau_range(t);
        if bAllOne == 1 % set to only one threshold for all the class labels
            thresholds = threshold*ones(size(output));
            predict_target = single( (output - thresholds) >= 0 );
            tempResult = evaluateOneMetric(target, predict_target, metricIndex);
            if tempResult > currentResult(1,1)
                currentResult(1,1) = tempResult;
                tau(1,1) = threshold;
            end
        else % set to one threshold for each label independently
            for l = 1:num_class
                thresholds = threshold*ones(1,num_train);
                predict_target_l = single( (output(l,:) - thresholds) >= 0 );
                tempResult = evaluateF1(target(l,:), predict_target_l);
                
                if tempResult > currentResult(1,l)
                    currentResult(1,l) = tempResult;
                    tau(1,l) = threshold;
                end
            end
        end
        
    end
    if bAllOne == 1
        tau = tau(1,1)*ones(1,num_class);
    end
end



function f1 = evaluateF1(target, predict)
% label-based f1 bor each label
    TP = target*predict';
    precision = TP/sum(predict~=0);
    recall = TP/sum(target~=0);
    f1 = 2*precision*recall/(precision + recall);
end

function  Result = evaluateOneMetric(target, predict_target, metric)
% predict_target
% target
%   
    Result = 0;
    if metric == 1
        HammingScore = 1 - Hamming_loss(predict_target,target);
        Result = HammingScore;
    elseif metric==2 || metric==3
        [ExampleBasedAccuracy,~,~,ExampleBasedFmeasure] = ExampleBasedMeasure(target,predict_target);
        if metric==2 
            Result = ExampleBasedAccuracy;
        else
            Result = ExampleBasedFmeasure;
        end
    elseif metric == 4 || metric == 5
        [LabelBasedAccuracy,~,~,LabelBasedFmeasure] = LabelBasedMeasure(target,predict_target);
        if metric==4 
            Result = LabelBasedAccuracy;
        else
            Result = LabelBasedFmeasure;
        end
    elseif metric == 6
        SubsetAccuracy = SubsetAccuracyEvaluation(target,predict_target);
        Result = SubsetAccuracy;
    end
end