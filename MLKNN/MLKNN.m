function out = MLKNN(dataTrain, dataTrainLabels, dataTest, dataTestLabels)

Num = 10;
Smooth = 1;

dataTrainLabels(dataTrainLabels == 0) = -1;
dataTestLabels(dataTestLabels == 0) = -1;

[Prior, PriorN, Cond, CondN] = MLKNN_train(dataTrain, dataTrainLabels', Num, Smooth);
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision, Outputs,Pre_Labels] = MLKNN_test(dataTrain, dataTrainLabels', dataTest, dataTestLabels', Num, Prior, PriorN, Cond, CondN); % Performing the test procedure

dataTestLabels(dataTestLabels == -1) = 0;
Pre_Labels(Pre_Labels == -1) = 0;
accuracy = Accuracy(Pre_Labels, dataTestLabels');

% accuracy = PrecisionRecall(Pre_Labels, dataTestLabels');

out = {accuracy, HammingLoss, RankingLoss, OneError, Coverage, Average_Precision};

end