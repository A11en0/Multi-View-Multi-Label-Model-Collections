
function ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)
% evluation for MLC algorithms, there are fifteen evaluation metrics
% 
% syntax
%   ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   Pre_Labels          - L x num_test data matrix of predicted labels
%   Outputs             - L x num_test data matrix of scores
%
% output
%     ResultAll
%     ResultAll(1,1)=HammingLoss;
%     ResultAll(2,1)=ExampleBasedAccuracy; 
%     ResultAll(3,1)=ExampleBasedPrecision; 
%     ResultAll(4,1)=ExampleBasedRecall; 
%     ResultAll(5,1)=ExampleBasedFmeasure;
% 
%     ResultAll(6,1)=SubsetAccuracy;
%     ResultAll(7,1)=LabelBasedAccuracy; 
%     ResultAll(8,1)=LabelBasedPrecision;
%     ResultAll(9,1)=LabelBasedRecall; 
%     ResultAll(10,1)=LabelBasedFmeasure; 
% 
%     ResultAll(11,1)=MicroF1Measure;
%     ResultAll(12,1)=Average_Precision;
%     ResultAll(13,1)=OneError;
%     ResultAll(14,1)=RankingLoss;
%     ResultAll(15,1)=Coverage;
    
    ResultAll=zeros(16,1); 

    HammingLoss=Hamming_loss(Pre_Labels,test_target);

    [ExampleBasedAccuracy,ExampleBasedPrecision,ExampleBasedRecall,ExampleBasedFmeasure]=ExampleBasedMeasure(test_target,Pre_Labels);
    SubsetAccuracy=SubsetAccuracyEvaluation(test_target,Pre_Labels);
    [LabelBasedAccuracy,LabelBasedPrecision,LabelBasedRecall,LabelBasedFmeasure]=LabelBasedMeasure(test_target,Pre_Labels);
    
    RankingLoss=Ranking_loss(Outputs,test_target);
    OneError=One_error(Outputs,test_target);
    Coverage=coverage(Outputs,test_target);
    Average_Precision=Average_precision(Outputs,test_target);
    MicroF1Measure=MicroFMeasure(test_target,Pre_Labels);
        
    [tpr,fpr] = mlr_roc(Outputs', test_target');
    [~, AUC] = mlr_auc(fpr,tpr);
    ResultAll(1,1)=HammingLoss;
    ResultAll(2,1)=ExampleBasedAccuracy; 
    ResultAll(3,1)=ExampleBasedPrecision; 
    ResultAll(4,1)=ExampleBasedRecall; 
    ResultAll(5,1)=ExampleBasedFmeasure;

    ResultAll(6,1)=SubsetAccuracy;
    ResultAll(7,1)=LabelBasedAccuracy; 
    ResultAll(8,1)=LabelBasedPrecision;
    ResultAll(9,1)=LabelBasedRecall; 
    ResultAll(10,1)=LabelBasedFmeasure; 

    ResultAll(11,1)=MicroF1Measure;
    ResultAll(12,1)=Average_Precision;
    ResultAll(13,1)=OneError;
    ResultAll(14,1)=RankingLoss;
    ResultAll(15,1)=Coverage;
    ResultAll(16,1) = AUC;
end