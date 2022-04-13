
function [Average_Precision,OneError,RankingLoss,Coverage]=RankingBasedMeasure(Outputs,test_target)
% syntax
%   [Average_Precision,OneError,RankingLoss,Coverage]=RankingBasedMeasure(Outputs,test_targets)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   Outputs             - L x num_test data matrix of predicted probability
%
% output
%   [Average_Precision,OneError,RankingLoss,Coverage]=RankingBasedMeasure(Outputs,test_targets)


    RankingLoss=Ranking_loss(Outputs,test_target);
    OneError=One_error(Outputs,test_target);
    Coverage=coverage(Outputs,test_target);
    %Coverage=Coverage/num_class;
    Average_Precision=Average_precision(Outputs,test_target);

%     HammingLoss=Hamming_loss((Pre_Labels-0.5)*2,(test_target-0.5)*2);
%     RankingLoss=Ranking_loss(Outputs,(test_target-0.5)*2);
%     OneError=One_error(Outputs,(test_target-0.5)*2);
%     Coverage=coverage(Outputs,(test_target-0.5)*2);
%     Coverage=Coverage/num_class;
%     Average_Precision=Average_precision(Outputs,(test_target-0.5)*2);
end