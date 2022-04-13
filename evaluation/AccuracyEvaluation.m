
function Accuracy=AccuracyEvaluation(predict_target,test_target)
%% Evaluate the classification accuracy for single-label learning
% Input:
%       predict_target: 1 X N vector, predicted result by classifier
%       test_target   £º1 X N vector, the ground truth label set
%
% Output:
%       Accuracy      : classifer's accuracy

    num_test=size(test_target,2);
    correctones=(predict_target==test_target);
    Accuracy=sum(correctones)/num_test;

end