
function SubsetAccuracy=SubsetAccuracyEvaluation(test_target,predict_target)
%% Evaluate the subset classification accuracy for multi label classification
%  this evaluation have the same meaning of subset 0/1 loss, and the value
%  of subset 0/1 loss = 1 - subsetAccuracy
%
% syntax
%   SubsetAccuracy=SubsetAccuracyEvaluation(predict_target,test_target)
% input
%   test_target        - L x num_test data matrix of groundtruth labels
%   predict_target     - L x num_test data matrix of predicted labels
%
% Output:
%       Accuracy      : classifer's accuracy

    [~,num_test]=size(test_target);
    count=0;
    for i=1:num_test
        if isequal(test_target(:,i),predict_target(:,i))==1
            count=count+1;
        end
    end
    SubsetAccuracy=count/num_test;
end