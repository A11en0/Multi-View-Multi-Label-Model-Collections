
function Accuracy=MultiLabelAccuracyEvaluation(predict_target,test_target)
%% Evaluate the classification accuracy for multi label classification, alpha evaluation method
% Input:
%       predict_target: k X N vector, predicted result by classifier
%       test_target   £ºk X N vector, the ground truth label set
%
% Output:
%       Accuracy      : classifer's accuracy

    [num_label,num_test]=size(test_target);
    count=0;
    for i=1:num_test
%         numerator=sum(and(test_target(:,i),predict_target(:,i)));
%         denominator=sum(test_target(:,i))+sum(predict_target(:,i))-numerator;
%         count=count+numerator/denominator;

        numerator=test_target(:,i)'*predict_target(:,i);
        denominator=sum(or(test_target(:,i),predict_target(:,i)));
        if denominator~=0
            count=count + numerator/denominator;
        end
    end
    Accuracy=count/num_test;
end