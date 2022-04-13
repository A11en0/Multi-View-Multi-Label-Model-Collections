
function F_Measure=F_MeasureEvaluation(predict_target,test_target)
%% Evaluate the F_Measure for multi label classification result
% Input:
%       predict_target: k X N vector, predicted result by classifier
%       test_target   £ºk X N vector, the ground truth label set
%
% Output:
%       Accuracy      : classifer's accuracy

    [num_label,num_test]=size(test_target);
    count=0;
    for i=1:num_test
         numerator=sum(and(test_target(:,i),predict_target(:,i)));
        denominator=sum(test_target(:,i))+sum(predict_target(:,i));
        count=count+2*numerator/denominator;
    end
    F_Measure=count/num_test;
end