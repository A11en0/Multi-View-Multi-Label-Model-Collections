
function recall=Recall(Pre_Labels,test_target)
%% Computing the Precision for multi label learning algorithm
%Pre_Labels: the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(j,i)=1, otherwise Pre_Labels(j,i)=-1
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance]=size(Pre_Labels);
    total=0;
    for i=1:num_instance
        numerator=Pre_Labels(:,i)'*test_target(:,i);
        denominator=test_target(:,i)'*test_target(:,i);
        if denominator ~=0
            total=total+numerator/denominator;
        end
    end
    recall=total/num_instance;
    