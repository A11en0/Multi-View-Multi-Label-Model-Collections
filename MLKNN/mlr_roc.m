function [tpr1,fpr1] = mlr_roc(f, y_test)

% [tpr,fpr] = mlr_roc(f, y_test)
% Calculates TPR (True Positive Rate) and FPR (False Positive Rate) and
% plots ROC curve.
%%% INPUTS %%%
% f       : matrix of ranking function outputs
% y_test  : label matrix for test set
%%% OUTPUTS %%%
% tpr : true positive rate values
% fpr : false positive rate values

ind=find(y_test==0);
y_test(ind) = -1;
K=size(y_test,2);

for i=1:K
    [match(:,i),fpp(:,i),fnn(:,i)] = performance(y_test,f,i);
    tp1(i)=sum(match(:,i));
    fn1(i)=sum(fnn(:,i));
    fp1(i)=sum(fpp(:,i));
    tn1(i)=K*size(f,1)-(tp1(i)+fp1(i)+fn1(i));
    tpr1(i)=tp1(i)/(tp1(i)+fn1(i)); % Samples are considered all-together
    fpr1(i)=fp1(i)/(fp1(i)+tn1(i));
%     if (tp1(i)+fn1(i)==0)
%         tpr1(i) = 0;
%     else
%         tpr1(i)=tp1(i)/(tp1(i)+fn1(i)); % Samples are considered all-together
%     end
%     if (fp1(i)+tn1(i)==0)
%         fpr1(i) = 0;
%     else
%         fpr1(i)=fp1(i)/(fp1(i)+tn1(i));
%     end
end
