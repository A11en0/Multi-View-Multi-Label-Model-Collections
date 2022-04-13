function [match,fp,fn] = performance(y,f,T)
% [match,fp,fn] = performance(y,f,T)
% finds the number of matches, fp and fn for the top T ranking values.
%%% INPUTS %%%
% y = labels
% f = ranking function outputs
% T = number of predicted labels
%%% OUTPUTS %%%
% match : is a vector which contains the number of hits for each instance in the test data set
% fp    : vector containing number of FPs (false positives) for each instance
% fn    : vector containing number of FNs (false negatives) for each instance


n=size(f,1);
K=size(f,2);

match=zeros(n,1);
fn=zeros(n,1);
fp=zeros(n,1);
for i=1:n
    clear pred_labels;
    [so, si]=sort(f(i,:),'descend');
    words=y(i,:);
    correct_labels=find(words>-1);
    if isempty(correct_labels)==0
        si=si(1:T);
        match(i)=0;
        for j=1:max(size(correct_labels))
            if(find(si==correct_labels(j)))
                match(i)=match(i)+1;
            end
        end
        fn(i)=max(size(correct_labels))-match(i);
        fp(i)=T-match(i);
    end
end