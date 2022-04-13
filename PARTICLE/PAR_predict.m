function lab = PAR_predict(train_data,test_data,test_p_target,model,o)
if nargin<3
    error('Not enough input parameters, please check again.');
end

if model.type~='PAR'
    error('The input model does not match the prediction model')
end
fea_num = size(test_data,2);
if fea_num ~= model.fea_num
    error('feature size of test data does not match the feature size of training data');
end

if size(test_data,1)~=size(test_p_target,2)
    error('Length of label vector does match the number of instances');
end

k = model.k;
[testDist,testNeighbor] = pdist2(train_data,test_data,'euclidean','Smallest',k);
testDist=testDist';
testNeighbor=testNeighbor';
y = model.finalConfidence;

[label_num,test_num] = size(test_p_target);
predictLabel = zeros(1,test_num);
outputValue = zeros(label_num,test_num);
for test=1:size(test_p_target,2)
    sumDis = sum(testDist(test,:));
    label = zeros(1,label_num);
    for near=1:k
        %     label = label+y(:,testNeighbor(test,near))';
        label = label+(1-testDist(test,near)/sumDis)*y(:,testNeighbor(test,near))';%/sum(train_p_target(:,dis(near,2)));
        %   label = label + y(:,testNeighbor(test,near))'/sum(y(:,testDist(near,1)));
    end
    %1-dis(test,near)/sumDis) 
    [~,idx]=max(label);
    predictLabel(test) = idx;
    outputValue(:,test) = label';
end
[~,real] = max(full(test_p_target));
accuracy = sum(predictLabel==real)/size(test_p_target,2);
LabelMat = repmat((1:label_num)',1,test_num);
predictLabel = repmat(predictLabel,label_num,1)==LabelMat;

outputValue=outputValue';
for i=1:test_num
    outputValue(i,:)=(outputValue(i,:)-min(min(outputValue(i,:))))/(max(max(outputValue(i,:)))-min(min(outputValue(i,:))));
end
outputValue=outputValue';
lab=zeros(label_num,test_num);
for i=1:test_num
    for j=1:label_num
        if outputValue(j,i)>=o
            lab(j,i)=1;
        else
            lab(j,i)=0;
        end
    end
end

 mm=lab+test_p_target;
 mm(mm<2)=0;
 mm=mm/2;
 mmm=sum(mm);
 for i=1:test_num
     if mmm(i)==0
         mm(:,i)=lab(:,i);
     end
 end
 lab=mm;
end

