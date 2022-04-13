function model = PAR_train(train_data,train_target,k,alpha)

%  PAR_train deals with paritial multi label learning problem via an iterative label propagation procedure, and the then classifie
% the unseen instance based on mininum error reconstruction from its labels.
% This function is the training phase of the algorithm.
%       model = PAR_train(train_data,train_target,k,alpha)
%
%    Description
%
%        PAR_train takes,
%           train_data                  - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target              - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_target(j,i) equals +1, otherwise train_target(j,i) equals 0
%           k                           - The number of nearest neighbors be considered     (defalut 10)
%           alpha                       - A balancing coefficient parameter which 0<alpha<1 (defalut 0.95)
%
%      and returns,
%           model is a structure continues following elements
%           model.kdtree                -A kdtree which is used to find k-nearest neighbors in the test phase
%           model.finalConfidence       -A QxM array,finalConfidence(j,i) is the final confidence of j th label of i th training instance
%           model.disambiguatedLabel    -A QxM array,disambiguatedLabel(j,i) equals +1 if j th class label if i th instance is the single label after disambiguating, otherwise disambiguatedLabel(j,i) equals 0
%           model.k                     - The number of nearest neighbors be considered
if nargin<4
    alpha = 0.95;
end
if nargin<3
    k=10;
end
if nargin<2
    error('Not enough input parameters, please check again.');
end
if size(train_data,1)~=size(train_target,2)
    error('Length of label vector does match the number of instances');
end


[label_num,ins_num] = size(train_target);
fea_num = size(train_data,2);
%kdtree = KDTreeSearcher(train_data);
ins_num = size(train_data,1);
%[neighbor,dist] = knnsearch(kdtree,train_data,'k',k+1);
[dist,neighbor] = pdist2(train_data,train_data,'euclidean','Smallest',k);
dist=dist';
neighbor=neighbor';
% neighbor = neighbor(:,2:k+1);
% dist = dist(:,2:k+1);
P = train_target';

  % P = P./repmat(sum(P,2),1,label_num);
trans = zeros(ins_num);
rows = repmat((1:ins_num)',1,k);
datas = zeros(ins_num,k);
for i=1:ins_num
    neighborIns = train_data(neighbor(i,:),:)';
    w = lsqnonneg(neighborIns,train_data(i,:)');
    datas(i,:) = w;
end
trans = sparse(rows,neighbor,datas,ins_num,ins_num);

sumW = full(sum(trans,2));
sumW(sumW==0)=1;
trans = bsxfun(@rdivide,trans,sumW);
p0 = P;
iterVal = zeros(1,1000);
for iter=1:1000
    tmp= P;
    P = alpha*trans*P+(1-alpha)*p0;                          %propagate Y<-T
    P = P.*train_target';                                %row-normalize Y'
    P = P./repmat(sum(P,2),1,label_num);
    diff=norm(full(tmp)-full(P),2);
    iterVal(iter) = abs(diff);
    if abs(diff)<0.00001
        break
    end
end

labelSum = sum(p0);
predSum = sum(P);
poster = labelSum./predSum;
P = P.*repmat(poster,ins_num,1);
predLabel = zeros(ins_num,label_num);

for i=1:ins_num
    P(i,:)=(P(i,:)-min(min(P(i,:))))/(max(max(P(i,:)))-min(min(P(i,:))));
end

for i=1:ins_num
    [val,idx] = max(P(i,:));
    predLabel(i,idx)=1;
end
%model.kdtree = kdtree;
model.finalConfidence = P';
%     model.iterVal = iterVal;
model.disambiguatedLabel = predLabel';
model.type = 'PAR';
model.fea_num = fea_num;
model.k = k;
end

