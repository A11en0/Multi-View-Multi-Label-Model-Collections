function model = MVPL(trainData,trainTarget,k,alpha)

% IPAL deals with paritial label learning problem via an iterative label propagation procedure, and the then classifie
% the unseen instance based on mininum error reconstruction from its labels. 
% This function is the training phase of the algorithm. 
%       model = IPAL_train( trainData,trainTarget,k,alpha )
%
%    Description
%
%       IPAL_train takes,
%           train_data                  - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target              - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           k                           - The number of nearest neighbors be considered     (defalut 10)
%           alpha                       - A balancing coefficient parameter which 0<alpha<1 (defalut 0.95)
% 
%      and returns,
%           model is a structure continues following elements
%           model.kdtree                -A kdtree which is used to find k-nearest neighbors in the test phase
%           model.finalConfidence       -A QxM array,finalConfidence(j,i) is the final confidence of j th label of i th training instance
%           model.disambiguatedLabel    -A QxM array,disambiguatedLabel(j,i) equals +1 if j th class label if i th instance is the single label after disambiguating, otherwise disambiguatedLabel(j,i) equals 0
%           model.k                     - The number of nearest neighbors be considered
% M.-L. Zhang, F. Yu. Solving the partial label learning problem: An instance-based approach. In: Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI'15), Buenos Aires, Argentina, 2015, 4048-4054% 
  if nargin<4
    alpha = 0.97;
end
if nargin<3
    k=10;
end
if nargin<2
    error('Not enough input parameters, please check again.');
end
if size(trainData{1},1)~=size(trainTarget,2)
    error('Length of label vector does match the number of instances');
end

%rewrite the process of W generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    view_num = size(trainData,1);
    [label_num,ins_num] = size(trainTarget);
    trans = sparse([],[],[],ins_num,ins_num);
    parfor v = 1:view_num
        disp(['The ',num2str(v),'-th view is start!'])
        train_data = zscore(trainData{v});
        fea_num = size(train_data,2);
        kdtree = KDTreeSearcher(train_data);
        [neighbor,dist] = knnsearch(kdtree,train_data,'k',k+1);
        neighbor = neighbor(:,2:k+1);
        dist = dist(:,2:k+1);
        rows = repmat((1:ins_num)',1,k);
        datas = zeros(ins_num,k);
        for i = 1:ins_num
            neighborIns = train_data(neighbor(i,:),:);
            w = exp(-pdist2(neighborIns,train_data(i,:),'euclidean'));
            datas(i,:) = w;
        end
        trans = trans+sparse(rows,neighbor,datas,ins_num,ins_num);
        disp(['The ',num2str(v),'-th view is finished!']);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %trans = (trans+trans')/2;
    y = trainTarget';
    y = y./repmat(sum(y,2),1,label_num);

    sumW = full(sum(trans,2));
    sumW(sumW==0)=1;
    trans = bsxfun(@rdivide,trans,sumW);
    model.W = trans;
   
   
    y0 = y;
    iterVal = zeros(1,100);
    for iter=1:30
        tmp= y;
        y = alpha*trans*y+(1-alpha)*y0;                          
        y = y.*trainTarget';                                
        y = y./repmat(sum(y,2),1,label_num);
        diff=norm(full(tmp)-full(y),2);
        iterVal(iter) = abs(diff);
        if abs(diff)<0.0001
            break
        end
    end

    
    labelSum = sum(y0);
    predSum = sum(y);
    poster = labelSum./predSum;
    y = y.*repmat(poster,ins_num,1);
        predLabel = zeros(ins_num,label_num);
    for i=1:ins_num
        [val,idx] = max(y(i,:));
        predLabel(i,idx)=1;
    end
    model.finalConfidence = y';
    model.disambiguatedLabel = predLabel';
    model.type = 'MVPL';
    model.k = k;
   
end

