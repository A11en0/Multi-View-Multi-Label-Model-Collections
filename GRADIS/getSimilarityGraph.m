function W = getSimilarityGraph(trainData,pos_index,K)

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



if nargin<2
    K=10;
end


k = min(K,size(pos_index,1)-1);

%rewrite the process of W generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    view_num = size(trainData,1);
    pos_data = cell(view_num,1);
    for i = 1:view_num
        temp = trainData{i};
        pos_data{i} = temp(pos_index,:);
    end

    ins_num = size(pos_data{1},1);
    disp(ins_num);
    trans = sparse([],[],[],ins_num,ins_num);
    parfor v = 1:view_num
        train_data = zscore(pos_data{v});
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
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    W = trans;
end

