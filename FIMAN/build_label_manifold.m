function [alpha,Confidences]=build_label_manifold(d,data,train_p_target,k,r,max_iter)
%build_label_manifold
%
% Syntax
%
%       [alpha,Confidences]=build_label_manifold(d,data,train_p_target,k,r,max_iter)
%
% Description
%
%       build_label_manifold takes,
%           data                - An Vx1 cell, the characteristics of the i-th view are stored in data{i,1}
%           train_p_target      - A MxQ array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(i,j) equals +1, otherwise train_p_target(i,j) equals 0
%           d                   - Dimensions of low-dimensional space
%           k                   - the number of the neighboors
%           r                   - the number of views
%           max_iter            - the number of iterations
%          
%       and returns,
%           Confidences         - An MxQ array, Confidences (i,j) denotes the confidence of the jth class label as its true label in the ith instance
%           alpha               - the weight vector for each view

vnum=size(data,1);
[p,q]=size(train_p_target);
Weight=cell(vnum,1);
Laplacian=cell(vnum,1);
W_t=zeros(p,p);
for i=1:vnum
    train_data = data{i,1};
    kdtree = KDTreeSearcher(train_data);
    [neighbor,~] = knnsearch(kdtree,train_data,'k',k+1);
    neighbor = neighbor(:,2:k+1);
    
    W = zeros(p,k);
    rows = repmat((1:p)',1,k);
  
    for ii=1:p
            neighborIns = train_data(neighbor(ii,:),:)';
            w = lsqnonneg(neighborIns,train_data(ii,:)');
            W(ii,:) = w';          
    end    
    Ww= sparse(rows,neighbor,W,p,p);  
    L=cal_Laplacian(Ww,p);
    
    Laplacian{i,1}=L;
    Weight{i,1}=Ww;
   
end
alpha=get_view_weight(d,p,Laplacian, vnum ,r,max_iter);
for i=1:vnum
    W_t = W_t+Weight{i,1}*alpha(i);
end
    
fprintf('\n')
fprintf('Generate the labeling confidence...\n');
M = sparse(p,p);
fprintf('Obtain Hessian matrix...\n');
WT = W_t';
T =WT*W_t+ W_t*ones(p,p)*WT.*eye(p,p)-2*WT;
T1 = repmat({T},1,q);
M = spblkdiag1(T1{:});
lb=sparse(p*q,1);
ub=reshape(train_p_target,p*q,1);
II = sparse(eye(p));
A = repmat(II,1,q);
b=ones(p,1);
M = (M+M');
fprintf('quadprog...\n');
options = optimoptions('quadprog',...
'Display', 'iter','Algorithm','interior-point-convex' );
Outputs= quadprog(M, [], [],[], A, b, lb, ub,[], options);
Confidences=reshape(Outputs,p,q);
end
function L=cal_Laplacian(W,n) 
    D = zeros(n,n);
    W=full(W);
    W(1:n+1:(n^2)) = 0;
    D(1:n+1:(n^2)) = sum(W,2);
    temp =  eye(n) - D^(-1/2)*W*D^(-1/2);
    if(numel(find(isnan(temp)))>0)
        L=D-W;
    else
        L=temp;
    end
end