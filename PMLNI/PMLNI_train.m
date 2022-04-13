function model = PMLNI_train(train_data, train_target, true_target, opt)

% This function is the training phase of the PML-NI algorithm. 
%
%    Syntax
%
%      model = PMLNI_train(train_data, train_target, true_target, opt)
%
%    Description
%
%       PARM_train takes,
%           train_data                - A NxD array, the instance of the i-th PML example is stored in train_data(i,:)
%           train_target              - A NxQ array, if the jth class label is one of the candidate labels for the i-th PML example, then train_target(i,j) equals 1, otherwise train_target(i,j) equals 0
%           opt.lambda                - The balancing parameter 
%           opt.beta                  - The regularization parameter 
%           opt.gamma                 - The regularization parameter 
%           opt.max_iter              - The maximum iterations
%       
%       and returns,
%           model                     - The learned model

lambda = opt.lambda;
beta = opt.beta;
gamma = opt.gamma;
max_iter = opt.max_iter;

model = [];

[num_train,dim]=size(train_data);
[~,num_label]=size(train_target);

%% Training
fea_matrix = [train_data, ones(num_train,1)];
V = zeros(num_label,dim+1);
U = zeros(num_label,dim+1);
Y = zeros(num_label,dim+1);
mu = 1e-4;
rho = 1.1;

YX = train_target'*fea_matrix;
XX = fea_matrix'*fea_matrix;
    
for t = 1:max_iter
    
    % Update W
    W = (YX+mu*U+mu*V+Y)/(XX+(lambda+mu)*eye(dim+1));
    
    % Update U & V
    Uk=U;
    Vk=V;
    
    [M,S,Nhat] = svd(W-V-Y/mu,'econ');
    sp = diag(S);
    svp = length(find(sp>beta/mu));
    if svp>=1
        sp = sp(1:svp)-beta/mu;
    else
        svp=1;
        sp=0;
    end
    Uhat =  M(:,1:svp)*diag(sp)*Nhat(:,1:svp)' ;
    U = Uhat;
    
    % L1 norm
    Vraw = W-U-Y/mu;
    Vhat = max(Vraw - gamma/mu,0)+min(Vraw+gamma/mu,0);
%     L2,1 norm
%     Vraw = W-U-Y/mu;
%     Vhat = zeros(size(Vraw));
%     for j = 1:num_label
%         v = Vraw(j,:);
%         vNorm = norm(v);
%         if vNorm>gamma/mu
%              Vhat(j,:)= (vNorm-gamma/mu)/vNorm*v;
%         end
%     end
    
    V = Vhat;
  
    % Stop Criterion
    convg2 = false;
    stopCriterion2 = mu*norm(U-Uk,'fro')/norm(W,'fro');
    if stopCriterion2<1e-5
        convg2=true;
    end
    convg1 = false;
    tmp = W-U-V;
    stopCriterion1 = norm(tmp,'fro')/norm(W,'fro');
    if stopCriterion1<1e-7
        convg1 = true;
    end
    
    if convg2
        mu = min(rho*mu,1e10);
    end
    Y = Y+mu*(U+V-W);
    
    if convg1 && convg2
        break;
    end
end


%% Computing the size predictor using linear least squares model
Outputs = fea_matrix*U';
Left=Outputs;
Right=zeros(num_train,1);
for i=1:num_train
    temp=Left(i,:);
    [temp,index]=sort(temp);
    candidate=zeros(1,num_label+1);
    candidate(1,1)=temp(1)-0.1;
    for j=1:num_label-1
        candidate(1,j+1)=(temp(j)+temp(j+1))/2;
    end
    candidate(1,num_label+1)=temp(num_label)+0.1;
    miss_class=zeros(1,num_label+1);
    for j=1:num_label+1
        temp_notlabels=index(1:j-1);
        temp_labels=index(j:num_label);
        [~,false_neg]=size(setdiff(temp_notlabels,find(true_target(i,:)==0)));
        [~,false_pos]=size(setdiff(temp_labels,find(true_target(i,:)==1)));
        miss_class(1,j)=false_neg+false_pos;
    end
    [~,temp_index]=min(miss_class);
    Right(i,1)=candidate(1,temp_index);
end
Left=[Left,ones(num_train,1)];
tempvalue=(Left\Right)';
Weights_sizepre=tempvalue(1:num_label);
Bias_sizepre=tempvalue(num_label+1);

model.W = U;
model.Weights_sizepre=Weights_sizepre;
model.Bias_sizepre=Bias_sizepre;

end

