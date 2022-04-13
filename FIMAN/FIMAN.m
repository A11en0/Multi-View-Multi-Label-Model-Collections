function FIMAN(data,target,noise_target,idx,p,rn,thr1,thr2,eta,k,n_fold)
% FIMAN [1] deals with the multi-view partial multi-label problem.
% Syntax
%
%       FIMAN(data,target,noise_target,idx,p,rn,name,thr1,thr2,eta,k)
%
% Description
%
%       FIMAN takes,
%           data                - An Vx1 cell, the characteristics of the i-th view are stored in data{i,1}
%           noise_target        - A MxQ array, if the jth class label is one of the partial labels for the ith training instance, then noise_target(i,j) equals +1, otherwise noise_target(i,j) equals 0
%           eta                 - the regularization parameter for ridge regression
%           k                   - the number of the neighboors
%           thr1                - the disambiguation threshold
%           thr2                - the prediction threshold
%           rn                  - the number of false positive labels
%           p                   - the fraction of examples which are partially labeled in the data set
%      
%       and obtains,
%           Pre_cell            - the predicted label set
%           Output_cell         - the labeling confidence
%          
%
%  [1]Jing-Han Wu, Xuan Wu, Qing-Guo Chen, Yao Hu, Min-Ling Zhang. Feature-Induced Manifold Disambiguation for Multi-View Partial Multi-label Learning. (KDD'20), San Diego, California USA, 2020, in press.
%

    v_num=size(data,1);

    for i=1:v_num
        [n_sample,n_fea]= size(data{i,1});
        for j=1:n_fea
            data{i,1}(:,j)=(data{i,1}(:,j)-min(min(data{i,1}(:,j))))/((max(max(data{i,1}(:,j))))-min(min(data{i,1}(:,j))));
        end
    end

    r=v_num;
    max_iter=5;
    demention=zeros(v_num,1);
    data_con=[];
    for i=1:v_num
        demention(i)=size(data{i,1},2);
        data_con=[data_con,data{i,1}];
    end
    d=min(demention);

    Output_cell=cell(n_fold,1);
    Pre_cell=cell(n_fold,1);
    Test_cell=cell(n_fold,1);
    C_cell=cell(n_fold,1);
    for i=1:n_fold
        fprintf('fold %d \n', i);
        temp_data = data;
        weak_target = noise_target;
        test_idx = idx{i, 1};
        test_data=data_con(test_idx,:);
        temp=target;
        test_target = temp(test_idx, : );
        for j=1:v_num
            temp_data{j,1}(test_idx,:) = [];
        end
        train_data = temp_data;
        weak_target(test_idx, :) =[];
        train_target = weak_target;
        
        merge_train_data=[];
        for kk=1:v_num
            merge_train_data=[merge_train_data,train_data{kk,1}];
        end
        
        fprintf('staring training & testing \n');
        [alpha,Confidences]=build_label_manifold(d,train_data,train_target,k,r,max_iter);
        
        C_cell{i,1}=Confidences';
         
        Pre_1=mapminmax(Confidences,0,1);
        Pre_1(Pre_1>thr1)=1;
        Pre_1(Pre_1<=thr1)=0;
        A = merge_train_data'*merge_train_data+eta*eye(size(merge_train_data,2));
        A(isnan(A)) = 0;
        W=pinv(A)*merge_train_data'*Pre_1;
        
        Outputs = test_data * W;
        [m, q] = size(test_target);
        Pre_Labels = Outputs;
        
        Pre_Labels=mapminmax(Pre_Labels,0,1);
        Pre_Labels(Pre_Labels>thr2)=1;
        Pre_Labels(Pre_Labels<=thr2)=0;
    
        Output_cell{i,1}=Outputs';
        Pre_cell{i,1}=Pre_Labels';
        Test_cell{i,1}=test_target';  
       
    end
    fprintf('-- Evaluation\n');
    cvResult  = zeros(16, n_fold);
    for cv = 1:n_fold
        ResultAll = EvaluationAll(Pre_cell{cv, 1}, Output_cell{cv, 1}, Test_cell{cv, 1});
        cvResult(:, cv) = cvResult(:, cv) + ResultAll;
        Avg_Result      = zeros(16,2);
        Avg_Result(:,1) = mean(cvResult,2);
        Avg_Result(:,2) = std(cvResult,1,2);
    end
    PrintResults(Avg_Result);
save(['mvpml_r',num2str(rn),'p',num2str(p),'thr1_',num2str(thr1),'thr2_',num2str(thr2),'k_',num2str(k),'.mat' ], 'Output_cell', 'Pre_cell', 'Test_cell','C_cell');
end
