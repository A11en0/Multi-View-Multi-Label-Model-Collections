function [Pre_Labels,Outputs] = testK(MV_train_data,train_target,MV_test_data, test_target, threshold, rate)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    view_num = size(MV_train_data,1);
    mv_train_data = cell(view_num,1);
    mv_test_data = cell(view_num,1);
    train_ins_num = size(MV_train_data{1});
    test_ins_num = size(MV_test_data{1});
    for i = 1:view_num
        temp = zscore([MV_train_data{i};MV_test_data{i}]);
        mv_train_data{i} = temp(1:train_ins_num,:);
        mv_test_data{i} = temp(train_ins_num+1:train_ins_num+test_ins_num,:);
    end
    model = MVPL(mv_train_data,train_target');
    disp('The first stage is finished!');
    W = model.W;

    final_target = mapminmax(model.finalConfidence',0,1);
    final_target(final_target>threshold) = 1;
    final_target(final_target<=threshold) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    view_num = size(mv_train_data,1);
    [train_data_num,label_num] = size(train_target);
    test_data_num = size(mv_test_data{1},1);
    Outputs = zeros(test_data_num,label_num);
    Pre_Labels = zeros(test_data_num,label_num);
    new_train_data = [];
    new_test_data = [];
    for i = 1:view_num
        new_train_data = [new_train_data,mv_train_data{i}];
        new_test_data = [new_test_data,mv_test_data{i}];
    end
    new_train_data = zscore(new_train_data);
    new_test_data = zscore(new_test_data);
   
    for i=1:label_num
        if sum(final_target(:,i))~=0
            pos_index = find(final_target(:,i)==1);
            neg_index = find(final_target(:,i)==0);
            k = ceil(min(size(pos_index,1),size(neg_index,1))*rate);
            %pos_L = W(pos_index,pos_index);
            %neg_L = W(neg_index,neg_index);
            pos_W = getSimilarityGraph(mv_train_data,pos_index,10);
            neg_W = getSimilarityGraph(mv_train_data,neg_index,10);
            p_cluster_index = SpectralClustering(pos_W,k);
            n_cluster_index = SpectralClustering(neg_W,k);
            view_train_data = [];
            view_test_data = [];
            for j = 1:view_num
                temp_train_data = mv_train_data{j};
                temp_test_data = mv_test_data{j};
                pdata = temp_train_data(pos_index,:);
                ndata = temp_train_data(neg_index,:);
                centers = clusterCenter(pdata,p_cluster_index,ndata,n_cluster_index);
                view_train_data = [view_train_data,generateSpecificFeatures(temp_train_data,centers)];
                view_test_data = [view_test_data,generateSpecificFeatures(temp_test_data,centers)];
            end
            temp = zscore([view_train_data;view_test_data]);
            clf = svmtrain(final_target(:,i),temp(1:train_data_num,:),'-t 0 -b 1 -q');
            [predicted_label,accuracy,prob_estimates]=svmpredict(test_target(:,i),temp(train_data_num+1:train_data_num+test_data_num,:),clf,'-b 1');
            svm_pos_index=find(clf.Label==1);
            Prob_pos = prob_estimates(:,svm_pos_index);
            Outputs(:,i) = Prob_pos;
            Pre_Labels(:,i) = predicted_label';
        end
    end
end

