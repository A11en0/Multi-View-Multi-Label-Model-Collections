function [Pre_Labels, Outputs] = PAR_VLS(train_data,train_target,pre_target,test_data,test_target,str)
%
%     train_data     - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%     train_target   - Labels after disambiguation¡£A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%     pre_target     -Labels without disambiguation¡£A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%     test_data       - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%     test_target     - A QxM2 array, if the jth class label is the ground-truth label for the ith test instance, then test_target(j,i) equals 1; otherwise test_target(j,i) equals 0
%

[~,num_training]=size(train_target);
[num_class,num_testing]=size(test_target);
CLR={};
bar_target=~pre_target;
% train q(q-1)/2 classifier
for j0=1:num_class
    for k0=1:num_class
        if j0<k0
            pos_id=[];
            neg_id=[];
            t_id=[];
            for i0=1:num_training
                if (train_target(j0,i0)==1&&bar_target(k0,i0)==1)
                    pos_id=[pos_id,i0];
                elseif (train_target(k0,i0)==1&&bar_target(j0,i0)==1)
                    neg_id=[neg_id,i0];
                end
            end
            CLR{j0,k0}{1}=pos_id;
            CLR{j0,k0}{2}=neg_id;
            t_id=[pos_id,neg_id];
            CLR{j0,k0}{3}=t_id;
            CLR{j0,k0}{4}=length(t_id)/num_training;
            training_instance_matrix=train_data(t_id,:);
            training_label_vector=[ones(length(pos_id),1);zeros(length(neg_id),1)];
            if length(pos_id)==0||length(neg_id)==0
                CLR{j0,k0}{5}=[];
            else
                baseModel = libsvmtrain(training_label_vector, training_instance_matrix , str);
                CLR{j0,k0}{5}=baseModel;
            end
        end
    end
end
%train q virtual label classifier
for j0=1:num_class
    pos_id=[];
    neg_id=[];
    t_id=[];
    for i0=1:num_training
        if (train_target(j0,i0)==1)
            pos_id=[pos_id,i0];
        elseif(bar_target(j0,i0)==1)
            neg_id=[neg_id,i0];
        end
    end
    CLR{j0,num_class+1}{1}=pos_id;
    CLR{j0,num_class+1}{2}=neg_id;
    t_id=[pos_id,neg_id];
    CLR{j0,num_class+1}{3}=t_id;
    CLR{j0,num_class+1}{4}=length(t_id)/num_training;
    training_instance_matrix=train_data(t_id,:);
    training_label_vector=[ones(length(pos_id),1);zeros(length(neg_id),1)];
    if length(pos_id)==0||length(neg_id)==0
        CLR{j0,num_class+1}{5}=[];
    else
        baseModel = libsvmtrain(training_label_vector, training_instance_matrix , str);
        CLR{j0,num_class+1}{5}=baseModel;
    end
end
%calculate q(q-1)/2 label classifier accuracy
for j0=1:num_class
    for k0=1:num_class
        if j0<k0
            baseModel=CLR{j0,k0}{5};
            h_test_data=train_data(CLR{j0,k0}{3},:);
            h_testing_label_vector=train_target(j0,CLR{j0,k0}{3});
            h_testing_label_vector=h_testing_label_vector';
            testing_label_vector=ones(num_testing,1);
            if isempty(CLR{j0,k0}{5})
                CLR{j0,k0}{8}=0;
            else
                [predicted_label, accuracy, decision_values] = libsvmpredict(h_testing_label_vector, h_test_data, baseModel);
                CLR{j0,k0}{8}=accuracy(1)/100;
            end
        end
    end
end
%calculate q virtual label classifier accuracy
for j0=1:num_class
    baseModel=CLR{j0,num_class+1}{5};
    h_test_data=train_data(CLR{j0,num_class+1}{3},:);
    h_testing_label_vector=train_target(j0,CLR{j0,num_class+1}{3});
    h_testing_label_vector=h_testing_label_vector';
    testing_label_vector=ones(num_testing,1);
    if isempty(CLR{j0,num_class+1}{5})
        CLR{j0,num_class+1}{8}=0;
    else
        [predicted_label, accuracy, decision_values] = libsvmpredict(h_testing_label_vector, h_test_data, baseModel);
        CLR{j0,num_class+1}{8}=accuracy(1)/100;
    end
end

%predict q(q-1)/2 label classifier result
for j0=1:num_class
    for k0=1:num_class
        if j0<k0
            baseModel=CLR{j0,k0}{5};
            testing_label_vector=ones(num_testing,1);
            if isempty(CLR{j0,k0}{5})
                CLR{j0,k0}{6}=testing_label_vector;
            else
                [predicted_label, accuracy, decision_values] = libsvmpredict(testing_label_vector, test_data, baseModel);
                CLR{j0,k0}{6}=predicted_label;
                CLR{j0,k0}{7}=decision_values;
            end
        end
    end
end
%predict q viatual label classifier result
for j0=1:num_class
    baseModel=CLR{j0,num_class+1}{5};
    testing_label_vector=ones(num_testing,1);
    if isempty(CLR{j0,num_class+1}{5})
        CLR{j0,num_class+1}{6}=testing_label_vector;
    else
        [predicted_label, accuracy, decision_values] = libsvmpredict(testing_label_vector, test_data, baseModel);
        CLR{j0,num_class+1}{6}=predicted_label;
        CLR{j0,num_class+1}{7}=decision_values;
    end
end

lab=zeros(size(test_target));
vlab=zeros(size(test_target));

for j0=1:num_class
    for k0=1:num_class
        if ~isempty(CLR{j0,k0})
            lab(j0,:)=lab(j0,:)+CLR{j0,k0}{8}*CLR{j0,k0}{6}';
        end
        if ~isempty(CLR{k0,j0})
            lab(j0,:)=lab(j0,:)+CLR{k0,j0}{8}*(~CLR{k0,j0}{6}');
        end
    end
end
for j0=1:num_class
    lab(j0,:)=lab(j0,:)+CLR{j0,num_class+1}{8}*CLR{j0,num_class+1}{6}';
end
for j0=1:num_class
    vlab(1,:)=vlab(1,:)+CLR{j0,num_class+1}{8}*(~CLR{j0,num_class+1}{6}');
end
vlab=repmat(vlab(1,:),num_class,1);
lab=ceil(lab);
vlab=ceil(vlab);
Pre_Labels=lab>vlab;
outputValue=lab';
for i=1:num_testing
    outputValue(i,:)=(outputValue(i,:)-min(min(outputValue(i,:))))/(max(max(outputValue(i,:)))-min(min(outputValue(i,:))));
end
Outputs=outputValue';
outputValue=outputValue'; 
end

