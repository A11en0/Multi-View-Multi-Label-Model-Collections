function [Pre_Labels, Outputs] = PAR_MAP(train_data,train_target,pre_target,test_data,test_target,str)
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
lab=zeros(size(test_target));
vlab=zeros(size(test_target));
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
                baseModel = svmtrain(training_label_vector, training_instance_matrix , str);
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
        baseModel = svmtrain(training_label_vector, training_instance_matrix , str);
        CLR{j0,num_class+1}{5}=baseModel;
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
                [predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector, test_data, baseModel);
                CLR{j0,k0}{6}=predicted_label;
                CLR{j0,k0}{7}=decision_values;
            end
        end
    end
end

Smooth=1;
%Computing Prior and PriorN

for j0=1:num_class
    for k0=1:num_class
        if ~isempty(CLR{j0,k0})
            lab(j0,:)=lab(j0,:)+CLR{j0,k0}{6}';
        end
        if ~isempty(CLR{k0,j0})
            lab(j0,:)=lab(j0,:)+(~CLR{k0,j0}{6}');
        end
    end
end

k=10;
[~,neighbor] = pdist2(train_data,train_data,'euclidean','Smallest',k);
neighbor=neighbor';
pre=zeros(size(pre_target));
for i=1:num_training
    for j=1:num_class
        if(sum(pre_target(j,neighbor(i,:)))>=k/2)
            pre(j,i)=1;
        end
    end
end

for i=1:num_class
    temp_Ci=sum(pre(i,:)==ones(1,num_training));
    Prior(i,1)=(Smooth+temp_Ci)/(Smooth*2+num_training);
    PriorN(i,1)=1-Prior(i,1);
end
k=5;
[~,neighbor] = pdist2(test_data,test_data,'euclidean','Smallest',k);
neighbor=neighbor';

temp_Ci=zeros(num_class,num_class); %The number of instances belong to the ith class which have k nearest neighbors in Ci is stored in temp_Ci(i,k+1)
temp_NCi=zeros(num_class,num_class); %The number of instances not belong to the ith class which have k nearest neighbors in Ci is stored in temp_NCi(i,k+1)
Outputs=zeros(num_class,num_testing);
for i0=1:num_testing
    temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)   
    for j=1:num_class
        temp(1,j)=round(sum(lab(j,neighbor(i0,:)))/k);
    end
    for j=1:num_class
        if(test_target(j,i0)==1)
            temp_Ci(j,temp(j)+1)=temp_Ci(j,temp(j)+1)+1;
        else
            temp_NCi(j,temp(j)+1)=temp_NCi(j,temp(j)+1)+1;
        end
    end
end
for i=1:num_class
    temp1=sum(temp_Ci(i,:));
    temp2=sum(temp_NCi(i,:));
    for j=1:num_class
        Cond(i,j)=(Smooth+temp_Ci(i,j))/(Smooth*(num_class)+temp1);
        CondN(i,j)=(Smooth+temp_NCi(i,j))/(Smooth*(num_class)+temp2);
    end
end
for i0=1:num_testing
    temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
    for j=1:num_class
        temp(1,j)=lab(j,i0);
    end
    for j=1:num_class
        Prob_in=Prior(j)*Cond(j,temp(1,j)+1);
        Prob_out=PriorN(j)*CondN(j,temp(1,j)+1);
        if(Prob_in+Prob_out==0)
            Outputs(j,i0)=Prior(j);
        else
            Outputs(j,i0)=Prob_in/Prob_out;
        end
    end
end
outputValue=Outputs';
for i=1:num_testing
    outputValue(i,:)=(outputValue(i,:)-min(min(outputValue(i,:))))/(max(max(outputValue(i,:)))-min(min(outputValue(i,:))));
end
Outputs=outputValue';
for i=1:num_testing
    for j=1:num_class
        if(Outputs(j,i)>=0.5)
            Pre_Labels(j,i)=1;
        else
            Pre_Labels(j,i)=0;
        end
    end
end
outputValue=Outputs;
end

