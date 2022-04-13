function main(path,dataname,threshold,rate,k,alpha) 
%threshold是过滤噪声标记的阈值
%rate是聚类的比例超参数
%k是k近邻参数,即控制矩阵稀疏程度的参数
%alpha是指数移动加权平均的加权参数
%emotions.mat是示例数据集,p3r3,p5r3,p7r3分别代表三种噪声比例的标记矩阵
    dataset = load(path);
    target = dataset.target ; %p是噪声比例,3是30%,以此类推
    target, noisy_nums = random_noisy(target, 0.3);
    for fold = 1:10
        noise_target = target;
        idx = dataset.idx{fold};
        view_num = size(dataset.data,1);
        train_data = dataset.data;
        test_data = cell(view_num,1);
        test_target = dataset.target(idx,:);
        noise_target(idx,:) = [];
        for i=1:view_num
            temp = train_data{i};
            test_data{i} = temp(idx,:);
            temp(idx,:) = [];
            train_data{i} = temp;
        end
        disp(['The ',num2str(fold),'-th fold is going on...']);
        [Pre_Labels,Outputs]=GRADIS(train_data,noise_target,test_data,test_target,threshold,rate,k,alpha);
        save(['result/',dataname,'/',dataname,'_p3g',num2str(threshold),'r',num2str(rate),'k',num2str(k),'alpha',num2str(alpha),'_',num2str(fold),'_fold_result.mat'],'Outputs','Pre_Labels','test_target','-v7');
    end
end
