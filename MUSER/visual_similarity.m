%% 功能：获得图像相似性矩阵Wc，以及用于拉普拉斯映射的Pc
%%
%% 输入：训练特征图像矩阵
%%      训练图片索引
%%
%% 输出：Wc图像相似矩阵
%%      Pc度矩阵
function[Wc,Pc]=visual_similarity(Train_X,TrainIndex)

disp('visual_similarity开始')
%数据集改动需要改
k=10;%近邻图像数据

[D_knn_pic,sigma_c]=knn_picture(Train_X,TrainIndex,k);

%% Wc
disp('Wc开始了');
pic_num=size(Train_X,2);
% sigma_c=sigma_c/(pic_num*pic_num);%距离矩阵中值
Wc=zeros(pic_num,pic_num);
for i=1:pic_num
     for j=1:pic_num%当图片i在j的近邻图像集合中或者j在i的近邻图像集合中（=>>Wc对称）
            if any((TrainIndex(i)==D_knn_pic(:,j))|(TrainIndex(j)==D_knn_pic(:,i)))
                 Wc(i,j)=exp(-(norm(Train_X(:,i)- Train_X(:,j))^2)/(sigma_c^2));
            else
                Wc(i,j)=0;
            end   
      end
 end


%% Pc
disp('Pc开始了');
diag_pc=zeros(1,pic_num);%行向量容易求解
for i=1:pic_num
    diag_pc(i)=sum(Wc(i,:))-Wc(i,i);%%一个图片和所有其他图片的相似性（除去自己）
end
Pc=diag(diag_pc);%向量对角化，得度矩阵Pc
disp('Pc结束了');

% disp('visual_similarity结束了')
save('visual_similarity');
end


%% 功能：对于计算图像相似性的矩阵的KNN算法
%%
%% 输入：esp_feature去噪后的特征图像矩阵
%%      esp_index无噪声图的图片索引
%%      k图片近邻的个数
%%
%% 输出：D_knn_pic紧邻图像矩阵序号
%%      sigma_c计算距离的参数
function[D_knn_pic,sigma_c]=knn_picture(Train_X,TrainIndex,k)
disp('knn_pic开始了')
sigma_c=0;%最终为D_knn_pic中所有元素之和
pic_num=size(Train_X,2);
dist=zeros(pic_num,1);%%单个图像和其他各个图像的距离（列向量）
D_knn_pic=zeros(k,pic_num);
for i=1:pic_num
    knn_list=zeros(k,1);%k个最接近的图像c
    for j=1:pic_num
        dist(j)=norm(Train_X(:,i)-Train_X(:,j));%ij图像图像特征（列）求二范式
        sigma_c=sigma_c+dist(j);%i和所有图像特征距离之和（经j次循环）=》所有图像特征距离值和
    end
    [~,index]=sort(dist);%省略了B，即省略了升序排序好的dist
    %%[B,ind]=sort(A)，计算后，B是A排序后的向量，A保持不变，
    %ind是B中每一项对应于A 中项的索引（B的第一项在A的位置!!ind（1）表示B中第一在A中位置，索引）
    %排序是安升序进行的，找到A中位值依然是原来位置，然后找到全图索引，找到此图
    for j=1:k%距离近的相似，所以取前k(距离小，相似)
        %~作为升序排列好的数组，
        %以下，利用dist中的索引位置，将初始的位置索引放入ind
        ind=TrainIndex(index(j));
        knn_list(j)=ind;%cnt原本是全0矩阵，+1之后表示在K近邻之中？
                   %找到近邻图像编号（原始编号，包括有噪声图像的时候的编号）
    end
     D_knn_pic(:,i)=knn_list;   %cnt_mat矩阵为图像与近邻图像矩阵
end
% save('knn_pic')
disp('knn_pic结束')
end