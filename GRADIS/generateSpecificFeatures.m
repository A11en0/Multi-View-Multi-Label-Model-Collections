function [new_data] = generateSpecificFeatures(data, centers)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    num_center=size(centers,1);
    num_data = size(data,1);
    new_data=[];
        
    if(num_center>=5000)
        error('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...');
    else
        blocksize=5000-num_center;
        num_block=ceil(num_data/blocksize);
        for j=1:num_block-1
            low=(j-1)*blocksize+1;
            high=j*blocksize;
                
            tmp_mat=[centers;data(low:high,:)];
            Y=pdist(tmp_mat);
            Z=squareform(Y);
            new_data=[new_data;Z((num_center+1):(num_center+blocksize),1:num_center)];                
        end
            
        low=(num_block-1)*blocksize+1;
        high=num_data;
            
        tmp_mat=[centers;data(low:high,:)];
        Y=pdist(tmp_mat);
        Z=squareform(Y);
        new_data=[new_data;Z((num_center+1):(num_center+high-low+1),1:num_center)];
    end
end

