function [Center] = clusterCenter(pdata,p_cluster_index,ndata,n_cluster_index)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    k = size(unique(p_cluster_index),1);
    %Center = cell(view_num,1);
    
    p_center = [];
    n_center = [];
    
    for j=1:k
        p_idx = find(p_cluster_index==j);
        n_idx = find(n_cluster_index==j);
        p_center = [p_center;mean(pdata(p_idx,:),1)];
        n_center = [n_center;mean(ndata(n_idx,:),1)];
    end
    Center = [p_center;n_center];
end
