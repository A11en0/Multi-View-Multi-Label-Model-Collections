function sample_data = NegSample(neg_data, sample_num)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    index = randsample(neg_data,sample_num);
    sample_data = neg_data(index,:);
end

