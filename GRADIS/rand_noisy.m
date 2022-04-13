function [y_noisy, noisy_nums] = rand_noisy(y,noisy_num,noisy_ratio)
% y 输入的无噪音标签
% noisy_num 每个实例添加的噪音个数 int
% noisy_ratio 噪音实例占所有样本的比例 double
% y_noisy 处理完带噪音标签
% noisy_nums 噪音数指示矩阵
y_noisy=y;
[N, C]=size(y);
noisy_nums=zeros(N,1);

% 确定添加噪音的实例数量
noise_num_p = ceil(noisy_ratio*N); % 噪音实例数
rand_idx_p=randperm(N); % 随机选择噪音样本
choose_noise_p = rand_idx_p(1:noise_num_p);  % 噪音实例 index
for p=1:noise_num_p
    i = choose_noise_p(p);
    u_idx=find(y(i,:)==0);  
    U_num=length(u_idx);
    if U_num >= noisy_num
        rand_idx=randperm(U_num);
        rand_label= u_idx(rand_idx(1:noisy_num));
        y_noisy(i,rand_label)=1;
        noisy_nums(i)=noisy_num;
    end
    if U_num <  noisy_num
        y_noisy(i,u_idx)=1;
        noisy_nums(i)=U_num;
    end
end