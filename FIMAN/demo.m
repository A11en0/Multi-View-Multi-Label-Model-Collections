load('Corel5k.mat');
% load('Pascal.mat');  % Mirflickr

% 转置
% data = reshape(data,size(data,2),size(data,1));
%target = target';
eta=1;
k=10;
thr1=0.4;
thr2=0.6;

rn=2;
p=3;

n_fold = 5;
cvResult  = zeros(16, n_fold);

% 生成噪音
[noise_target, noisy_nums] = rand_noisy(target,3,0.3);

% 生成idx
m = size(data{1},1);
slice = ceil(m/n_fold);
kk = randperm(size(data{1}, 1));                             % shuffle
idx = {};
for index = 1: n_fold
    idx{index, 1} = kk((index - 1) * slice + 1: min(index * slice , m)) ;
end

FIMAN(data,target, noise_target, idx, p, rn, thr1, thr2, eta, k,n_fold);

