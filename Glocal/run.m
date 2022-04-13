clc;clear;

param = importdata('arts_param.mat');
% data = importdata('dt/Arts_sp.mat');

load('Pascal.mat');
data = reshape(data, size(data, 2), size(data, 1));
data = cell2mat(data);
target = (target>0)*1.0;

param.tooloptions.maxiter = 30;
param.tooloptions.gradnorm = 1e-3;
param.tooloptions.stopfun = @mystopfun;

out_result = [];
in_result = [];

s = RandStream.create('mt19937ar','seed',1);
RandStream.setGlobalStream(s);

n_fold = 5;
cvResult  = zeros(16, n_fold);
[noise_target, noisy_nums] = rand_noisy(target,3,0.0);

m = size(data,1);
slice = ceil(m/n_fold);
kk = randperm(size(data, 1));                             % shuffle
idxs = {};
for index = 1: n_fold
    idxs{index, 1} = kk((index - 1) * slice + 1: min(index * slice , m)) ;
end

for cv=1:n_fold
    train_idx = [];
    test_idx = idxs{cv};
    for idx = 1: n_fold
        if idx == cv
            continue
        end
        train_idx = [train_idx, idxs{idx}];
    end
    
    Xtrn = data(train_idx, :)';
    Ytrn = noise_target(train_idx, :)';
    Xtst = data(test_idx, :)';
    Ytst = target(test_idx, :)';
    [J] = genObv( Ytrn, 1); 
    tic;
    [V,U,W,SP,Beta] = MLCTrain(J,Ytrn, Xtrn, Ytst,Xtst,param);
    tm = toc;
    zz = mean(Ytst);
    Ytst(:,zz==-1) = [];
    Xtst(:,zz==-1) = [];
    tstv = (U*W'*Xtst);
    
    Outputs = tstv;
    test_target = Ytst;
%     thr = (max(Outputs(:))-min(Outputs(:)))/2;  
    thr = 0.5;
    Pre_Labels = sign(Outputs-thr);
    Pre_Labels = (Pre_Labels>0)*1.0;
    test_target = (test_target>0)*1.0;
    
    fprintf('-- Evaluation\n');
    ResultAll = EvaluationAll(Pre_Labels, Outputs, test_target);
    cvResult(:, cv) = cvResult(:, cv) + ResultAll;
    Avg_Result      = zeros(16,2);
    Avg_Result(:,1) = mean(cvResult,2);
    Avg_Result(:,2) = std(cvResult,1,2);
end
PrintResults(Avg_Result);

function stopnow = mystopfun(problem, x, info, last)
    if last < 5
        stopnow = 0;
        return;
    end
    flag = 1;
    for i = 1:3
        flag = flag & abs(info(last-i).cost-info(last-i-1).cost) < 1e-5;
    end
    stopnow = flag;
end