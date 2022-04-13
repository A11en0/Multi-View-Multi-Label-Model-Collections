function [in_result, out_result] = run_arts()
% 杩欎釜鏄渶鍘熷鐨?

param = importdata('arts_param.mat');
data = importdata('dt/Arts_sp.mat');

param.tooloptions.maxiter = 30;
param.tooloptions.gradnorm = 1e-3;
param.tooloptions.stopfun = @mystopfun;

 out_result = [];
 in_result = [];

    for j=3:4:8
        s = RandStream.create('mt19937ar','seed',1);
        RandStream.setGlobalStream(s);
        
        for kk = 1:1
            Xtrn = data.train{kk,1};
            Ytrn = data.train{kk,2};
            Xtst = data.test{kk,1};
            Ytst = data.test{kk,2};
            [J] = genObv( Ytrn, 0.1*j); %这里是1 % 杩欓噷澶勭悊鐨刴issing 这里这届写J=1 就不是处理missing了
            tic;
            [V,U,W,SP,Beta] = MLCTrain(J,Ytrn, Xtrn, Ytst,Xtst,param);
            tm = toc;
            zz = mean(Ytst);
            Ytst(:,zz==-1) = [];
            Xtst(:,zz==-1) = [];
            % 他这里操作了两次,是干啥，不用改？dui 你不用管 我那边给删了，重新加上去看看
            tstv = (U*W'*Xtst); % 1
            ret =  evalt(tstv,Ytst, (max(tstv(:))-min(tstv(:)))/2);%在这就结束了
            ret.time = tm;
            out_result = [out_result;ret];
            zz = mean(Ytrn);
            Ytrn(:,zz==-1) = [];
            Xtrn(:,zz==-1) = [];
            tstv2 = U*W'*Xtrn; % 2  这里不用管 他这是 Xtran 嗯嗯 那现在应该
            ret =  evalt(tstv2,Ytrn, (max(tstv2(:))-min(tstv2(:)))/2);
            in_result = [in_result;ret];
        end
    end
end

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