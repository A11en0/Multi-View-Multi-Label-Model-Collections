function [in_result, out_result] = run_arts()
% 这个是最原始的

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
            [J] = genObv( Ytrn, 0.1*j);
            tic;
            [V,U,W,SP,Beta] = MLCTrain(J,Ytrn, Xtrn, Ytst,Xtst,param);
            tm = toc;
            zz = mean(Ytst);
            Ytst(:,zz==-1) = [];
            Xtst(:,zz==-1) = [];
            tstv = (U*W'*Xtst);
            ret =  evalt(tstv,Ytst, (max(tstv(:))-min(tstv(:)))/2);
            ret.time = tm;
            out_result = [out_result;ret];
            zz = mean(Ytrn);
            Ytrn(:,zz==-1) = [];
            Xtrn(:,zz==-1) = [];
            tstv2 = U*W'*Xtrn;
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