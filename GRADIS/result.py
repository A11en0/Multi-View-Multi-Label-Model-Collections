import evaluate as ev
import scipy.io as sio
import numpy as np
import sys
#from tqdm import tqdm


dataset,k,rate,threshold,alpha = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
#'emotions_noise1_results_for_10_fold.mat'
base_path = 'result/'+dataset+'/'+dataset+'_p3g'#+threshold+'r'+rate+'k'
metric = np.zeros((7,9))
KList = [8]
alphaList = [0.95]
rList=[0.1]
tList = [0.1]
#for idx,alpha in enumerate(alphaList):
for idx,k in enumerate(KList):
#for idx,rate in enumerate(rList):
#for idx,threshold in enumerate(tList):
    second_path = base_path+str(threshold)+'r'+str(rate)+'k'+str(k)+'alpha'+str(alpha)+'_'
    temp = np.zeros((10,9))
    for i in range(10):
        print(i)
        path = second_path+str(i+1)+'_fold_result.mat'
        result = sio.loadmat(path)
        PreLabels,Outputs,test_target = result['Pre_Labels'],result['Outputs'],result['test_target']
        hm = ev.HammingLoss(PreLabels,test_target)
        rl = ev.rloss(Outputs,test_target)
        oe = ev.OneError(Outputs,test_target)
        cv = ev.Coverage(Outputs,test_target)
        av = ev.avgprec(Outputs,test_target)
        ma = ev.MacroAveragingAUC(Outputs,test_target)
        mf = ev.MicroF1(PreLabels,test_target)
        maf = ev.MacroF1(PreLabels,test_target)
        mia = ev.MicroAUC(Outputs,test_target)
        temp[i] = [hm,rl,oe,cv,av,ma,mf,maf,mia]
    metric[idx] = np.mean(temp,0)
print(metric)
