import evaluate as ev
import scipy.io as sio
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
#noise_version = int(sys.argv[2])
#dataset = sys.argv[1]
#'emotions_noise1_results_for_10_fold.mat'
#save_path='result/youku5k/r{}/r{}p{}eta_{}_k_{}_result.csv'
#base_path = 'result/youku5k/r{}/mvpml_r{}p{}eta_{}k_{}.mat'

save_path='runex/youku15k/r{}/r{}p{}_result.csv'
base_path = 'runex/youku15k/r{}/mvpml_r{}p{}thr1_0.4thr2_0.6k_10i_{}.mat'

r=['3']
p=['7']
thr2=['0.6']
#thr1=['0.4','0.5','0.6']
#thr2=['0.5','0.6']
#eta=['1']
#k=['7','8','10']
metric = np.zeros((10,9))

for rr in r:
    for pp in p:
        #for tt in thr2:
       # for tt1 in thr1:
        #for ee in eta:
            #for tt2 in thr2:
             #for kk in k:
                for i in tqdm(range(1)):
    
                    result = sio.loadmat(base_path.format(rr,rr,pp,i+1))
                    PreLabels,Outputs,test_target = result['P'],result['O'],result['T']
                    #PreLabels,Outputs,test_target = result['Pre_cell'][i][0].T,result['Output_cell'][i][0].T,result['Test_cell'][i][0].T
                    hm = ev.HammingLoss(PreLabels,test_target)
                    rl = ev.rloss(Outputs,test_target)
                    oe = ev.OneError(Outputs,test_target)
                    cv = ev.Coverage(Outputs,test_target)
                    av = ev.avgprec(Outputs,test_target)
                    ma = ev.MacroAveragingAUC(Outputs,test_target)
                    mf = ev.MicroF1(PreLabels,test_target)
                    maf = ev.MacroF1(PreLabels,test_target)
                    mia = ev.MicroAUC(Outputs,test_target)
                    metric[i] = [hm,rl,oe,cv,av,ma,mf,maf,mia]

                    mean_result=np.mean(metric,0)
                    std_result=np.std(metric,0)


                print(mean_result)
                print(std_result)
                result_final=np.concatenate((metric,np.array([mean_result]),np.array([std_result])),axis=0)
                result_final=pd.DataFrame(result_final)
                result_final.columns = ['hm','rl','oe','cv','av','ma','mf','maf','mia']
                result_final.index=['1','2','3','4','5','6','7','8','9','10','mean','std']

                result_final.to_csv(save_path.format(rr,rr,pp))
