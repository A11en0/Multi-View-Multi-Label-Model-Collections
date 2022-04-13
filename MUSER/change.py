from scipy.io import savemat
import hdf5storage

dataname = "scenex.mat"
dataset = hdf5storage.loadmat(dataname)
d = []
for i in range(1, dataset['data'].ndim + 1):
    d.append(dataset['data'][i])
target = dataset['target']
savemat('scenexx.mat', mdict={'data': d, 'target':target})
