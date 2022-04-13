# -*- coding: UTF-8 -*-
from copy import copy

from scipy.io import loadmat
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset
import torch
import h5py

from random_noise import random_noise


def gen_idx_list(length):
    idxs = np.random.permutation(length)
    idx = []
    fold_size = length // 10
    for i in range(10):
        if i == 9:
            idx.append(list(idxs[i * fold_size:]))
        else:
            idx.append(list(idxs[i*fold_size:(i+1)*fold_size]))

    return idx

def load_mat_data_v1(file_name, need_zscore=False):
    try:
        dataset = loadmat(file_name)
        data = dataset['data'].T
        target = dataset['target'].T
        target = np.array(target, dtype=np.float32)
        target = torch.from_numpy(target)
        idx_list = gen_idx_list(target.size()[0])
        views_features = {}
        for i in range(data.shape[0]):
            view_feature = data[i][0]
            view_feature = np.array(view_feature, dtype=np.float32)
            if need_zscore:
                view_feature = preprocessing.scale(view_feature)
            # 转换成torch类型
            views_features[i] = torch.from_numpy(view_feature)
    except Exception as e:
        print(f"[syai] {e}")
        dataset = h5py.File(file_name)
        data_hdf5 = dataset['data']
        target_hdf5 = dataset['target']
        idx_hdf5 = dataset['idx']
        idx_list = []
        for idx in idx_hdf5[0]:
            idx = dataset[idx]
            idx = np.transpose(idx) - 1
            idx_list.append(np.array(idx[0], dtype=np.int))
        target = np.transpose(target_hdf5)
        target = np.array(target, dtype=np.float32)
        target = torch.from_numpy(target)
        views_features = {}
        i = 0
        for view_feature in data_hdf5[0]:
            view_feature = dataset[view_feature]
            view_feature = np.transpose(view_feature)
            view_feature = np.array(view_feature, dtype=np.float32)
            if need_zscore:
                view_feature = preprocessing.scale(view_feature)
            views_features[i] = torch.from_numpy(view_feature)
            i += 1
    return views_features, target, idx_list

def load_mat_data_v2(file_name, need_zscore=False):
    try:
        dataset = loadmat(file_name)
        data = dataset['data']
        target = dataset['target']
        target = np.array(target, dtype=np.float32)
        target = torch.from_numpy(target)
        idx = dataset['idx']
        idx_list = []
        for i in range(len(idx)):
            idx_list.append(list((idx[i][0][0] - 1)))
        views_features = {}
        for i in range(data.shape[0]):
            view_feature = data[i][0]
            view_feature = np.array(view_feature, dtype=np.float32)
            if need_zscore:
                view_feature = preprocessing.scale(view_feature)
            # 转换成torch类型
            views_features[i] = torch.from_numpy(view_feature)
    except Exception as e:
        print(f"[syai] {e}")
        dataset = h5py.File(file_name)
        data_hdf5 = dataset['data']
        target_hdf5 = dataset['target']
        idx_hdf5 = dataset['idx']
        idx_list = []
        for idx in idx_hdf5[0]:
            idx = dataset[idx]
            idx = np.transpose(idx) - 1
            idx_list.append(np.array(idx[0], dtype=np.int))
        target = np.transpose(target_hdf5)
        target = np.array(target, dtype=np.float32)
        target = torch.from_numpy(target)
        views_features = {}
        i = 0
        for view_feature in data_hdf5[0]:
            view_feature = dataset[view_feature]
            view_feature = np.transpose(view_feature)
            view_feature = np.array(view_feature, dtype=np.float32)
            if need_zscore:
                view_feature = preprocessing.scale(view_feature)
            views_features[i] = torch.from_numpy(view_feature)
            i += 1
    return views_features, target, idx_list

def split_data_set_by_idx(features, labels, idx_list, test_split_id):
    train_idx_list = []
    test_idx_list = idx_list[test_split_id]
    for i in range(len(idx_list)):
        if i != test_split_id:
            train_idx_list.extend(idx_list[i])
    train_idx_list = [i.astype(dtype=np.int64) for i in train_idx_list]
    test_idx_list = [i.astype(dtype=np.int64) for i in test_idx_list]
    train_labels = labels[train_idx_list]
    test_labels = labels[test_idx_list]
    train_features = {}
    test_features = {}
    for code, value in features.items():
        train_features[code] = value[train_idx_list]
        test_features[code] = value[test_idx_list]

    train_partial_labels = torch.Tensor(random_noise(train_labels.numpy().copy(), 5))
    return train_features, train_labels, train_partial_labels, test_features, test_labels


class ViewsDataset(Dataset):
    def __init__(self, views_features, labels):
        self.views_features = views_features
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        feature = {}
        for key, value in self.views_features.items():
            feature[key] = value[item]
        return feature, self.labels[item]


class MergeDataset(Dataset):
    def __init__(self, views_features, labels):
        self.views_features = views_features
        self.labels = labels
        self.view_count = len(self.views_features)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        merge_feature = []
        for i in range(self.view_count):
            merge_feature.append(self.views_features[i][item])
        merge_feature = torch.cat(merge_feature)
        return merge_feature, self.labels[item]


# read data with matlab format by loadmat
def read_mat_data(file_name, need_zscore=True):
    # since the data is small, we load all data to the memory
    data = loadmat(file_name)
    features = data['data']
    views_count = features.shape[0]
    views_features = {}
    for i in range(views_count):
        view_feature = features[i][0]
        # change sparse to dense
        if type(view_feature) != type(np.array([1])):
            view_feature = view_feature.toarray()
        view_feature = np.array(view_feature, dtype=np.float32)
        if need_zscore:
            view_feature = preprocessing.scale(view_feature)
        # views_features['view_' + str(i)] = view_feature
        views_features[i] = torch.from_numpy(view_feature)
    labels = data['target']
    if type(labels) != type(np.array(1)):
        labels = labels.toarray()
    labels = np.array(labels, dtype=np.float32)
    labels = torch.from_numpy(labels)
    return views_features, labels
