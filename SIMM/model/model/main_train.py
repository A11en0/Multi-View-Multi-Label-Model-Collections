# -*- coding: UTF-8 -*-
import torch.optim as optim
from torch.utils.data import DataLoader
from model.utilities.common_tools import ViewsDataset

from model.view_block import ViewBlock
from model.simm_model import SIMM_Model
from model.train_simm_model import Train_SIMM_Model


def main_simm_model_train(features, labels, args,
                     loss_coefficient,
                           model_args = None, weight_decay=1e-5, fold=1):
    # step 1: load views features and labels
    views_dataset = ViewsDataset(features, labels)
    views_data_loader = DataLoader(views_dataset, batch_size=128, shuffle=True,
                                   num_workers=0)

    # step 2: instantiation View Model
    label_nums = labels.shape[1]
    view_code_list = list(features.keys())
    view_feature_nums_list = [features[code].shape[1] for code in view_code_list]
    view_blocks = [ViewBlock(view_code_list[i], view_feature_nums_list[i],
                             args['comm_feature_nums']) for i in range(len(view_code_list))]
    simm_model = SIMM_Model(view_blocks, args['comm_feature_nums'],
                             label_nums, model_args)

    # step 3: init optimizer
    optimizer = optim.Adam(simm_model.parameters(), lr=0.001, weight_decay=weight_decay)

    # step 4: train model
    trainer = Train_SIMM_Model(simm_model, views_data_loader, args['epoch'],
                                optimizer, args['show_epoch'], loss_coefficient,
                                args['model_save_epoch'], args['model_save_dir'])
    loss_list = trainer.train(fold)
    return loss_list