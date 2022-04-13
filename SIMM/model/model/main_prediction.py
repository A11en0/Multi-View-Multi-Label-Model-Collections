# -*- coding: UTF-8 -*-
import torch

from model.view_block import ViewBlock
from model.simm_model import SIMM_Model


def main_prediction(features, label_nums, model_state_path, args, model_args):
    # step 1: load  Model
    view_code_list = list(features.keys())
    view_feature_nums_list = [features[code].shape[1] for code in view_code_list]
    view_blocks = [ViewBlock(view_code_list[i], view_feature_nums_list[i],
                             args['comm_feature_nums']) for i in range(len(view_code_list))]
    simm_model = SIMM_Model(view_blocks, args['comm_feature_nums'],
                             label_nums, model_args)
    simm_model.load_state_dict(torch.load(model_state_path))
    simm_model.eval()
    
    # step 2: prediction
    with torch.no_grad():
        results = simm_model(features, False)
        outputs = results['label_predictions']
        outputs = outputs.numpy()
    return outputs

