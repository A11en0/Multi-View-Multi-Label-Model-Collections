# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class ViewBlock(nn.Module):

    def __init__(self, code, input_feature_num, output_feature_num):
        super(ViewBlock, self).__init__()
        self.code = code
        self.fc_extract_comm = nn.Linear(input_feature_num, output_feature_num)
        self.fc_private = nn.Linear(input_feature_num, output_feature_num)

    def forward(self, input):
        x_private = F.relu(self.fc_private(input))
        x_comm_feature = F.relu(self.fc_extract_comm(input))
        return x_private, x_comm_feature