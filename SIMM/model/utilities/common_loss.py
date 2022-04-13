# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn


def My_KL_loss(predictions, true_distributions):
    predictions = F.log_softmax(predictions, dim=1)
    KL = (true_distributions * predictions).sum()
    KL = -1.0 * KL / predictions.shape[0]
    return KL


def My_logit_ML_loss(view_predictions, true_labels):
    view_predictions_sig = torch.sigmoid(view_predictions)
    criterion = nn.BCELoss()
    ML_loss = criterion(view_predictions_sig, true_labels)
    return ML_loss
