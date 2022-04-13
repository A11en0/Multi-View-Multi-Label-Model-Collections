# -*- coding: UTF-8 -*-
from model.utilities.common_loss import My_logit_ML_loss
import torch
import os


class Train_SIMM_Model(object):
    def __init__(self, simm_model, train_data_loader, epoch, optimizer, show_epoch,
                 loss_coefficient, model_save_epoch, model_save_dir):
        self.simm_model = simm_model
        self.train_data_loader = train_data_loader
        self.epoch = epoch
        self.optimizer = optimizer
        self.show_epoch = show_epoch
        self.loss_coefficient = loss_coefficient
        self.model_save_epoch = model_save_epoch
        self.model_save_dir = model_save_dir

    def train(self, fold):
        loss_list = []
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        for epoch in range(self.epoch):
            self.simm_model.train()
            for step, train_data in enumerate(self.train_data_loader):
                inputs, labels = train_data
                self.optimizer.zero_grad()
                results = self.simm_model(inputs, True, labels)
                label_predictions = results['label_predictions']
                ML_loss = My_logit_ML_loss(label_predictions, labels)

                loss = self.loss_coefficient['ML_loss'] * ML_loss

                print_str = 'Epoch: ' + str(epoch) + ';\t ML Loss: ' + str(ML_loss.item())

                if 'GAN_loss' in results:
                    GAN_loss = results['GAN_loss']
                    loss = loss + self.loss_coefficient['GAN_loss'] * GAN_loss
                    print_str = print_str + ';\t GAN Loss: ' + str(GAN_loss.item())

                if 'comm_ML_loss' in results:
                    comm_ML_loss = results['comm_ML_loss']
                    loss = loss + self.loss_coefficient['comm_ML_loss'] * comm_ML_loss
                    print_str = print_str + ';\t Comm ML Loss: ' + str(comm_ML_loss.item())

                if 'orthogonal_regularization' in results:
                    orthogonal_regularization = results['orthogonal_regularization']
                    loss = loss + self.loss_coefficient['orthogonal_regularization'] * orthogonal_regularization
                    print_str = print_str + ';\t regularization: ' + str(orthogonal_regularization.item())

                print_str = print_str + ';\t Total Loss: ' + str(loss.item())

                if epoch % self.show_epoch == 0 and step == 0:
                    epoch_loss =dict()
                    epoch_loss['ML_loss'] = ML_loss.item()
                    if 'GAN_loss' in results:
                        epoch_loss['GAN_loss'] = GAN_loss.item()
                    else:
                        epoch_loss['GAN_loss'] = 0
                    if 'comm_ML_loss' in results:
                        epoch_loss['comm_ML_loss'] = comm_ML_loss.item()
                    else:
                        epoch_loss['comm_ML_loss'] = 0
                    if 'orthogonal_regularization' in results:
                        epoch_loss['orthogonal_regularization'] = orthogonal_regularization.item()
                    else:
                        epoch_loss['orthogonal_regularization'] = 0
                    loss_list.append(epoch_loss)
                    print(print_str)

                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % self.model_save_epoch == 0:
                torch.save(self.simm_model.state_dict(),
                        os.path.join(self.model_save_dir,
                                     'fold' + str(fold)+'_' + 'epoch' + str(epoch + 1) + '.pth'))

        return loss_list
