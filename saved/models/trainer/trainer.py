import torch
import torch.nn as nn

import random, os
import numpy as np
from tqdm import tqdm

from utils.utils import rand_bbox, accuracy, EarlyStopping
from utils.dataloader import CustomImageFolder, get_transform
from model.optimizer import SAM
from model.mymodel import ConvMixer, init_weights

class Trainer(object):

    def __init__(self, train_dataset, valid_dataset, base_model, config):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.base_model = base_model
        self.CONFIG = config
        self.device = self._get_device()
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.075)
        self.model_dict = {"mymodel": ConvMixer}

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _get_model(self):
        try:
            model = self.model_dict[self.base_model]
            return model
        except:
            raise ("Invalid model name. Pass one of the model dictionary.")

    def _validate(self, model: nn.Module, dataloader, loss_fn):
        valid_avg_acc, valid_avg_loss = 0, 0

        model.eval()
        with torch.no_grad(): 
            for X, y in dataloader: 
                X, y = X.to(self.device), y.to(self.device)
                yhat = model(X)
                loss = loss_fn(yhat, y) ## valid loss
                acc = accuracy(y.cpu().data.numpy(), yhat.cpu().data.numpy().argmax(-1))       
                valid_avg_acc += (acc * len(y) / len(dataloader.dataset)) 
                valid_avg_loss += loss.item() / len(dataloader) 

        return valid_avg_loss, valid_avg_acc

    def _train(self, model: nn.Module, dataloader, optimizer, loss_fn, scheduler=None, beta=1, cutmix_prob=0.5):
        model.train()
        train_avg_loss = 0 
        train_avg_acc = 0
        train_total_batch = len(dataloader)

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(X.size()[0]).cuda()
                target_a = y
                target_b = y[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
                X[:, :, bbx1:bbx2, bby1:bby2] = X[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size()[-1] * X.size()[-2]))
                # compute output
                yhat = model(X)
                loss = loss_fn(yhat, target_a) * lam + loss_fn(yhat, target_b) * (1. - lam)
                loss.backward() 
                optimizer.first_step(zero_grad=True)

                yhat = model(X)
                loss = loss_fn(yhat, target_a) * lam + loss_fn(yhat, target_b) * (1. - lam)
                loss.backward() 
                optimizer.second_step(zero_grad=True)
            else:
                # compute output
                yhat = model(X)
                loss = loss_fn(yhat, y)
                loss.backward() 
                optimizer.first_step(zero_grad=True)

                loss_fn(model(X), y).backward()
                optimizer.second_step(zero_grad=True)            

            train_avg_loss += (loss.item() / train_total_batch) 
            acc = accuracy(y.cpu().data.numpy(), yhat.cpu().data.numpy().argmax(-1))       
            train_avg_acc += (acc * len(y) / len(dataloader.dataset)) 

        if scheduler is not None:
            scheduler.step()

        return model, train_avg_loss, train_avg_acc    

    def train_step(self):

        best_acc = 0
        
        early_stopping = EarlyStopping(patience=50,
                                    verbose=False,
                                    path=self.CONFIG['save_path'])
        
        mmodel = self._get_model()
        net = mmodel(dim=self.CONFIG['dim'], depth=self.CONFIG['depth'], kernel_size=self.CONFIG['kernel'], patch_size=self.CONFIG['patch'], n_classes=self.CONFIG['nclasses']).to(self.device)
        net.apply(init_weights)
        

        ## optimizer
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(net.parameters(), base_optimizer, lr=self.CONFIG['learning_rate'])
        loss_fn = self.loss.to(self.device)
        
        ## scheduler  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        for epoch in tqdm(range(self.CONFIG['epochs'])):
            net, train_avg_loss, train_avg_acc = self._train(net, self.train_dataset, optimizer, loss_fn, scheduler,cutmix_prob=self.CONFIG['cutmix_prob'])  ## 모델 학습
            valid_avg_loss, valid_avg_acc = self._validate(net, self.valid_dataset, loss_fn)  ## 모델 평가

            if epoch % 10 == 0 or epoch == self.CONFIG['epochs'] - 1: 
                print('[Epoch: {:>3}] train loss = {:>.5}  valid loss = {:>.5} valid acc = {:>.5} train acc = {:>.5}'.format(epoch + 1, train_avg_loss, valid_avg_loss, valid_avg_acc, train_avg_acc)) 
                
            early_stopping(net, valid_avg_loss, valid_avg_acc) 
            if early_stopping.early_stop: 
                if epoch % 10 != 0 and epoch != self.CONFIG['epochs'] - 1:
                    print('[Epoch: {:>3}] train loss = {:>.5}  valid loss = {:>.5}'.format(epoch + 1, train_avg_loss, valid_avg_loss)) 
                print('Early stopping!')
                break
                
        print(f'Best Valid Loss: {early_stopping.val_loss_min:.4f}  Best Valid Accuracy: {early_stopping.val_acc_max:.4f}\n\n')
    