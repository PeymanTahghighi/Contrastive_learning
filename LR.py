from numpy import outer
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes) -> None:
        super().__init__();
        self.layer = nn.Linear(in_features, num_classes);
        self.layer.to('cuda');
        self.optimzer = optim.Adam(self.layer.parameters(), 1e-4);
        
    def train(self, e, loader):
        self.layer.train();
        pbar = tqdm(enumerate(loader), total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}');
        epoch_loss = [];
        epoch_acc = [];
        for idx, (batch) in pbar:
            imgs, lbl = batch;
            imgs, lbl = imgs.to('cuda'), lbl.to('cuda');
            output = self.layer(imgs);
            loss = F.cross_entropy(output, lbl, reduction='mean')
            acc = (lbl == torch.argmax(torch.softmax(output,dim=1), dim = 1)).sum() / len(lbl);
            loss.backward();
            self.optimzer.step();
            self.layer.zero_grad(set_to_none=True);

            epoch_loss.append(loss.item());
            epoch_acc.append(acc.item());

            pbar.set_description(('%10i' + '%10.4g'*2)%(e, np.mean(epoch_loss), np.mean(epoch_acc)));
    
    def valid(self, e, loader):
        self.layer.eval();
        pbar = tqdm(enumerate(loader), total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}');
        epoch_loss = [];
        epoch_acc = [];
        for idx, (batch) in pbar:
            imgs, lbl = batch;
            imgs, lbl = imgs.to('cuda'), lbl.to('cuda');
            output = self.layer(imgs);
            loss = F.cross_entropy(output, lbl, reduction='mean')
            acc = (lbl == torch.argmax(torch.softmax(output,dim=1), dim = 1)).sum() / len(lbl);

            epoch_loss.append(loss.item());
            epoch_acc.append(acc.item());

            pbar.set_description(('valid: %10i' + '%10.4g'*2)%(e, np.mean(epoch_loss), np.mean(epoch_acc)));
