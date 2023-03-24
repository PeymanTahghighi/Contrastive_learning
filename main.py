## Standard libraries
from config import *
import pickle
import os
from copy import deepcopy
from tkinter import Image
from turtle import forward

## Imports for plotting
import matplotlib.pyplot as plt

from utils import loss_fn, tensor_img_to_numpy
plt.set_cmap('cividis')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()
import numpy as np
## tqdm for loading bars
from tqdm import tqdm
from LR import LogisticRegression
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import cv2
## Torchvision
import torchvision
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets.vit import ViT as ViTM
from transformer import ViT as ViTS

from datautils import ContrastiveTransformations

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import urllib.request
from urllib.error import HTTPError

class RECHead(nn.Module):
    def __init__(self, in_dim, in_channels = 3, patch_size = 12) -> None:
        super().__init__();
        self.mlp= nn.Sequential(nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
            nn.GELU());
        
        self.conv_trans = nn.ConvTranspose2d(in_dim, in_channels, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size));
    
    def forward(self, x):
        out = self.mlp(x);
        out = out.transpose(1,2)
        out = out.unflatten(2, (8,8));
        x_rec = self.conv_trans(out);

        return x_rec;

class CLSHead(nn.Module):
    def __init__(self, in_dim, bottleneck_dim, expansion_rate = 4, n_layers = 3) -> None:
        super().__init__();
        layers = [];
        layers.append(nn.Linear(in_dim, in_dim*expansion_rate, bias=False));
        layers.append(nn.BatchNorm1d(in_dim*expansion_rate));
        layers.append(nn.ReLU(inplace=True));
        for _ in range(n_layers-2):
            layers.append(nn.Linear(in_dim*expansion_rate, in_dim*expansion_rate, bias=False));
            layers.append(nn.BatchNorm1d(in_dim*expansion_rate));
            layers.append(nn.GELU());
        
        layers.append(nn.Linear(in_dim*expansion_rate, bottleneck_dim));
        layers.append(nn.BatchNorm1d(bottleneck_dim, affine=False));

        self.mlp = nn.Sequential(*layers);
    
    def forward(self, x):
        return self.mlp(x);

class MLP(nn.Module):
    def __init__(self, inp_size, out_size, hidden_size) -> None:
        super().__init__();
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size),
        )
    def forward(self, x):
        return self.net(x);

class SelfSupervisedModel(nn.Module):
    def __init__(self, projection, predicition=None) -> None:
        super().__init__();
        self.encoder = ViTS(3, 96,12,768,12,0.0,4,10 );

        self.projection = projection;
        self.prediction = predicition;
        self.rec_head = RECHead(768);

    def forward(self, x):
        model_output = self.encoder(x);

        projection = self.projection(model_output[:,0]);
        if self.prediction is not None:
            predicition = self.prediction(projection);
            #rconstruction head
            rh = self.rec_head(model_output[:,1:]);
            return rh, predicition;
        else:
            return projection;

def update_ema(ma_model, current_model, beta=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data;
        ma_params.data = old_weight * beta + (1-beta) * up_weight;

def train(e, online, target, optimizer, loader,):
    print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'Rec_Loss'));
    pbar = enumerate(loader);
    pbar = tqdm(pbar, total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}');
    epochs_loss = [];

    epoch_rec_loss = [];
    
    for idx, (batch) in pbar:
        imgs, _ = batch
        imgs_crops = torch.cat(imgs[0], dim = 0)
        imgs_corrupted = torch.cat(imgs[1], dim = 0);
        imgs_masks = torch.cat(imgs[2], dim=0)
        imgs_crops, imgs_corrupted, imgs_masks = imgs_crops.to(DEVICE), imgs_corrupted.to(DEVICE), imgs_masks.to(DEVICE);

        if DBG is True:
            for i in range(BATCH_SIZE):
                first = tensor_img_to_numpy(imgs_corrupted[i]);
                second = tensor_img_to_numpy(imgs_corrupted[i+BATCH_SIZE]);
                fig, ax = plt.subplots(1,2);
                ax[0].imshow(first);
                ax[1].imshow(second);
                ax[0].set_axis_off();
                ax[1].set_axis_off();
                plt.show();
                plt.clf();
            

        #run images through online and target networks
        rh, prediction = online(imgs_corrupted.float())
        with torch.no_grad():
            proj = target(imgs_corrupted.float());
            proj.detach_();

        #get loss
        loss, rec_loss = loss_fn(prediction, proj.detach(), rh, imgs_crops, imgs_masks);

        
        loss.backward();
        optimizer.step();
        loss = loss.item();
        online.zero_grad(set_to_none = True);

        epochs_loss.append(loss);
        epoch_rec_loss.append(rec_loss.item());

        #update EMA
        update_ema(target.encoder, online.encoder);
        update_ema(target.projection, online.projection);

        pbar.set_description(("%10s" + "%10.4g"*2) %(e, np.mean(epochs_loss), np.mean(epoch_rec_loss)))
        pass

    return np.mean(epochs_loss);

def save_samples(e, model, loader):
    with torch.no_grad():
        if os.path.exists(('samples\\%i')%(e)) is False:
            os.makedirs(('samples\\%i')%(e));
        b = next(iter(loader));
        imgs, _ = b
        imgs_crops = torch.cat(imgs[0], dim=0)
        imgs_corrupted = torch.cat(imgs[1], dim=0)
        imgs_crops, imgs_corrupted = imgs_crops.to(DEVICE), imgs_corrupted.to(DEVICE);

        if DBG is True:
            for i in range(BATCH_SIZE):
                first = tensor_img_to_numpy(imgs_corrupted[i]);
                second = tensor_img_to_numpy(imgs_corrupted[i+BATCH_SIZE]);
                fig, ax = plt.subplots(1,2);
                ax[0].imshow(first);
                ax[1].imshow(second);
                ax[0].set_axis_off();
                ax[1].set_axis_off();
                plt.show();
                plt.clf();
            

        # # Encode all images
        ch,rh = model(imgs_corrupted.float())
        for i in range(20):
            r = np.random.randint(0,BATCH_SIZE);
            orig = tensor_img_to_numpy(imgs_crops[i]);
            rec = tensor_img_to_numpy(rh[i]);
            cv2.imwrite(f'samples\\{e}\\{i}_orig.png', orig);
            cv2.imwrite(f'samples\\{e}\\{i}_rec.png', rec);

def valid(e, online, target, loader,):
    print(('\n' + '%10s'*3) %('Valid: Epoch', 'Loss', 'Rec_loss'));
    pbar = enumerate(loader);
    pbar = tqdm(pbar, total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}');
    epochs_loss = [];
    epoch_rec_loss = [];
    plot = True;
    with torch.no_grad():
        for idx, (batch) in pbar:
            imgs, _ = batch
            imgs_crops = torch.cat(imgs[0], dim = 0)
            imgs_corrupted = torch.cat(imgs[1], dim = 0);
            imgs_masks = torch.cat(imgs[2], dim=0)
            imgs_crops, imgs_corrupted, imgs_masks = imgs_crops.to(DEVICE), imgs_corrupted.to(DEVICE), imgs_masks.to(DEVICE);

            rh, prediction = online(imgs_corrupted.float())

            proj = target(imgs_corrupted.float());
            proj.detach_();

            #get loss
            loss, rec_loss = loss_fn(prediction, proj.detach(), rh, imgs_crops, imgs_masks);

            epochs_loss.append(loss.item());
            epoch_rec_loss.append(rec_loss.item());

            if plot is True:
                plot = False;
                bz = 8;

                imagesToPrint = torch.cat([imgs_crops[0: min(15, bz)].cpu(),  imgs_corrupted[0: min(15, bz)].cpu(),
                                        rh[0: min(15, bz)].cpu(), imgs_masks[0: min(15, bz)].cpu()], dim=0)
                torchvision.utils.save_image(imagesToPrint, f'samples\\{e}.png', nrow=min(15, bz), normalize=True, range=(-1, 1))

            pbar.set_description(("%10s" + "%10.4g"*2) %(e, np.mean(epochs_loss), np.mean(epoch_rec_loss)))

    return np.mean(epochs_loss), np.mean(epoch_rec_loss);

@torch.no_grad()
def prepare_data_features(model, dataset):
    # Prepare model
    network = deepcopy(model)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(DEVICE)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(DEVICE)
        batch_feats,_ = network(batch_imgs)
        batch_feats = torch.mean(batch_feats, dim = 1);
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return data.TensorDataset(feats, labels)

def get_smaller_dataset(original_dataset, num_imgs_per_label):
    new_dataset = data.TensorDataset(
        *[t.unflatten(0, (10, -1))[:,:num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors]
    )
    return new_dataset

if __name__ == "__main__":

    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial17/"
    # Files to download
    pretrained_files = ["SimCLR.ckpt", "ResNet.ckpt",
                        "tensorboards/SimCLR/events.out.tfevents.SimCLR",
                        "tensorboards/classification/ResNet/events.out.tfevents.ResNet"]
    pretrained_files += [f"LogisticRegression_{size}.ckpt" for size in [10, 20, 50, 100, 200, 500]]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if "/" in file_name:
            os.makedirs(file_path.rsplit("/",1)[0], exist_ok=True)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)


    if TRAIN_CONTRASTIVE is True:
        unlabeled_data = STL10(root=DATASET_PATH, split='unlabeled', download=True,
                            transform=ContrastiveTransformations())
        train_data_contrast = STL10(root=DATASET_PATH, split='train', download=True,
                                    transform=ContrastiveTransformations())
        total = len(unlabeled_data);
        data3 = torch.utils.data.random_split(unlabeled_data, [total//100, total-total//100])[0]
        train_loader = data.DataLoader(unlabeled_data, batch_size=BATCH_SIZE, shuffle=True,
                                        drop_last=True, pin_memory=True, num_workers=0)
        val_loader = data.DataLoader(train_data_contrast, batch_size=BATCH_SIZE, shuffle=False,
                                        drop_last=False, pin_memory=True, num_workers=0, )
        
        online = SelfSupervisedModel(projection=CLSHead(768,256), predicition=MLP(256,256,4096));
        target = SelfSupervisedModel(projection=CLSHead(768,256));
        target.load_state_dict(online.state_dict(), strict=False);
        
        online = online.to(DEVICE);
        target = target.to(DEVICE);

        summary_writer = SummaryWriter('exp\\self-imp');
        best_loss = 1e10;
        optimizer = optim.AdamW(online.parameters(),
                                lr=LR)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=MAX_EPOCHS,
                                                            eta_min=LR/50)
        best_model = None;
        for e in range(MAX_EPOCHS):
            online.train();
            target.train();
            train_loss = train(e, online, target, optimizer, train_loader);
            online.eval();
            target.eval();
            loss, rec_loss = valid(e, online, target, val_loader);

            summary_writer.add_scalar('Train/Loss', train_loss, e);
            summary_writer.add_scalar('Valid/Loss', loss, e);
            #summary_writer.add_scalar('Valid/Acc', acc, e);

            if loss < best_loss:
                best_loss = loss;
                best_model = online.state_dict();
                pickle.dump(best_model, open('ckpt.pt', 'wb'))
            lr_scheduler.step();
    else:

        img_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])

        train_img_data = STL10(root=DATASET_PATH, split='train', download=True,
                            transform=img_transforms)
        test_img_data = STL10(root=DATASET_PATH, split='test', download=True,
                            transform=img_transforms)

        model = ViTS(3, 96,12,768,12,0.0,4,True, 10 );
        state_dict = pickle.load(open('ckpt.pt', 'rb'));
        state_dict_encoder = dict();
        for s in state_dict:
            if 'encoder' in s:
                state_dict_encoder[s[8:]] = state_dict[s];


        model.load_state_dict(state_dict_encoder, strict=False);

        train_lr_data = prepare_data_features(model, train_img_data);
        test_lr_data = prepare_data_features(model, test_img_data);

        lr_data_loader_train = DataLoader(train_lr_data, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, drop_last=True);
        lr_data_loader_test = DataLoader(test_lr_data, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, drop_last=True);

        d = get_smaller_dataset(train_lr_data, 5);

        lr = LogisticRegression(768,10);
        for e in range(100):
            lr.train(e, lr_data_loader_train);
            lr.valid(e, lr_data_loader_test);