import torch
import torch.nn.functional as F
import config

def tensor_img_to_numpy(img):
    img = img.permute(1,2,0).detach().cpu().numpy();
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406];
    img = (img*255).astype("uint8")
    return img;

def mse(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def loss_fn(prediction, proj, recons, orig, masks):
    loss_1 = mse(prediction[:int(prediction.shape[0]/2)], proj[int(prediction.shape[0]/2):]);
    loss_2 = mse(prediction[int(prediction.shape[0]/2):], proj[:int(prediction.shape[0]/2)]);
    simclr_loss = loss_1 + loss_2;
    simclr_loss = simclr_loss.mean();

    #calculate rec loss
    rec_loss = F.l1_loss(recons, orig, reduction='none');
    rec_loss = rec_loss[masks == 1].mean();

    loss = simclr_loss + rec_loss;
    return loss, rec_loss;
    

def old_cls_loss(enc, temp, device):
    cos_sim = F.cosine_similarity(enc[:,None,:], enc[None,:,:], dim=-1)
    # # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # # InfoNCE loss
    cos_sim = cos_sim / temp
    simclr_loss = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    simclr_loss = simclr_loss.mean()
