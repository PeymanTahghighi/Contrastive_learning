import torch

def tensor_img_to_numpy(img):
    img = img.permute(1,2,0).detach().cpu().numpy();
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406];
    img = (img*255).astype("uint8")
    return img;