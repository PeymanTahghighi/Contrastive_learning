import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import transforms
import config
import torch
from utils import tensor_img_to_numpy
import albumentations as A
from albumentations.pytorch import ToTensorV2


final_transforms = A.Compose([
    ToTensorV2()
], additional_targets={'mask': 'mask'})

final_transforms_norm = A.Compose([
    A.Normalize(),
    ToTensorV2()
], additional_targets={'mask': 'mask'})


crops_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=96),
                                         ])
initial_transforms = transforms.Compose([crops_transforms,
                                            transforms.RandomApply([
                                            transforms.ColorJitter(brightness=0.5,
                                                                    contrast=0.5,
                                                                    saturation=0.5,
                                                                    hue=0.1)
                                        ], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.GaussianBlur(kernel_size=9),
                                        ])



class ContrastiveTransformations(object):

    def __init__(self):
        pass

    def __GMML_drop_random_patch(self, img):
        img = np.asarray(img);
        img = img/255;
        img = ((img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
        num_drops = np.random.randint(4, config.MAX_DROPS);
        mask = np.zeros_like(img);
        for n in range(num_drops):
            drop_w = np.random.randint(int(img.shape[1]*0.05),int(img.shape[1]*0.3));
            drop_h = np.random.randint(int(img.shape[0]*0.05),int(img.shape[0]*0.3));
            drop_start_x = np.random.randint(0,img.shape[1]);
            drop_start_h = np.random.randint(0,img.shape[0]);
            end_point_x = min(img.shape[1], drop_w+drop_start_x);
            end_point_y = min(img.shape[0], drop_h+drop_start_h);
            mask[drop_start_x:end_point_x, drop_start_h:end_point_y] = 1;

            op = np.random.randint(1,4);
            if op == 1:
                img[drop_start_x:end_point_x, drop_start_h:end_point_y] =  np.random.random((end_point_x-drop_start_x, end_point_y-drop_start_h,3));
                if config.DBG is True:
                    img_tmp = ((img * [0.229, 0.224, 0.225] ) + [0.485, 0.456, 0.406])
                    cv2.imshow('t',(img_tmp*255).astype("uint8"));
                    cv2.waitKey();
            elif op == 2:
                img[drop_start_x:end_point_x, drop_start_h:end_point_y] =  np.ones((end_point_x-drop_start_x, end_point_y-drop_start_h,3));
                if config.DBG is True:
                    img_tmp = ((img * [0.229, 0.224, 0.225] ) + [0.485, 0.456, 0.406])
                    cv2.imshow('t',(img_tmp*255).astype("uint8"));
                    cv2.waitKey();
                    #random noise

        img_trans = final_transforms(image = img, mask = mask);
        img = img_trans['image'];
        mask = img_trans['mask'];
        return img, mask.permute(2,0,1);

    def __call__(self, x):
        crops_samples = [initial_transforms(x) for i in range(2)];
        corrupted_samples = [];
        masks = [];
        crops_ret = [];
        for i in range(2):
            corrupted, mask = self.__GMML_drop_random_patch(crops_samples[i]);
            corrupted_samples.append(corrupted);
            masks.append(mask);
        
        for i in range(2):
            t = final_transforms_norm(image = np.asarray(crops_samples[i]), mask=np.zeros_like(np.asarray(crops_samples[i])));
            crops_ret.append(t['image']);

        return crops_ret, corrupted_samples, masks;