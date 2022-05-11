import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Additional/supporting methods and code for the 2021 IN5400 lab on image segmentation.
# Some of this code has been copied or heavily inspired by chunks found online.


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def write_sample(ims, outputs, targets, pth, epoch, phase):
    if pth is not None:
        y_hat = np.argmax(outputs.detach().cpu().numpy(), 1)
        Image.fromarray((y_hat[0, :, :] * 255 / 2).astype(np.uint8)).save(os.path.join(pth, 'out_sample_{:05d}_{}_pred.png'.format(epoch, phase)))
        im_0 = ims[0, :, :, :].detach().cpu().numpy()
        im_0 = (im_0 - np.min(im_0))
        im_0 = (im_0 / np.max(im_0) * 255).astype(np.uint8)
        Image.fromarray(np.transpose(im_0, [1, 2, 0])).save(os.path.join(pth, 'out_sample_{:05d}_{}_input.png'.format(epoch, phase)))
        y_0 = targets[0, :, :].detach().cpu().numpy()
        y_0 = (y_0 / 2 * 255).astype(np.uint8)
        Image.fromarray(y_0).save(os.path.join(pth, 'out_sample_{:05d}_{}_target.png'.format(epoch, phase)))


class PetDataset(Dataset):

    def __init__(self, im_list, pth_images, pth_gt):
        self.imgs = [(os.path.join(pth_images, im), os.path.join(pth_gt, im.replace('.jpg', '.png'))) for im in im_list]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        im_filename, gt_filename = self.imgs[index]
        im = Image.open(im_filename).convert('RGB')
        im_gt = Image.open(gt_filename)

        im = im.resize((256, 256))
        im_gt = im_gt.resize((256, 256), Image.NEAREST)

        im = self.transform(im)
        im_gt = torch.tensor(np.array(im_gt)-1)  # Make labels 1,2,3 -> 0,1,2

        # Random crop, though identical for image and gt
        i, j, h, w = transforms.RandomCrop.get_params(im, output_size=(224, 224))
        im = transforms.functional.crop(im, i, j, h, w)
        im_gt = transforms.functional.crop(im_gt, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            im = torch.flip(im, dims=[2])
            im_gt = torch.flip(im_gt, dims=[1])

        return im, im_gt

    def __len__(self):
        return len(self.imgs)

