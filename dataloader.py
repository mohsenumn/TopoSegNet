import torch
import cv2
import os, glob, sys
import numpy as np
# from PIL import Image
from os.path import join as pjoin
from torchvision import transforms
from torch.utils import data
from skimage import io
import pdb
import SimpleITK as sitk

class DRIVE(data.Dataset):
    def __init__(self, listpath, folderpaths, task, crop_size):

        self.listpath = listpath
        self.imgfolder = folderpaths[0]
        self.gtfolder = folderpaths[1]
        self.crop_size = crop_size

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['label'] = []
        self.dataCPU['filename'] = []

        self.task = task
 
        self.to_tensor = transforms.ToTensor() # converts HWC in [0,255] to CHW in [0,1]

        self.loadCPU()

    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]
    
        for i, entry in enumerate(mylist):
            filename = entry
    
            if self.task == "test":
                im_path = pjoin(self.imgfolder, filename)
            else:
                im_path = pjoin(self.imgfolder, filename)
    
            gt_path = pjoin(self.gtfolder, filename)
            img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    
            # Check if gt has more than one channel (m > 1), and use only the second channel
            if len(gt.shape) == 3 and gt.shape[2] > 1:
                gt = gt[:, :, 1]
    
            img = self.to_tensor(img)
            gt = self.to_tensor(gt)
    
            for j in range(img.shape[0]):
                meanval = img[j].mean()
                stdval = img[j].std()
                img[j] = (img[j] - meanval) / stdval

            self.dataCPU['image'].append(img)
            self.dataCPU['label'].append(gt)
            self.dataCPU['filename'].append(filename)

    def __len__(self): # total number of 2D slices
        return len(self.dataCPU['filename'])

    def __getitem__(self, index): # select random crop and return CHW torch tensor

        torch_img = self.dataCPU['image'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        if self.task == "train":
            # crop: compute top-left corner first
            _, H, W = torch_img.shape
            corner_h = np.random.randint(low=0, high=H-self.crop_size)
            corner_w = np.random.randint(low=0, high=W-self.crop_size)

            torch_img = torch_img[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]
            torch_gt = torch_gt[:, corner_h:corner_h+self.crop_size, corner_w:corner_w+self.crop_size]

        return torch_img, torch_gt, self.dataCPU['filename'][index]

