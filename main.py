from sklearn.decomposition import PCA
import torch
from os.path import join as pjoin
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import os
import imp
import math
import scipy.misc
import scipy.io
from model.VAE_fc6 import VAE_fc6
from brain.load_vim1 import load_stimuli, load_voxels
from torchvision import transforms

#=================================
# Define params and path
#=================================

file_path = '/nfs/e3/natural_vision/vim1/data_set_vim1/'
net_name = 'fc6'
batch_size = 64
train_size = 1664 # the training size that we use now

caffenet_path = 'model/pytorch_caffenet.pth'
generator_path = f'model/VAE_{net_name}_state_dict.pth'

#=================================
# Load stimulus and fMRI activity
#=================================

# get stim data
_, stim, _ = load_stimuli(file_path, npx=227, npc=3)
stim = stim.transpose()

# get brain info
subject = 'S1'
voxel_data, voxel_idx = load_voxels(file_path, subject, ROI=['V1', 'V2', 'V3', 'V3A', 'V3B', 'V4', 'LO']) 

# split train and test set 
stim_train = stim[:train_size, :]
stin_test = stim[-batch_size:, :]
voxel_train = voxel_data[:train_size, :]
voxel_test = voxel_data[-batch_size:, :]

#=================================
# Load generative model
#=================================

#initialize the caffenet and generator
MainModel = imp.load_source('MainModel', "model/pytorch_caffenet.py")
caffenet = torch.load(caffenet_path)
caffenet.eval()

generator = eval(f'VAE_{net_name}')()
generator.load_state_dict(torch.load(generator_path))
generator.eval()


# define transform 
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])])

stim_transform = torch.zeros(0)
for idx in range(stim.shape[0]):
    img_tmp = stim[idx, :].transpose(1,2,0)
    img_tmp = transform(img_tmp)
    img_tmp = torch.unsqueeze(img_tmp, 0)
    stim_transform = torch.cat((stim_transform, img_tmp))
    print(f'Finish transforming {idx+1} images now')

# run caffenet to extract the features
caffenet.set_feature_name(net_name)
fc_feature = caffenet(stim_transform)
del caffenet

#============================================
# Map between fMRI activity and latent space
#============================================








#============================================
# Evalution happens here
#============================================






