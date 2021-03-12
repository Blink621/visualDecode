from load_data import load_stimuli, load_voxels
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
#import patchShow
import scipy.misc
import scipy.io
from VAE_fc6 import VAE_fc6

#=================================
# Define params and path
#=================================

file_path = '/nfs/e3/natural_vision/vim1/data_set_vim1/'



#=================================
# Load stimulus and fMRI activity
#=================================

# get stim data
stimuli_lowrez, stimuli_hirez, trn_size = load_stimuli(file_path, npx=256, npc=3)

# get brain info
subject = 'S1'
voxel_data, voxel_idx = load_voxels(file_path, subject,ROI=['V1', 'V2', 'V3', 'V3A', 'V3B', 'V4', 'LO']) #voxel_subset=range(3400, 3700))
voxel_train = voxel_data[:real_train_num, :]
voxel_test = voxel_data[trn_size:trn_size+batch_size, :]

#=================================
# Load generative model
#=================================

#initialize the caffenet to extract the features
MainModel = imp.load_source('MainModel', "release_deepsim_v0.5/trained_models/caffenet/pytorch_caffenet.py")
caffenet = torch.load(caffenet_path)
caffenet.eval()

# run caffenet and extract the features
caffenet.set_feature_name(net_name)

generator = eval(f'VAE_{net_name}')()
generator.load_state_dict(torch.load(generator_path))
generator.eval()
generated_tensor = generator(latent_space_test)
generated = generated_tensor.detach().numpy()

#============================================
# Map between fMRI activity and latent space
#============================================





#============================================
# Evalution happens here
#============================================






