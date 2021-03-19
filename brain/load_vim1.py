import numpy as np
import h5py
import scipy.io as sio
from glob import glob
import PIL.Image as pim
import pickle



def load_stimuli(path, load_hirez=True, npx=500, npc=1):
    stimuli_lowrez = sio.loadmat(path + "Stimuli.mat") ### this loads a low-rez version of the stimuli
    trn_size = len(stimuli_lowrez["stimTrn"])
    val_size = len(stimuli_lowrez["stimVal"])
    data_size = trn_size + val_size
    print(f'trn:{trn_size}, val:{val_size}')
    if load_hirez is not True:
        return np.concatenate([stimuli_lowrez["stimTrn"], stimuli_lowrez["stimVal"]], axis=0), trn_size
    ###
    train_stim_files = glob(path+"Stimuli_Trn_FullRes*.mat")
    val_stim_file = path+"Stimuli_Val_FullRes.mat"
    ##load validation stim
    val_h5 = h5py.File(val_stim_file,'r')
    val_stimuli_hirez = np.transpose(val_h5['stimVal'][:],[2,1,0]).astype(np.float32)
    val_h5.close()
    ##allocate memory for stim
    hirez_resolution = val_stimuli_hirez.shape[1:3]
    trn_stimuli_hirez = np.zeros((trn_size,)+hirez_resolution, dtype=np.float32)
    ##load training stim
    cnt = 0
    for sl in sorted(train_stim_files):
        this_h5 = h5py.File(sl,'r')
        this_train_stim = this_h5['stimTrn']
        this_num_stim = this_train_stim.shape[-1]
        trn_stimuli_hirez[cnt:cnt+this_num_stim,:,:] = np.transpose(this_train_stim[:],[2,1,0])
        cnt += this_num_stim
        this_h5.close()

    assert npc==1 or npc==3, "Invalid color chanel values. Either 1 or 3."
    mode = 'RGB' if npc==3 else 'L'   
    stimuli_hirez = np.ndarray(shape=(data_size, npx, npx, npc), dtype=np.float32)
    for i,rawim in enumerate(trn_stimuli_hirez):
        rawmin, rawmax = np.min(rawim), np.max(rawim)
        sim = (rawim - rawmin) * 255 / (rawmax - rawmin)               
        im = pim.fromarray(sim, mode='F').resize((npx, npx), resample=pim.BILINEAR).convert(mode) 
        if npc==3:
            stimuli_hirez[i,...] = np.asarray(im)
        else:
            stimuli_hirez[i,:,:,0] = np.asarray(im)  

    for i,rawim in enumerate(val_stimuli_hirez):
        rawmin, rawmax = np.min(rawim), np.max(rawim)
        sim = (rawim - rawmin) * 255 / (rawmax - rawmin)         
        im = pim.fromarray(sim, mode='F').resize((npx, npx), resample=pim.BILINEAR).convert(mode)
        if npc==3:
            stimuli_hirez[trn_size+i,...] = np.asarray(im)
        else:
            stimuli_hirez[trn_size+i,:,:,0] = np.asarray(im)  

    stimuli_hirez = np.transpose(stimuli_hirez, (0,3,1,2))
    stimuli_hirez = stimuli_hirez.astype(np.uint8)
    
    print(f'Data shape = {stimuli_hirez.shape}')
    return np.concatenate([stimuli_lowrez["stimTrn"], stimuli_lowrez["stimVal"]], axis=0), stimuli_hirez, trn_size




def load_voxels(path, subject, ROI='all', voxel_subset=None):
    """
    Parameters:
    -----------
    ROI[str/list]:
        default is str, 'all' means using all the ROIs
        when using list, its elements should be names of ROI in
        'other, 'V1', 'V2', 'V3', 'V3A', 'V3B', 'V4', 'LO'
    """
    voxelset = h5py.File(path+"EstimatedResponses.mat", 'r')
    ROI_all = {'other':0, 'V1':1, 'V2':2, 'V3':3, 'V3A':4, 'V3B':5, 'V4':6, 'LO':7}
    
    voxeldata = np.concatenate([voxelset['dataTrn%s'%subject], voxelset['dataVal%s'%subject]], axis=0).astype(dtype=np.float32)
    voxelroi = np.array(voxelset['roi%s'%subject]).flatten()
    voxelidx = np.array(voxelset['voxIdx%s'%subject]).flatten()
    
    voxelNanMask = ~np.isnan(voxeldata).any(axis=0)
    roiMask = True
    if ROI != 'all':
        roiMask = False
        for roi in ROI:
            roiMask = roiMask | (voxelroi==ROI_all[roi])
    Mask = voxelNanMask & roiMask
    nv = np.sum(Mask)
    print(f'{nv} voxels are selected, including:')
    Mask = Mask.tolist()
    
    voxel_data = voxeldata[:, Mask].astype(dtype=np.float32)
    voxelROI  = voxelroi[Mask]
    voxelIDX  = voxelidx[Mask]
    
    roi_info = f''
    for roi in ROI_all.keys():
        num = np.sum(voxelROI==ROI_all[roi])
        if num != 0:
            roi_info += f'{roi}:{num} voxels\n'
    print(roi_info)
    
    if voxel_subset is not None:
        voxel_data = voxel_data[:, voxel_subset]

    return voxel_data, voxelIDX