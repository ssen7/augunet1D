import h5py
import numpy as np
import torchaudio
import torch

def sigmoid(x):return 1.0 / (1.0 + np.exp(-x))

DOWN_TRANSFORM = torchaudio.transforms.Resample(orig_freq=400, new_freq=100)
SPLIT_SIZE=2000

def read_process_mat_file(filepath):
    label, ecog, _, _ = read_mat_file(filepath)
    ecog_list=[]
    label_list=[]

    ecog_tensor=torch.FloatTensor(ecog).unsqueeze(0)
    label_tensor=torch.FloatTensor(label).unsqueeze(0)

    ecog_downs = DOWN_TRANSFORM(ecog_tensor)
    label_downs = DOWN_TRANSFORM(label_tensor)
    label_downs = torch.where(label_downs[0] > 0.5, 1, 0)

    ecog_list+=split_given_size(ecog_downs[0], SPLIT_SIZE)
    label_list+=split_given_size(label_downs, SPLIT_SIZE)
    
    return ecog_list, label_list, ecog_downs, label_downs

def read_mat_file(filepath):
    with h5py.File(filepath, 'r+') as f:
        label=np.array(f['rec']['SWDlabel']).reshape(-1)
        ecog=np.array(f['rec']['ecog']).reshape(-1)
        fs=np.array(f['rec']['fs'])[0][0]
        mid=np.array(f['rec']['mID'])[0][0]
        # file_name=filepath.split('/')[-1].split('.')[0]
        # scale ecog
        ecog = (ecog - np.min(ecog))/(np.max(ecog)-np.min(ecog))

    return label, ecog, fs, mid

def split_given_size(arr, size):
    splits = np.split(arr, np.arange(size,len(arr),size))
    splits=[x for x in splits if len(x)==size]
    return splits


def sigmoid(x):return 1.0 / (1.0 + np.exp(-x))