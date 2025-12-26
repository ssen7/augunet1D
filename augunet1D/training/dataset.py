import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import h5py
from tqdm import tqdm
import torchaudio
import pytorch_lightning as pl

from ecg_augmentations.ecg_augmentations import *
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os
from torchvision.transforms import Compose
import random
# from config import settings

class MiceData(Dataset):

    def __init__(self, df_path:str, dtype:str='train'):
        super().__init__()

        self.df_path=df_path
        self.dtype=dtype

        df = pd.read_csv(self.df_path)
        self.df = df[df['dtype']==self.dtype]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        npz_path=self.df.iloc[index]['sample_paths']

        with np.load(npz_path, 'rb') as data:
            label=torch.LongTensor(data['arr_0'])
            ecog=torch.FloatTensor(data['arr_1']).unsqueeze(0)
        
        return label, ecog
        
        

class MiceDataRaw(Dataset):

    def __init__(self, df_path:str, dtype:str='train', split_size=1000, transform=None):
        super().__init__()

        self.df_path=df_path
        self.dtype=dtype
        self.split_size= split_size
        self.transform=transform

        df = pd.read_csv(self.df_path)
        self.df = df[df['dtype']==self.dtype]

        self.ecog_list=[]
        self.label_list=[]

        for file in tqdm(self.df.files, desc='Segmenting raw data'):

            label, ecog, _, _ = self.read_mat_file(file)
            self.ecog_list+=self.split_given_size(ecog, split_size)         
            self.label_list+=self.split_label_given_size(label, split_size)


    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, index):
        label=torch.tensor(self.label_list[index], dtype=torch.int64)
        ecog=torch.FloatTensor(self.ecog_list[index]).unsqueeze(0)
        
        if self.transform is not None:
            ecog = self.transform(ecog)
        
        return label, ecog
    
    def read_mat_file(self, filepath):
        with h5py.File(filepath, 'r+') as f:
            label=np.array(f['rec']['SWDlabel'])[0]        
            ecog=np.array(f['rec']['ecog'])[0]
            fs=np.array(f['rec']['fs'])[0][0]
            mid=np.array(f['rec']['mID'])[0][0]
            file_name=filepath.split('/')[-1].split('.')[0]
            # print(f'Dataset size: {len(label)} at frequency {fs}Hz for MID {file_name}')
            # scale ecog
            ecog = (ecog - np.min(ecog))/(np.max(ecog)-np.min(ecog))
    
        return label, ecog, fs, mid
    
    def split_given_size(self, arr, size):
        splits = np.split(arr, np.arange(size,len(arr),size))
        splits=[x for x in splits if len(x)==size]
        return splits
    
    def split_label_given_size(self, arr, size):
        arr_split = self.split_given_size(arr, size)
        segment_labels = [1 if np.sum(x)>500 else 0 for x in arr_split]
        return segment_labels
    
    
class MiceDataSegmented(Dataset):

    def __init__(self, df_path:str, down_freq:int, split_size:int, process_dir:str, dtype:str='train', transform=None,filetype='mat', data_prop=1.0, sample_signal=False, if_convert=False):
        super().__init__()

        self.df_path=df_path
        self.dtype=dtype
        self.split_size= split_size
        self.transform=transform
        self.down_freq=down_freq
        self.filetype=filetype
        self.process_dir=process_dir
        os.makedirs(self.process_dir, exist_ok=True)
        self.data_prop=data_prop
        self.sample_signal=sample_signal
        self.if_convert=if_convert

        df = pd.read_csv(self.df_path)
        self.df = df[df['dtype']==self.dtype].sample(frac=self.data_prop)

        self.ecog_list=[]
        self.label_list=[]
        
        self.signal_list=[]

        for file in tqdm(self.df.files, desc='Segmenting raw data'):
            
            ## if data already processed, read processed data and split
            if os.path.exists(os.path.join(self.process_dir,f'{os.path.basename(file)}_ecogs.pt')):
                ecog_downs=torch.load(os.path.join(self.process_dir,f'{os.path.basename(file)}_ecogs.pt'), weights_only=True)
                label_downs=torch.load(os.path.join(self.process_dir,f'{os.path.basename(file)}_label.pt'), weights_only=True)
                label_downs = torch.where(label_downs > 0.5, 1, 0)
                
                self.ecog_list+=self.split_given_size(ecog_downs[0], self.split_size)         
                self.label_list+=self.split_given_size(label_downs, self.split_size)
                
                continue
            if self.filetype=='mat':
                label, ecog, freq, _ = self.read_mat_file(file)
                if len(label.shape)!=1:
                    label=label.reshape(-1)
                    ecog=ecog.reshape(-1)
                
                self.down_transform = torchaudio.transforms.Resample(orig_freq=freq, new_freq=self.down_freq)
            else:
                label, ecog = self.read_npz_file(file)
                self.down_transform = torchaudio.transforms.Resample(orig_freq=1000, new_freq=self.down_freq) ## TODO: change frequency
            
            ecog_tensor=torch.FloatTensor(ecog).unsqueeze(0)
            label_tensor=torch.FloatTensor(label).unsqueeze(0)
            
            ecog_downs = self.down_transform(ecog_tensor)
            label_downs = self.down_transform(label_tensor)
            label_downs = torch.where(label_downs[0] > 0.5, 1, 0)
            
            ## Save downsampled tensors
            torch.save(f=os.path.join(self.process_dir,f'{os.path.basename(file)}_ecogs.pt'), obj=ecog_downs)
            torch.save(f=os.path.join(self.process_dir,f'{os.path.basename(file)}_label.pt'), obj=label_downs)
    
            self.ecog_list+=self.split_given_size(ecog_downs[0], self.split_size)         
            self.label_list+=self.split_given_size(label_downs, self.split_size)
        
        if (self.dtype=='train') & (self.sample_signal):
            for ecog, label in zip(self.ecog_list,self.label_list):
                if torch.sum(label)>=1:
                    self.signal_list.append([label, ecog])
                    
            
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, index):
        
        label=self.label_list[index]
        ecog=self.ecog_list[index].unsqueeze(0)
        
        if (self.transform is not None):
            if type(self.transform)==list:
                self.transform=Compose(self.transform)
            # print(self.transform)
            ecog=self.transform(ecog)
        
        if (self.dtype=='train') & (self.sample_signal) & (not self.if_convert):
            return label, ecog, random.sample(self.signal_list, k=2)
        elif (self.if_convert):
            label = self.convert_labels(label)
            return label, ecog 
        else:
            return label,ecog
        
    def convert_labels(self, labels):
        """Converts list of labels to single tensor if proportion of labels is greater than 0.5"""
        # labels = torch.FloatTensor(labels)
        if torch.sum(labels) > 0.5 * len(labels):
            return torch.tensor(1, dtype=torch.int64)
        else:
            return torch.tensor(0, dtype=torch.int64)
            
    def read_mat_file(self, filepath):
        with h5py.File(filepath, 'r+') as f:
            if len(np.array(f['rec']['SWDlabel'])[0])==1:
                label=np.array(f['rec']['SWDlabel'])
                ecog=np.array(f['rec']['ecog'])
            else:
                label=np.array(f['rec']['SWDlabel'])[0]
                ecog=np.array(f['rec']['ecog'])[0]
            fs=np.array(f['rec']['fs'])[0][0]
            mid=np.array(f['rec']['mID'])[0][0]
            file_name=filepath.split('/')[-1].split('.')[0]
            # scale ecog
            ecog = (ecog - np.min(ecog))/(np.max(ecog)-np.min(ecog))

        return label, ecog, fs, mid
    
    def split_given_size(self, arr, size):
        splits = np.split(arr, np.arange(size,len(arr),size))
        splits=[x for x in splits if len(x)==size]
        return splits
    
    def split_label_given_size(self, arr, size):
        arr_split = self.split_given_size(arr, size)
        segment_labels = [1 if np.sum(x)>500 else 0 for x in arr_split]
        return segment_labels
    
    def read_npz_file(self, filepath):
        npz = np.load(filepath)
        
        label=npz['res_label']
        ecog=npz['res_ecog']
        
        ecog = (ecog - np.min(ecog))/(np.max(ecog)-np.min(ecog))
        
        return label, ecog
    
class MiceDataSegmented_mod_pandas(Dataset):

    def __init__(self, df:pd.DataFrame, down_freq:int, split_size:int, process_dir:str, dtype:str='train', transform=None,filetype='mat', data_prop=1.0, sample_signal=False, if_convert=False):
        super().__init__()

        self.df=df
        self.dtype=dtype
        self.split_size= split_size
        self.transform=transform
        self.down_freq=down_freq
        self.filetype=filetype
        self.process_dir=process_dir
        os.makedirs(self.process_dir, exist_ok=True)
        self.data_prop=data_prop
        self.sample_signal=sample_signal
        self.if_convert=if_convert

        # df = pd.read_csv(self.df_path)
        # self.df = df[df['dtype']==self.dtype].sample(frac=self.data_prop)
        
        self.ecog_list=[]
        self.label_list=[]
        
        self.signal_list=[]

        for file in tqdm(self.df.files, desc='Segmenting raw data'):
            
            ## if data already processed, read processed data and split
            if os.path.exists(os.path.join(self.process_dir,f'{os.path.basename(file)}_ecogs.pt')):
                ecog_downs=torch.load(os.path.join(self.process_dir,f'{os.path.basename(file)}_ecogs.pt'), weights_only=True)
                label_downs=torch.load(os.path.join(self.process_dir,f'{os.path.basename(file)}_label.pt'), weights_only=True)
                label_downs = torch.where(label_downs > 0.5, 1, 0)
                
                self.ecog_list+=self.split_given_size(ecog_downs[0], self.split_size)         
                self.label_list+=self.split_given_size(label_downs, self.split_size)
                
                continue
            if self.filetype=='mat':
                label, ecog, freq, _ = self.read_mat_file(file)
                if len(label.shape)!=1:
                    label=label.reshape(-1)
                    ecog=ecog.reshape(-1)
                
                self.down_transform = torchaudio.transforms.Resample(orig_freq=freq, new_freq=self.down_freq)
            else:
                label, ecog = self.read_npz_file(file)
                self.down_transform = torchaudio.transforms.Resample(orig_freq=1000, new_freq=self.down_freq) ## TODO: change frequency
            
            ecog_tensor=torch.FloatTensor(ecog).unsqueeze(0)
            label_tensor=torch.FloatTensor(label).unsqueeze(0)
            
            ecog_downs = self.down_transform(ecog_tensor)
            label_downs = self.down_transform(label_tensor)
            label_downs = torch.where(label_downs[0] > 0.5, 1, 0)
            
            ## Save downsampled tensors
            torch.save(f=os.path.join(self.process_dir,f'{os.path.basename(file)}_ecogs.pt'), obj=ecog_downs)
            torch.save(f=os.path.join(self.process_dir,f'{os.path.basename(file)}_label.pt'), obj=label_downs)
    
            self.ecog_list+=self.split_given_size(ecog_downs[0], self.split_size)         
            self.label_list+=self.split_given_size(label_downs, self.split_size)
        
        if (self.dtype=='train') & (self.sample_signal):
            for ecog, label in zip(self.ecog_list,self.label_list):
                if torch.sum(label)>=1:
                    self.signal_list.append([label, ecog])
                    
            
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, index):
        
        label=self.label_list[index]
        ecog=self.ecog_list[index].unsqueeze(0)
        
        if (self.transform is not None):
            if type(self.transform)==list:
                self.transform=Compose(self.transform)
            # print(self.transform)
            ecog=self.transform(ecog)
        
        if (self.dtype=='train') & (self.sample_signal) & (not self.if_convert):
            return label, ecog, random.sample(self.signal_list, k=2)
        elif (self.if_convert):
            label = self.convert_labels(label)
            return label, ecog 
        else:
            return label,ecog
        
    def convert_labels(self, labels):
        """Converts list of labels to single tensor if proportion of labels is greater than 0.5"""
        # labels = torch.FloatTensor(labels)
        if torch.sum(labels) > 0.5 * len(labels):
            return torch.tensor(1, dtype=torch.int64)
        else:
            return torch.tensor(0, dtype=torch.int64)
            
    def read_mat_file(self, filepath):
        with h5py.File(filepath, 'r+') as f:
            if len(np.array(f['rec']['SWDlabel'])[0])==1:
                label=np.array(f['rec']['SWDlabel'])
                ecog=np.array(f['rec']['ecog'])
            else:
                label=np.array(f['rec']['SWDlabel'])[0]
                ecog=np.array(f['rec']['ecog'])[0]
            fs=np.array(f['rec']['fs'])[0][0]
            mid=np.array(f['rec']['mID'])[0][0]
            file_name=filepath.split('/')[-1].split('.')[0]
            # scale ecog
            ecog = (ecog - np.min(ecog))/(np.max(ecog)-np.min(ecog))

        return label, ecog, fs, mid
    
    def split_given_size(self, arr, size):
        splits = np.split(arr, np.arange(size,len(arr),size))
        splits=[x for x in splits if len(x)==size]
        return splits
    
    def split_label_given_size(self, arr, size):
        arr_split = self.split_given_size(arr, size)
        segment_labels = [1 if np.sum(x)>500 else 0 for x in arr_split]
        return segment_labels
    
    def read_npz_file(self, filepath):
        npz = np.load(filepath)
        
        label=npz['res_label']
        ecog=npz['res_ecog']
        
        ecog = (ecog - np.min(ecog))/(np.max(ecog)-np.min(ecog))
        
        return label, ecog


class MiceDataSegmentedV2(Dataset):

    def __init__(self, df_path:str, dtype:str='train', transform=None):
        super().__init__()

        self.df_path=df_path
        self.dtype=dtype
        self.transform=transform
        
        df = pd.read_csv(self.df_path)
        self.df = df[df['dtype']==self.dtype]
        self.ecog_list = []
        self.label_list = []
        self.ecog_list +=[torch.load(x, weights_only=True) for x in self.df.ecog]
        self.label_list +=[torch.load(x, weights_only=True) for x in self.df.label]
        
        self.ecog_list = self.flatten(self.ecog_list)
        self.label_list = self.flatten(self.label_list)
            
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, index):
        
        label=self.label_list[index]
        ecog=self.ecog_list[index].unsqueeze(0)
        
        if (self.transform is not None):
            ecog=self.transform(ecog)
        
        return label, ecog
    
    def flatten(self, xss):
        return [x for xs in xss for x in xs]
    

class MiceDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config=config
        # print(f'This is configs : {settings.seed}')
        self.generator = torch.Generator().manual_seed(config['seed'])
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.transform = transforms.Compose(
        [
            RandomApply([Scale()], p=0.5),
            RandomApply([Permute()], p=0.6),
            GaussianNoise(max_snr=0.005),
            RandomApply([Invert()], p=0.2),
            RandomApply([Reverse()], p=0.2),
            RandomApply([TimeWarp()], p=0.2),
            RandomApply([RandMask()], p=0.2),
        ]
        )

    def setup(self, stage:str):
        if stage == "fit":
            train_dataset=MiceDataSegmented(self.config['df_path'], orig_freq=self.config['orig_freq'], new_freq=self.config['new_freq'], split_size=self.config['split_size'], dtype='train', transform=self.transform)
            self.train_dataset, self.val_dataset=torch.utils.data.random_split(train_dataset, [0.95, 0.05], generator=self.generator)
        
        if stage == "test":
            self.testds = MiceDataSegmented(self.config['df_path'], orig_freq=400, new_freq=100, split_size=self.config['split_size'], dtype='test')
            

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testds, batch_size=self.batch_size, num_workers=self.num_workers)
    

class MiceDataModuleV2(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config=config
        # print(f'This is configs : {settings.seed}')
        self.generator = torch.Generator().manual_seed(config['seed'])
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.transform = transforms.Compose(
        [
            RandomApply([Scale()], p=0.5),
            RandomApply([Permute()], p=0.6),
            GaussianNoise(max_snr=0.005),
            RandomApply([Invert()], p=0.2),
            RandomApply([Reverse()], p=0.2),
            RandomApply([TimeWarp()], p=0.2),
            RandomApply([RandMask()], p=0.2),
        ]
        )

    def setup(self, stage:str):
        if stage == "fit":
            train_dataset=MiceDataSegmentedV2(self.config['df_path'], dtype='train', transform=self.transform)
            self.train_dataset, self.val_dataset=torch.utils.data.random_split(train_dataset, [0.95, 0.05], generator=self.generator)
        
        if stage == "test":
            self.testds = MiceDataSegmentedV2(self.config['df_path'], dtype='test')
            

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testds, batch_size=self.batch_size, num_workers=self.num_workers)
    
# if __name__=='__main__':
#     print(settings.seed)

def custom_collate_fn(batch):
    
    labels = [item[0] for item in batch]
    ecogs = [item[1] for item in batch]
    signals = [item[2] for item in batch]
    
    ex_label1=[x[0][0] for x in signals]
    ex_label2=[x[1][0] for x in signals]
    ex_ecog1=[x[0][1].unsqueeze(0) for x in signals]
    ex_ecog2=[x[1][1].unsqueeze(0) for x in signals]

    labels=labels+random.sample(ex_label1,k=1)+ random.sample(ex_label2, k=1)
    ecogs=ecogs+random.sample(ex_ecog1,k=1)+ random.sample(ex_ecog2, k=1)
    
    ecogs=torch.stack(ecogs)
    # print(ecogs.shape)
    labels = torch.stack(labels)
    # print(labels.shape)

    return labels, ecogs

def custom_collate_fn_V2(batch):
    
    labels = [item[0] for item in batch]
    ecogs = [item[1] for item in batch]
    signals = [item[2] for item in batch]
    
    ex_label1=[x[0][0] for x in signals]
    ex_label2=[x[1][0] for x in signals]
    ex_ecog1=[x[0][1].unsqueeze(0) for x in signals]
    ex_ecog2=[x[1][1].unsqueeze(0) for x in signals]

    labels=labels+ex_label1+ex_label2
    ecogs=ecogs+ex_ecog1+ex_ecog2
    
    ecogs=torch.stack(ecogs)
    # print(ecogs.shape)
    labels = torch.stack(labels)
    # print(labels.shape)

    return labels, ecogs