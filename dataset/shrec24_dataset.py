import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import os.path as opt
from tqdm import tqdm


SHREC24_CONNECTIONS = [(0, 1), (0, 2), (0, 3), (0, 4),
                       (1, 2), (1, 3),
                       (2, 3), (2, 6), (2, 8),
                       (3, 12), (3, 10),
                       (4, 5),
                       (6, 7),
                       (8, 9),
                       (10, 11),
                       (12, 13),
                       (14, 15), (14, 16), (14, 17), (14, 18),
                       (15, 16), (15, 17),
                       (16, 17), (16, 20), (16, 22),
                       (17, 24), (17, 26),
                       (18, 19),
                       (20, 21),
                       (22, 23),
                       (24, 25),
                       (26, 27)]


def get_adjacency_matrix(connections):
    n_lands = len(set([joint for tup in connections for joint in tup]))
    adj = np.zeros((n_lands, n_lands))
    for i, j in connections:
        adj[i][j] = 1.
        adj[j][i] = 1.
    adj += np.eye(adj.shape[0])
    return adj


SHREC24_AM = get_adjacency_matrix(SHREC24_CONNECTIONS)
SHREC24_AM = torch.from_numpy(SHREC24_AM).float()

class PretrainingDataset(Dataset):
    def __init__(self, data_dir, normalize=False):
        
        assert opt.exists(data_dir), 'path {} does not exist'.format(data_dir)
        ## data
        self.data_dir = data_dir
        
        self.label_map = ["Pressing", 
                          "hole", 
                          "Tightening", 
                          "Centering",
                          "Raising",
                          "Smoothing", 
                          "sponge"]
        
        
        self.sequences = []

        self.normalize = normalize
        if normalize:
            min_ = torch.tensor([-27.665333, -150.1919, 0.0])
            max_ = torch.tensor([65.99878, 0.0, 96.903786])
            self.min = min_.unsqueeze(0).float() # joints, coords
            self.max = max_.unsqueeze(0).float() # joints, coords
            
        data = []
        for folder in tqdm(os.listdir(self.data_dir), desc='loading SHREC24 data[Pre-training]....'):
            folder_path = opt.join(self.data_dir, folder)
            for file in os.listdir(folder_path):
                file_path = opt.join(folder_path, file)
                if not file_path.endswith('.txt'):
                    continue
                with open(file_path) as fd:
                    lines = fd.readlines()
                
                skeleton = []
                for line in lines:
                    line = line.split(';')
                    frame, coords = line[0], line[1:-1]
                    coords = np.array(coords).astype(np.float64)
                    coords = np.reshape(coords, (28, 3))
                    skeleton.append(coords)
                
                self.sequences.extend(skeleton)
                
        self.len_data = len(self.sequences)
    
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, index):
        sequence = self.sequences[index]
        sequence = torch.from_numpy(sequence).float()

        if self.normalize:
            sequence = (sequence - self.min) / (self.max - self.min)

        return dict(Sequence=sequence,
                    AM=SHREC24_AM)



class FinetuningDataset(Dataset):
    def __init__(self, data_dir, T, normalize=False):
        
        assert opt.exists(data_dir), 'path {} does not exist'.format(data_dir)
        ## data
        self.data_dir = data_dir
        self.T = T
        
        self.label_map = ["Pressing", 
                          "MakingHole", 
                          "Tightening", 
                          "Centering",
                          "Raising",
                          "Smoothing", 
                          "Sponge"]
        
        self.sequences = []
        self.labels = []
        
        self.normalize = normalize
        if normalize:
            min_ = torch.tensor([-27.665333, -150.1919, 0.0])
            max_ = torch.tensor([65.99878, 0.0, 96.903786])
            self.min = min_.unsqueeze(0).unsqueeze(0).float() # frames, coords, joints
            self.max = max_.unsqueeze(0).unsqueeze(0).float() # frames, coords, joints

        data = []
        for folder in tqdm(sorted(os.listdir(self.data_dir)), desc='loading SHREC24 data [Finetuning]....'):
            folder_path = opt.join(self.data_dir, folder)
            for file in sorted(os.listdir(folder_path)):
                file_path = opt.join(folder_path, file)
                if not file_path.endswith('.txt'):
                    continue
                with open(file_path) as fd:
                    lines = fd.readlines()
                
                skeleton = []
                for line in lines:
                    line = line.split(';')
                    frame, coords = line[0], line[1:-1]
                    coords = np.array(coords).astype(np.float64)
                    coords = np.reshape(coords, (28, 3))
                    skeleton.append(coords)
                skeleton = np.array(skeleton).astype(np.float64)
                
                self.sequences.append(skeleton)
                self.labels.append(self.label_map.index(folder))
                
        self.len_data = len(self.sequences)
    
    def __len__(self):
        return self.len_data

    def pad_sequence(self, seq, target_length):
        seq_len = seq.shape[0]
        if seq_len >= target_length:
            return seq[:target_length]
        pad_len = target_length - seq_len
        padding = torch.zeros((pad_len, *seq.shape[1:]), dtype=seq.dtype)
        return torch.cat([seq, padding], dim=0)
    
    def __getitem__(self, index):
        
        sequence = self.sequences[index]
        label = self.labels[index]
        
        T = len(sequence)
        sequence = torch.from_numpy(sequence).float()
        
        if self.normalize:
            sequence = (sequence - self.min) / (self.max - self.min)
        
        sequence = self.pad_sequence(sequence, self.T)
        label = torch.tensor(label).type(torch.long)
        
        return dict(Sequence=sequence,
                    Label=label,
                    AM=SHREC24_AM,
                    orig_T=T)


if __name__ == '__main__':

    print('Pretraining dataset:')
    dataset = PretrainingDataset(data_dir='./Data Split/Train-set/', normalize=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    print('Batch shape:', batch.shape)

    print('\nFinetuning dataset:')
    dataset = FinetuningDataset(data_dir='./Data Split/Train-set/', T=3000, normalize=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    print('Sequence shape:', batch['Sequence'].shape)
    print('Labels        :', batch['Label'])