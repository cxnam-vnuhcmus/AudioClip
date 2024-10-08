import json
from tqdm import tqdm
import torch
import torch.utils.data as td
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import numpy as np
import random
import cv2
import librosa
from PIL import Image
import torchvision.transforms as T
from .utils import FACEMESH_ROI_IDX, extract_llf_features

class Dataset(td.Dataset):

    def __init__(self, 
                 data_root: str,
                 data_file: str,
                 n_folders: int,
                 audio_dataroot: str,
                 visual_dataroot: str,
                 transcript_dataroot: str, 
                 lm_dataroot: str,
                 fps: int,
                 img_size: int,
                 sample_rate: int,
                 n_mels: int,
                 n_fft: int,
                 win_length: int,
                 hop_length: int,
                 n_frames: int,
                 train: bool,
                 **_
                 ):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.data_file = data_file
        self.audio_dataroot = os.path.join(self.data_root, audio_dataroot)
        self.visual_dataroot = os.path.join(self.data_root, visual_dataroot)
        self.transcript_dataroot = os.path.join(self.data_root, transcript_dataroot)
        self.lm_dataroot = os.path.join(self.data_root, lm_dataroot)
        self.fps = fps
        self.img_size = img_size
        self.sr = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_frames = n_frames + 1
        self.train = train    
        
        self.transform = T.Compose([
            T.Resize((self.img_size , self.img_size )),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])   
        self.inv_normalize = T.Compose([
            T.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0/0.5, 1.0/0.5, 1.0/0.5]),
            T.ToPILImage()
        ]) 
        
        
        if os.path.isdir(self.data_file):
            persons = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.data_file))][:n_folders]
            data_path = os.path.join(self.data_file,'{p}.txt')
        else:
            persons, _ = os.path.splitext(os.path.basename(self.data_file))   
            persons = [persons] 
            data_path = self.data_file
        
        filelists = []
        for p in tqdm(persons, total=len(persons)):
            data_path_p = data_path.format(p=p)
            with open(data_path_p, 'r') as file:
                for line in file:
                    line = line.strip()
                    filelists.append(f"{p}\t{line}")
        
        random.seed(0)
        random.shuffle(filelists)
        if self.train:
            filelists = filelists[:int(len(filelists) * 0.9)]
        else:
            filelists = filelists[int(len(filelists) * 0.9):] 
            self.n_frames = self.n_frames * 2 - 1
                    
        self.all_datas = self.data_augmentation(filelists)
    
    def data_augmentation(self, filelists):
        all_datas = []
        for fileline in tqdm(filelists, desc="Loading datas"):
            p,line = fileline.strip().split("\t")
            audio_p = self.audio_dataroot.format(p=p)
            visual_p = self.visual_dataroot.format(p=p)
            lm_p = self.lm_dataroot.format(p=p)
            transcript_p = self.transcript_dataroot.format(p=p)
            
            audio_name = os.path.join(audio_p, f'{line}.json')
            lm_folder = os.path.join(lm_p, line)
            
            # all_datas.append((audio_name, lm_folder))
            #segment
            lm_paths = sorted(os.listdir(lm_folder))
            for i in range(0, len(lm_paths) - self.n_frames, 1):
                is_segment = True
                for j in range(self.n_frames):
                    if (i + j) < len(lm_paths):
                        if not os.path.exists(os.path.join(lm_folder,lm_paths[i + j])):
                            is_segment = False
                            continue
                if is_segment:
                    all_datas.append((audio_name, lm_folder, i))
                
        return all_datas
        
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def __len__(self):
        return len(self.all_datas)
        # return 32

    def __getitem__(self, idx):        
        while True:
            (audio_name, lm_folder, segment_start_idx) = self.all_datas[idx]

            #audio
            with open(audio_name, "r") as f:
                data = json.load(f)
            mel_spectrogram_db = torch.tensor(data["mfcc"])
            # llfs = torch.tensor(data["llfs"])
            
            #landmark
            lm_paths = sorted(os.listdir(lm_folder))
                        
            lm_data_list = []
            for i in range(segment_start_idx, segment_start_idx + self.n_frames):
                with open(os.path.join(lm_folder,lm_paths[i]), "r") as f:
                    lm_data = json.load(f)
                    lm_roi = []
                    for i in FACEMESH_ROI_IDX:
                        lm_roi.append(lm_data[i])
                    lm_roi = np.asarray(lm_roi)
                    lm_roi = torch.FloatTensor(lm_roi)
                    lm_roi = lm_roi / 256.0
                    lm_data_list.append(lm_roi)
            lm_data_list = np.stack(lm_data_list, axis=0)
            lm_data_list = torch.tensor(lm_data_list) #(N, lm_points, 2)
            
            mel_segment = mel_spectrogram_db[segment_start_idx:segment_start_idx + self.n_frames, :] #(N, 80)
            break
        
        return (mel_segment, lm_data_list, os.path.join(lm_folder,lm_paths[segment_start_idx + (self.n_frames + 1)//2-1]))

    def collate_fn(self, batch):
        batch_audio, batch_landmark, lm_paths = zip(*batch)
        keep_ids = [idx for idx, (_, _) in enumerate(zip(batch_audio, batch_landmark))]
            
        if not all(au is None for au in batch_audio):
            batch_audio = [batch_audio[idx] for idx in keep_ids]
            batch_audio = torch.stack(batch_audio)
        else:
            batch_audio = None
            
        if not all(img is None for img in batch_landmark):
            batch_landmark = [batch_landmark[idx] for idx in keep_ids]
            batch_landmark = torch.stack(batch_landmark, dim=0)
        else:
            batch_landmark = None
        
        if not all(img is None for img in lm_paths):
            lm_paths = [lm_paths[idx] for idx in keep_ids]
        else:
            lm_paths = None
        
        return batch_audio, batch_landmark, lm_paths
