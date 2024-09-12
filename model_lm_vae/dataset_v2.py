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
from .utils import FACEMESH_ROI_IDX, extract_llf_features, ROI
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
import torch.nn.functional as F

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
        self.n_frames = n_frames
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
            
            lm_folder = os.path.join(lm_p, line)
            vs_folder = os.path.join(visual_p, line)
            
            all_datas.append((lm_folder, vs_folder))
                
        return all_datas
        
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def __len__(self):
        return len(self.all_datas)
        # return 1

    def __getitem__(self, idx):        
        (lm_folder, vs_folder) = self.all_datas[idx]

        #landmark
        lm_paths = sorted(os.listdir(lm_folder))
        
        if self.train:
            random.seed(None)
        else:
            random.seed(0)
        random.shuffle(lm_paths)
                            
        for i in range(len(lm_paths)):
            if os.path.exists(os.path.join(vs_folder,lm_paths[i])):
                #landmark
                with open(os.path.join(lm_folder,lm_paths[i]), "r") as f:
                    face_landmarks = json.load(f)
                    landmarks = [landmark_pb2.NormalizedLandmark(
                        x=point[0] / 256, 
                        y=point[1] / 256, 
                        z=0.0) for point in face_landmarks]
                    # rect = mp.tasks.components.containers.NormalizedRect(x_center=128, y_center=128, width=256, height=256)
                    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)
                    image_lm = np.zeros((256, 256, 3), dtype=np.uint8)
                    mp_drawing.draw_landmarks(
                        image=image_lm,   
                        landmark_list=face_landmarks_proto,
                        connections= ROI,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(255,255,255)))
                    
                    image_lm = torch.from_numpy(image_lm).float() / 255.0
                    image_lm = image_lm.permute(2,0,1).unsqueeze(0) #(1,3,256,256)
                                        
                    # image_lm = F.interpolate(image_lm, size=(32, 32), mode='bilinear', align_corners=False)
                    
                    # from PIL import Image
                    # import torchvision.transforms as transforms
                    # to_pil = transforms.ToPILImage()
                    # image_save = image_lm.squeeze(0).byte() * 255#.squeeze(0)[0]
                    # image_save = to_pil(image_save)
                    # image_save.save(f'./assets/samples/M003/samples_lm_vae/test.jpg')   
                    
                    # mask = torch.zeros(1,1,32,32)
                    # mask[:, :, 2*8:3*8, 1*8:3*8] = 1
                    # image_lm = image_lm[:,0:1,...]
                    # image_lm = image_lm * mask    
                    

                #img_feature
                with open(os.path.join(vs_folder,lm_paths[i]), "r") as f:
                    img_feature = json.load(f)
                gt_img_feature = torch.FloatTensor(img_feature) #(1,4,32,32)
                # img_feature = gt_img_feature.clone()
                # img_feature[:, :, 2*8:3*8, 1*8:3*8] = 1
                
                # combined = torch.cat((img_feature, image_lm), dim=1)
                
                return (image_lm, gt_img_feature)
        return None

    def collate_fn(self, batch):
        batch_landmark, batch_gt_img_feature = zip(*batch)
        keep_ids = [idx for idx, (_, _) in enumerate(zip(batch_landmark, batch_gt_img_feature))]
            
        if not all(img is None for img in batch_landmark):
            batch_landmark = [batch_landmark[idx] for idx in keep_ids]
            batch_landmark = torch.cat(batch_landmark, dim=0)
        else:
            batch_landmark = None
        
        if not all(img is None for img in batch_gt_img_feature):
            batch_gt_img_feature = [batch_gt_img_feature[idx] for idx in keep_ids]
            batch_gt_img_feature = torch.cat(batch_gt_img_feature, dim=0)
        else:
            batch_gt_img_feature = None
            
        return batch_landmark, batch_gt_img_feature
