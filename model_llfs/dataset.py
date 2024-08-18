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

FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])

FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                           (4, 45), (45, 220), (220, 115), (115, 48),
                           (4, 275), (275, 440), (440, 344), (344, 278), ])


ROI =  frozenset().union(*[FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, 
FACEMESH_RIGHT_EYE,FACEMESH_RIGHT_EYEBROW,FACEMESH_FACE_OVAL,FACEMESH_NOSE])            #131 keypoints

FACEMESH_LIPS_IDX = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

SEQUENCE_LIPS_SHAPE_IDX = [164,167,165,92,186,57,43,106,182,83,18,313,406,335,273,287,410,322,391,393]

# Chuyển ROI thành danh sách các chỉ số
def get_indices_from_frozenset(frozenset_data):
    indices = set()
    for pair in frozenset_data:
        indices.add(pair[0])
        indices.add(pair[1])
    return sorted(indices)

FACEMESH_ROI_IDX = get_indices_from_frozenset(ROI)

emotion_labels = ["angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
emotion_to_index = {label: idx for idx, label in enumerate(emotion_labels)}

def emotion_to_one_hot(emotion_label):
    one_hot_vector = np.zeros(len(emotion_labels))
    index = emotion_to_index[emotion_label]
    one_hot_vector[index] = 1
    return one_hot_vector

def extract_llf_features(audio_data, sr, n_fft, win_length, hop_length):
    # Rút trích đặc trưng âm thanh
    # Âm lượng
    rms = librosa.feature.rms(y=audio_data, frame_length=win_length, hop_length=hop_length, center=False)

    # Tần số cơ bản
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)

    # Tần số biên độ
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)

    # Mức độ biến đổi âm lượng và tần số
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    
    #Poly-features
    poly_features = librosa.feature.poly_features(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    
    # Compute zero-crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=audio_data, frame_length=win_length, hop_length=hop_length, center=False)
    
    feats = np.vstack((chroma, #12
                spectral_contrast, #7
                spectral_centroid, #1
                spectral_bandwidth, #1
                spectral_flatness, #1
                spectral_rolloff, #1
                poly_features, #2
                rms, #1
                zcr #1
                )) 
    return feats

class Dataset(td.Dataset):

    def __init__(self, 
                 datalist_root: str,
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
        self.datalist_root = datalist_root
        self.audio_dataroot = audio_dataroot
        self.visual_dataroot = visual_dataroot
        self.transcript_dataroot = transcript_dataroot
        self.lm_dataroot = lm_dataroot
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
        
        
        if os.path.isdir(self.datalist_root):
            persons = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.datalist_root))][:5]
            data_path = os.path.join(self.datalist_root,'{p}.txt')
        else:
            persons, _ = os.path.splitext(os.path.basename(self.datalist_root))   
            persons = [persons] 
            data_path = self.datalist_root
        
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
                    
        self.all_datas = self.data_augmentation(filelists)
    
    def data_augmentation(self, filelists):
        all_datas = []
        for fileline in tqdm(filelists, desc="Loading datas"):
            p,line = fileline.strip().split("\t")
            audio_p = self.audio_dataroot.format(p=p)
            visual_p = self.visual_dataroot.format(p=p)
            lm_p = self.lm_dataroot.format(p=p)
            transcript_p = self.transcript_dataroot.format(p=p)
            
            audio_name = os.path.join(audio_p, f'{line}.wav')
            lm_folder = os.path.join(lm_p, line)
            
            all_datas.append((audio_name, lm_folder))
                
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
            (audio_name, lm_folder) = self.all_datas[idx]

            #audio
            audio_data, _ = librosa.load(audio_name, sr=self.sr)
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, 
                                                            sr=self.sr, 
                                                            n_mels=self.n_mels-27, 
                                                            n_fft=self.n_fft,
                                                            win_length=self.win_length, 
                                                            hop_length=self.hop_length,
                                                            center=False)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)            
            mel_spectrogram_db = torch.tensor(mel_spectrogram_db).T #(length, 80)
            
            llfs = extract_llf_features(audio_data, self.sr, self.n_fft, self.win_length, self.hop_length)
            llfs = torch.tensor(llfs).T

            audio_embedding = torch.cat((mel_spectrogram_db, llfs), dim=1)
            
            
            #landmark
            lm_paths = sorted(os.listdir(lm_folder))
            
            #random segment
            max_len = min(mel_spectrogram_db.shape[0], len(lm_paths))
            if max_len < self.n_frames:
                idx = random.randint(0, len(self.all_datas))
                continue
                
            segment_start_idx = -1
            while(segment_start_idx == -1):
                segment_start_idx = random.randint(0, max_len - self.n_frames)
                for i in range(segment_start_idx, segment_start_idx + self.n_frames):
                    if not os.path.exists(os.path.join(lm_folder,lm_paths[i])):
                        segment_start_idx = -1
                        break
            
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
            
            mel_segment = audio_embedding[segment_start_idx:segment_start_idx + self.n_frames, :] #(N, 80)
            
            break
        
        return (mel_segment, lm_data_list)

    def collate_fn(self, batch):
        batch_audio, batch_landmark = zip(*batch)
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
        
        return batch_audio, batch_landmark
