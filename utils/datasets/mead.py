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
FACEMESH_LIPS_IDX = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

class MEAD(td.Dataset):

    def __init__(self, 
                 audio_dataroot: str, 
                 visual_dataroot: str, 
                 lm_dataroot: str,
                 num_frames: int,
                 fps: int,
                 sample_rate: int,
                 n_mels: int,
                 n_fft: int,
                 img_size: int,
                 train: bool,
                 transform_train=None,
                 **_
                 ):
        super(MEAD, self).__init__()
        self.audio_dataroot = audio_dataroot
        self.visual_dataroot = visual_dataroot
        self.lm_dataroot = lm_dataroot
        self.num_frames = num_frames
        self.fps = fps
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.img_size = img_size
        self.train = train       
        
        filelists = []
        folders = os.listdir(audio_dataroot)
        for folder in tqdm(folders, total=len(folders)):
            files = sorted(os.listdir(os.path.join(audio_dataroot,folder)))
            for file in files:
                filelists.append(f'{folder}/{file[:-4]}')
        
        random.seed(0)
        random.shuffle(filelists)

        self.all_datas = []
        if self.train:
            self.all_datas = filelists[:int(len(filelists) * 0.8)]
        else:
            self.all_datas = filelists[int(len(filelists) * 0.8):]
            
        emo_list = ["angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
        
        self.class_idx_to_label = dict()
        for idx, emo in enumerate(emo_list):
            self.class_idx_to_label[idx] = emo
        self.label_to_class_idx = {lb: idx for idx, lb in self.class_idx_to_label.items()}

    def get_frame_id(self, frame):
        return int(os.path.basename(frame).split('.')[0])
    
    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        extension = os.path.basename(start_frame).split('.')[1]
        folder_name = os.path.dirname(start_frame)
        
        window_fnames = []
        for frame_id in range(start_id - self.num_frames//2, start_id + self.num_frames//2+1):
            frame = os.path.join(folder_name, f'{frame_id:05d}.{extension}')
            if not os.path.isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.img_size, self.img_size))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                return None
            window.append(img)
        return window

    def read_landmark(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            landmarks = []
            with open(fname, 'r') as f:
                data = json.load(f)
                for i in FACEMESH_LIPS_IDX:
                    landmarks.append(data[i])
            window.append(landmarks)
        return window

    def prepare_window(self, window):
        # T x 3 x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (0, 3, 1, 2))

        return x

    def __len__(self):
        return len(self.all_datas)
        # return 2

    def __getitem__(self, idx):
        while 1:
            folder_name = self.all_datas[idx]

            if self.visual_dataroot is not None:
                img_names = sorted(glob(os.path.join(self.visual_dataroot, f'{folder_name}/**/*.jpg'), recursive=True))
                if len(img_names) <= 3 * self.num_frames:
                    continue
                file_name = random.choice(img_names[self.num_frames//2:-self.num_frames//2-1])
            elif self.lm_dataroot is not None:
                lm_names = sorted(glob(os.path.join(self.lm_dataroot, f'{folder_name}/**/*.json'), recursive=True))
                if len(lm_names) <= 3 * self.num_frames:
                    continue
                file_name = random.choice(lm_names[self.num_frames//2:-self.num_frames//2-1])
            
            frame_id = self.get_frame_id(file_name)
                
            #Image
            img_window = None
            if self.visual_dataroot is not None:
                img_name = os.path.join(self.visual_dataroot, f'{folder_name}/{frame_id:05d}.jpg')
                window_fnames = self.get_window(img_name)      
                img_window = self.read_window(window_fnames)  
                
                if img_window is None or len(img_window) != self.num_frames:
                    continue
                img_window = self.prepare_window(img_window)        #[5,128,128]
                img_window = torch.FloatTensor(img_window)
            
            #Landmark
            lm_window = None
            if self.lm_dataroot is not None:
                lm_name = os.path.join(self.lm_dataroot, f'{folder_name}/{frame_id:05d}.json')
                window_fnames = self.get_window(lm_name)    
                lm_window = self.read_landmark(window_fnames)
                if lm_window is None or len(lm_window) != self.num_frames:
                    continue
                lm_window = torch.FloatTensor(lm_window) # [5, 40, 2]

            #Audio            
            wavpath = os.path.join(self.audio_dataroot, f'{folder_name}.wav')
            audio, _ = librosa.load(wavpath, sr=self.sample_rate)
            frame_duration = 1/self.fps
            hop_length = int(frame_duration * self.sample_rate)
            # mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=hop_length, center=False)
            # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            # mel = mel_spectrogram_db[:,frame_id - self.num_frames//2:frame_id + self.num_frames//2+1]
            # mel = torch.FloatTensor(mel.T) # [5, 80]
            # if mel.shape[0] != self.num_frames:
            #     continue
            
            audio_seg = [audio[i*hop_length:(i+1)*hop_length] for i in range(frame_id - self.num_frames//2,frame_id + self.num_frames//2+1)]
            # audio_seg = audio[(frame_id - self.num_frames//2) * hop_length:(frame_id + self.num_frames//2+1)*hop_length]
            audio_seg = np.asarray(audio_seg)
            audio_seg = torch.FloatTensor(audio_seg) # [5, 882] = [4410]
            if audio_seg.shape[0]  != self.num_frames:
                continue
            # if audio_seg.shape[0] // hop_length  != self.num_frames:
            #     continue

            
            #Emotion
            emo_text = folder_name.split('/')[0].split('_')[1]

            return audio_seg, img_window, None, lm_window

    def collate_fn(self, batch):
        batch_audio, batch_image, batch_text, batch_lm = zip(*batch)
        keep_ids = [idx for idx, (_, _) in enumerate(zip(batch_audio, batch_text))]

        if not all(audio is None for audio in batch_audio):
            batch_audio = [batch_audio[idx] for idx in keep_ids]
            batch_audio = torch.stack(batch_audio)
            batch_audio = batch_audio[0]
        else:
            batch_audio = None

        if not all(image is None for image in batch_image):
            batch_image = [batch_image[idx] for idx in keep_ids]
            batch_image = torch.stack(batch_image)
            batch_image = batch_image[0]
        else:
            batch_image = None

        if not all(text is None for text in batch_text):
            batch_text = [batch_text[idx] for idx in keep_ids]
        else:
            batch_text = None
            
        if not all(lm is None for lm in batch_lm):
            batch_lm = [batch_lm[idx] for idx in keep_ids]
            batch_lm = torch.stack(batch_lm)
        else:
            batch_lm = None

        return batch_audio, batch_image, batch_text, batch_lm
