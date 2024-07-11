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

emotion_labels = ["angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
emotion_to_index = {label: idx for idx, label in enumerate(emotion_labels)}

def emotion_to_one_hot(emotion_label):
    one_hot_vector = np.zeros(len(emotion_labels))
    index = emotion_to_index[emotion_label]
    one_hot_vector[index] = 1
    return one_hot_vector


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
        for fileline in filelists:
            p,line = fileline.strip().split("\t")
            audio_p = self.audio_dataroot.format(p=p)
            visual_p = self.visual_dataroot.format(p=p)
            lm_p = self.lm_dataroot.format(p=p)
            transcript_p = self.transcript_dataroot.format(p=p)
            
            #audio
            audio_path = os.path.join(audio_p, f'{line}.wav')
            if (os.path.getsize(audio_path) == 0): continue
            if (librosa.get_duration(filename=audio_path) < 1.0): continue
            
            #emotion
            emotion_label = line.split('_')[1]
            
            #phoneme
            txt_name = os.path.join(transcript_p, f'{line}.json')
            with open(txt_name, "r") as f:
                data = json.load(f)
                for word in data['words']:
                    for phoneme in word['phonemes']:
                        phoneme_text = phoneme["phoneme"]
                        phoneme_start = phoneme["start"]
                        phoneme_end = phoneme["end"]
                        
                        frame_start = int(phoneme_start * self.fps)
                        frame_end = int(phoneme_end * self.fps)
                        num_indices = frame_end - frame_start + 1
                        num_to_select = min(5, num_indices)
                        all_indices = list(range(frame_start, frame_end + 1))
                        if num_to_select >= num_indices:
                            selected_indices = all_indices
                        else:
                            selected_indices = random.sample(all_indices, num_to_select)
                        
                        for frame_id in selected_indices:
                            lm_path = os.path.join(lm_p, line, f'{frame_id:05d}.json')
                            img_path = os.path.join(visual_p, line, f'{frame_id:05d}.jpg')
                            if os.path.exists(lm_path) and os.path.exists(img_path) and (os.path.getsize(img_path) != 0):
                                all_datas.append((phoneme_text, phoneme_start, phoneme_end, audio_path, img_path, emotion_label))
        return all_datas
        
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def __len__(self):
        return len(self.all_datas)
        # return 8

    def __getitem__(self, idx):        
        phoneme, phoneme_start, phoneme_end, audio_path, img_path, emotion_label = self.all_datas[idx]
        
        #audio
        audio_data, _ = librosa.load(audio_path, sr=self.sr)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=self.sr, n_mels=self.n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        total_frame = mel_spectrogram_db.shape[1]
        second = len(audio_data) // self.sr
        start_sample = int(phoneme_start  * total_frame / second)
        end_sample = int(phoneme_end * total_frame / second)
        trimmed_mel = mel_spectrogram_db[:,start_sample-1:end_sample+1]

        mel_shape = trimmed_mel.shape
        max_dim = max(self.n_mels, mel_shape[1])
        padded_spectrogram = np.zeros((max_dim, max_dim))
        start_x_pos_overlay = max_dim//2 - mel_shape[0]//2
        start_y_pos_overlay = max_dim//2 - mel_shape[1]//2
        padded_spectrogram[start_x_pos_overlay:(start_x_pos_overlay + mel_shape[0]), start_y_pos_overlay : (start_y_pos_overlay + mel_shape[1])] = trimmed_mel
        padded_spectrogram = torch.tensor(padded_spectrogram)
        trimmed_mel = (trimmed_mel * 255).astype(np.uint8)
        image_mel = Image.fromarray(trimmed_mel).convert("RGB")
        image_mel = self.transform(image_mel)
        
        #emotion
        emotion_label_onehot = emotion_to_one_hot(emotion_label)
        emotion_label_onehot = torch.tensor(emotion_label_onehot, dtype=torch.float16)
        
        #visual
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return phoneme, image_mel, image, emotion_label_onehot

    def collate_fn(self, batch):
        batch_phoneme, batch_audio_embedding, batch_image, batch_emotion_label_onehot = zip(*batch)
        keep_ids = [idx for idx, (_, _) in enumerate(zip(batch_phoneme, batch_audio_embedding))]

        if not all(text is None for text in batch_phoneme):
            batch_phoneme = [batch_phoneme[idx] for idx in keep_ids]
        else:
            batch_phoneme = None
            
        if not all(au is None for au in batch_audio_embedding):
            batch_audio_embedding = [batch_audio_embedding[idx] for idx in keep_ids]
            batch_audio_embedding = torch.stack(batch_audio_embedding)
        else:
            batch_audio_embedding = None
            
        if not all(img is None for img in batch_image):
            batch_image = [batch_image[idx] for idx in keep_ids]
            batch_image = torch.stack(batch_image, dim=0)
        else:
            batch_image = None
        
        if not all(img is None for img in batch_emotion_label_onehot):
            batch_emotion_label_onehot = [batch_emotion_label_onehot[idx] for idx in keep_ids]
            batch_emotion_label_onehot = torch.stack(batch_emotion_label_onehot, dim=0)
        else:
            batch_emotion_label_onehot = None
            
        return batch_phoneme, batch_audio_embedding, batch_image, batch_emotion_label_onehot
