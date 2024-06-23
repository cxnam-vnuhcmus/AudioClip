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
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils

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
                 lm_scale: int,
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
        self.lm_scale = lm_scale
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

    def read_image(self, window_fnames):
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
        window = np.asarray(window) / 255.
        window = np.transpose(window, (0, 3, 1, 2))
        return window

    def read_landmark(self, window_fnames, return_raw_lm=False, return_lm_sketch=False, return_lip_lm=False, return_lip_mask=False):
        if window_fnames is None: return (None, None, None, None)
        window_raw_lm = []
        window_lm_sketch = []
        window_lip_lm = []
        window_lip_mask = []
        for fname in window_fnames:
            with open(fname, 'r') as f:
                face_landmarks = json.load(f)
                if return_raw_lm:
                    window_raw_lm.append(face_landmarks)
                    
                if return_lm_sketch:
                    landmark_list = []
                    for x,y in face_landmarks:
                        landmark = mp.framework.formats.landmark_pb2.NormalizedLandmark()
                        landmark.x = x / self.lm_scale
                        landmark.y = y / self.lm_scale
                        landmark.z = 0 
                        landmark_list.append(landmark)
                    face_landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList(landmark=landmark_list)
                    img_sketch = np.zeros((self.lm_scale,self.lm_scale,3), np.uint8)
                    mp_drawing.draw_landmarks(
                        image=img_sketch,   
                        landmark_list=face_landmarks_proto,
                        connections= ROI,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1,color=(255,255,255)))
                    img_sketch = cv2.resize(img_sketch, (self.img_size, self.img_size))
                    img_sketch = np.transpose(img_sketch, (2,0,1))
                    window_lm_sketch.append(img_sketch)
                    
                    # output_name = os.path.basename(fname).split('.')[0] + '.jpg'
                    # output_path = os.path.join('./assets', output_name)
                    # cv2.imwrite(output_path, img_sketch)
                    # print(output_path)
                    
                if return_lip_lm:
                    lip_landmarks = []
                    for i in FACEMESH_LIPS_IDX:
                        lip_landmarks.append(face_landmarks[i])
                    window_lip_lm.append(lip_landmarks)
                    
                if return_lip_mask:
                    lip_landmarks = []
                    for i in SEQUENCE_LIPS_SHAPE_IDX:
                        lip_landmarks.append(face_landmarks[i])
                    lip_landmarks = np.array(lip_landmarks)
                    points = lip_landmarks.reshape(-1,1,2).astype(np.int32)
                    img_mask =np.zeros((self.lm_scale,self.lm_scale),dtype=np.int32)
                    cv2.drawContours(img_mask,[points],-1,(1),thickness=-1)
                    list_of_points_indices=np.nonzero(img_mask)
                    mask = np.zeros((self.lm_scale,self.lm_scale), np.uint8)
                    mask[list_of_points_indices] = 255
                    mask = cv2.resize(mask, (self.img_size, self.img_size))
                    mask = np.expand_dims(mask, axis=0)
                    window_lip_mask.append(mask)
                    
        return window_raw_lm, window_lm_sketch, window_lip_lm, window_lip_mask

    def __len__(self):
        return len(self.all_datas)
        # return 2

    def __getitem__(self, idx):
        while 1:
            idx = random.randrange(len(self.all_datas))
            folder_name = self.all_datas[idx]

            if self.visual_dataroot is not None and self.visual_dataroot != "":
                img_names = sorted(glob(os.path.join(self.visual_dataroot, f'{folder_name}/**/*.jpg'), recursive=True))
                if len(img_names) <= 3 * self.num_frames:
                    continue
                file_name = random.choice(img_names[self.num_frames//2:-self.num_frames//2-1])
            elif self.lm_dataroot is not None and self.lm_dataroot != "":
                lm_names = sorted(glob(os.path.join(self.lm_dataroot, f'{folder_name}/**/*.json'), recursive=True))
                if len(lm_names) <= 3 * self.num_frames:
                    continue
                file_name = random.choice(lm_names[self.num_frames//2:-self.num_frames//2-1])
            
            frame_id = self.get_frame_id(file_name)
                
            #Image
            img_window = None
            if self.visual_dataroot is not None and self.visual_dataroot != "":
                img_name = os.path.join(self.visual_dataroot, f'{folder_name}/{frame_id:05d}.jpg')
                window_fnames = self.get_window(img_name)      
                img_window = self.read_image(window_fnames)                  
                if img_window is None or len(img_window) != self.num_frames:
                    continue
                img_window = torch.FloatTensor(img_window)
            
            #Landmark
            lm_window = None
            if self.lm_dataroot is not None and self.lm_dataroot != "":
                lm_name = os.path.join(self.lm_dataroot, f'{folder_name}/{frame_id:05d}.json')
                window_fnames = self.get_window(lm_name)    
                (_, _, _, lip_mask) = self.read_landmark(window_fnames, return_lip_mask=True)
                if lip_mask is None or len(lip_mask) != self.num_frames:
                    continue
                lm_window = torch.FloatTensor(lip_mask) # [15, 40, 2] / [15, 1, 128, 128]

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
            # emo_text = [folder_name.split('/')[0].split('_')[1]] * self.num_frames

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
            batch_lm = batch_lm[0]
        else:
            batch_lm = None

        return batch_audio, batch_lm, batch_text
