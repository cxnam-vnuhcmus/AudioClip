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



class Dataset(td.Dataset):

    def __init__(self, 
                 datalist_root: str,
                 visual_dataroot: str,
                 transcript_dataroot: str, 
                 lm_dataroot: str,
                 fps: int,
                 img_size: int,
                 train: bool,
                 **_
                 ):
        super(Dataset, self).__init__()
        self.datalist_root = datalist_root
        self.visual_dataroot = visual_dataroot
        self.transcript_dataroot = transcript_dataroot
        self.lm_dataroot = lm_dataroot
        self.fps = fps
        self.img_size = img_size
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
      
        filelists = []
        with open(self.datalist_root, 'r') as file:
            for line in file:
                line = line.strip()
                txt_name = os.path.join(self.transcript_dataroot, f'{line}.json')
                with open(txt_name, "r") as f:
                    data = json.load(f)
                    for word in data['words']:
                        for phoneme in word['phonemes']:
                            frame_start = phoneme["start"] * self.fps
                            frame_end = phoneme["end"] * self.fps
                            frame_mid = int((frame_start + frame_end)/2)
                            frame_range = [frame_mid, frame_mid + 1, frame_mid - 1, frame_mid + 2, frame_mid - 2]
                            for frame_id in frame_range:
                                lm_path = os.path.join(self.lm_dataroot, line, f'{frame_id:05d}.json')
                                img_path = os.path.join(self.visual_dataroot, line, f'{frame_id:05d}.jpg')
                                if os.path.exists(lm_path) and os.path.exists(img_path) and (os.path.getsize(img_path) != 0):
                                    filelists.append((phoneme['phoneme'],lm_path, img_path))
                                    break
        
        random.seed(0)
        random.shuffle(filelists)

        self.all_datas = []
        if self.train:
            self.all_datas = filelists[:int(len(filelists) * 0.8)]
        else:
            self.all_datas = filelists[int(len(filelists) * 0.8):]
    
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
        
    def __len__(self):
        # return len(self.all_datas)
        return 2

    def __getitem__(self, idx):
        (phoneme, lm_path, img_path) = self.all_datas[idx]
        
        while 1:
            ref_idx = random.randrange(len(self.all_datas))
            _, _, ref_img_path = self.all_datas[ref_idx]
            ref_folder_name = os.path.dirname(ref_img_path).replace(self.visual_dataroot,'').strip('/')
            real_folder_name = os.path.dirname(img_path).replace(self.visual_dataroot,'').strip('/')
            if ref_folder_name != real_folder_name:
                break

        with open(lm_path, "r") as f:
            lm_data = json.load(f)
        lm_lip = []
        for i in FACEMESH_LIPS_IDX:
            lm_lip.append(lm_data[i])
        lm_lip = np.asarray(lm_lip)
        lm_lip = torch.FloatTensor(lm_lip)
        lm_lip = lm_lip / 256.0
        mean = torch.tensor([0.5, 0.5])
        std = torch.tensor([0.5, 0.5])
        lm_lip = (lm_lip - mean) / std
        lm_lip = lm_lip.to(self.device)
                
        gt_image = self.preprocess_image(img_path)                #[3,256,256]
        gt_image = torch.tensor(gt_image).unsqueeze(0)            #[1,3,256,256]

        ref_image = self.preprocess_image(ref_img_path)             #[3,256,256]
        ref_image = torch.tensor(ref_image).unsqueeze(0)            #[1,3,256,256]

        mask = torch.ones_like(gt_image)  # (3,256,256)
        mask[:, gt_image.shape[1]//2: , :] = 0        
        mask_image = gt_image * mask
        
        return phoneme, lm_lip, mask_image, ref_image, gt_image

    def collate_fn(self, batch):
        batch_phoneme, batch_lm, batch_mask_image, batch_ref_image, batch_gt_imgage = zip(*batch)
        keep_ids = [idx for idx, (_, _) in enumerate(zip(batch_phoneme, batch_lm))]

        if not all(text is None for text in batch_phoneme):
            batch_phoneme = [batch_phoneme[idx] for idx in keep_ids]
        else:
            batch_phoneme = None
            
        if not all(lm is None for lm in batch_lm):
            batch_lm = [batch_lm[idx] for idx in keep_ids]
            batch_lm = torch.stack(batch_lm)
        else:
            batch_lm = None
            
        if not all(img is None for img in batch_mask_image):
            batch_mask_image = [batch_mask_image[idx] for idx in keep_ids]
            batch_mask_image = torch.cat(batch_mask_image, dim=0)
        else:
            batch_mask_image = None
            
        if not all(img is None for img in batch_ref_image):
            batch_ref_image = [batch_ref_image[idx] for idx in keep_ids]
            batch_ref_image = torch.cat(batch_ref_image, dim=0)
        else:
            batch_ref_image = None

        if not all(img is None for img in batch_gt_imgage):
            batch_gt_imgage = [batch_gt_imgage[idx] for idx in keep_ids]
            batch_gt_imgage = torch.cat(batch_gt_imgage, dim=0)
        else:
            batch_gt_imgage = None
                        
        return batch_phoneme, batch_lm, batch_mask_image, batch_ref_image, batch_gt_imgage

    def preprocess_image(self, image_path, target_size=(256, 256)):
        # Read image using OpenCV
        image = cv2.imread(image_path)
        
        # Resize image
        image_resized = cv2.resize(image, target_size)
        
        # Convert from BGR to RGB (if needed)
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize image (assuming you want to normalize to [0, 1])
        image_normalized = image_resized.astype(np.float32) / 255.0

        image_normalized = image_normalized.transpose((2, 0, 1))
        
        return image_normalized
