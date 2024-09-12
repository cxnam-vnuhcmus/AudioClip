import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import glob
import json
import argparse
import numpy as np
import random
import librosa
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
import soundfile as sf

import torch
import torch.utils.data
import torchvision.transforms as T

from ignite_trainer import _utils
from ignite_trainer import _interfaces

from utility import plot_landmark_connections
from utility import FACEMESH_ROI_IDX, ALL_GROUPS

from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")

inv_normalize = T.Compose([
    T.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0/0.5, 1.0/0.5, 1.0/0.5]),
    T.ToPILImage()
]) 

def load_data(data_file, lm_dataroot, vs_dataroot, vs_ft_dataroot, audio_dataroot):
    with open(data_file, "r") as f:
        data = json.load(f)
    
    gt_lm = data["gt_lm"]
    pred_lm = data["pred_lm"]
    vs_paths = data["paths"]
    
    image_lm_list = []
    gt_img_feature_list = []
    
    print("Loading datas...")
    for i in range(len(pred_lm)):
        vs_ft_path = vs_paths[i].replace(vs_dataroot, vs_ft_dataroot).replace(".jpg", ".json")
        
        #landmark
        image_lm = np.zeros((256, 256, 3), dtype=np.uint8)
        face_landmarks = torch.tensor(pred_lm[i]) * 256
        for group in ALL_GROUPS:
            for (start_idx, end_idx) in group:
                start_point = face_landmarks[FACEMESH_ROI_IDX.index(start_idx)]
                end_point = face_landmarks[FACEMESH_ROI_IDX.index(end_idx)]
                start_point = tuple(map(int, start_point))
                end_point = tuple(map(int, end_point))
                cv2.line(image_lm, start_point, end_point, (255,255,255), 1)
        
        image_lm = torch.from_numpy(image_lm).float() / 255.0
        image_lm = image_lm.permute(2,0,1).unsqueeze(0) #(1,3,256,256)
        
        with open(vs_ft_path, "r") as f:
            img_feature = json.load(f)
        gt_img_feature = torch.FloatTensor(img_feature) #(1,4,32,32)
            
        image_lm_list.append(image_lm)
        gt_img_feature_list.append(gt_img_feature)
    
    image_lm_list = torch.cat(image_lm_list, dim=0)
    gt_img_feature_list = torch.cat(gt_img_feature_list, dim=0)
    
    audio_path = vs_paths[0].replace(vs_dataroot, audio_dataroot)
    audio_file_name = f'{os.path.dirname(audio_path)}.wav'
    start_seg_idx = int(os.path.basename(audio_path).replace(".jpg","")) - 1
    audio_data, _ = librosa.load(audio_file_name, sr=16000)
    audio_seg = audio_data[start_seg_idx * 635 : (start_seg_idx + len(pred_lm) - 1) * 635 + 800 + 1]
    audio_seg = np.asarray(audio_seg, dtype=np.float32)
    
    return (image_lm_list, gt_img_feature_list, audio_seg)
    

def inference(model, batch, device, save_folder):
    image_lm_list, gt_img_feature_list, audio_seg = batch

    gt_img_feature_list = gt_img_feature_list.to(device)
        
    (pred_img_feature), _ = model(
        landmark = image_lm_list,
        gt_img_feature = gt_img_feature_list
    )
        
    pred_img_feature = pred_img_feature.detach().cpu()
    
    with torch.no_grad():
        for i in tqdm(range(pred_img_feature.shape[0]), desc="Face Generating"):
            pf = pred_img_feature[i].unsqueeze(0)
            samples = vae.decode(pf)
            output = samples.sample[0]
            inv_image = inv_normalize(output)
            inv_image.save(os.path.join(save_folder,f'pred_image_{i:05d}.jpg'))
    
    sf.write(os.path.join(save_folder,'audio.wav'), audio_seg, 16000)


        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-d', '--data_root', type=str, required=True)
    parser.add_argument('-f', '--data_file', type=str, required=True)
    parser.add_argument('--log_samples', type=str, required=False)
    parser.add_argument('--pretrained', type=str, required=False)

    args, unknown_args = parser.parse_known_args()

    config = json.load(open(args.config))
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_class = config['Model']['class']
    model_args = config['Model']['args']
    
    Network: Type = _utils.load_class(model_class)        
    model: _interfaces.AbstractNet = Network(**model_args)
    model = model.to(device)
    
    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint['model'])
    
    audio_dataroot = os.path.join(args.data_root, config["Dataset"]["args"]["audio_dataroot"])
    lm_dataroot = os.path.join(args.data_root, config["Dataset"]["args"]["lm_dataroot"])
    vs_dataroot = os.path.join(args.data_root, config["Dataset"]["args"]["visual_dataroot"])
    vs_ft_dataroot = os.path.join(args.data_root, config["Dataset"]["args"]["visual_feature_dataroot"])
    
    batch = load_data(args.data_file, lm_dataroot, vs_dataroot, vs_ft_dataroot, audio_dataroot)
    
    os.makedirs(args.log_samples, exist_ok=True)
    
    inference(model, batch, device, args.log_samples)
    
        
if __name__ == '__main__':
    main()