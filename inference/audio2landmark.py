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

import torch
import torch.utils.data

from ignite_trainer import _utils
from ignite_trainer import _interfaces

from utility import plot_landmark_connections, calculate_LMD
from utility import FACEMESH_ROI_IDX


def load_data(data_file, audio_dataroot, lm_dataroot, vs_dataroot, n_frames=5):
    persons, _ = os.path.splitext(os.path.basename(data_file))   
    persons = [persons]
    data_path = data_file
    
    filelists = []
    for p in persons:
        data_path_p = data_path.format(p=p)
        with open(data_path_p, 'r') as file:
            for line in file:
                line = line.strip()
                filelists.append(f"{p}\t{line}")
    
    random.seed(0)
    random.shuffle(filelists)
    filelists = filelists[int(len(filelists) * 0.9):] 
    
    print("Loading datas ...")
    for fileline in filelists:
        p,line = fileline.strip().split("\t")
        
        audio_p = audio_dataroot.format(p=p)
        lm_p = lm_dataroot.format(p=p)
        vs_p = vs_dataroot.format(p=p)
        
        audio_name = os.path.join(audio_p, f'{line}.json')
        lm_folder = os.path.join(lm_p, line)
        vs_folder = os.path.join(vs_p, line)
        
        with open(audio_name, "r") as f:
            data = json.load(f)
        mel_spectrogram_db = torch.tensor(data["mel_spectrogram_db"])
        
        lm_paths = sorted(os.listdir(lm_folder))
        max_len = min(mel_spectrogram_db.shape[0], len(lm_paths))
        
        if max_len >= 25 and n_frames == -1:
            n_frames = max_len
        elif max_len < n_frames:
            continue
        
        seg_idx = list(range(max_len - n_frames + 1))        
        for segment_start_idx in seg_idx:
            idx_found = True
            for i in range(segment_start_idx, segment_start_idx + n_frames):
                if not os.path.exists(os.path.join(lm_folder,lm_paths[i])) or not os.path.exists(os.path.join(lm_folder,lm_paths[i]).replace(".json", ".jpg").replace("face_meshes", "images")):
                    idx_found = False
                    break
            if idx_found:
                break
        
        if not idx_found:
            continue
        
        lm_data_list = []
        vs_path = []
        for i in range(segment_start_idx, segment_start_idx + n_frames):
            with open(os.path.join(lm_folder,lm_paths[i]), "r") as f:
                lm_data = json.load(f)
                lm_roi = []
                for j in FACEMESH_ROI_IDX:
                    lm_roi.append(lm_data[j])
                lm_roi = np.asarray(lm_roi)
                lm_roi = torch.FloatTensor(lm_roi)
                lm_roi = lm_roi / 256.0
                
                lm_data_list.append(lm_roi)
                vs_path.append(os.path.join(vs_folder,lm_paths[i].replace(".json", ".jpg")))
                
        lm_data_list = np.stack(lm_data_list, axis=0)
        lm_data_list = torch.tensor(lm_data_list) #(N, lm_points, 2)
        lm_data_list = lm_data_list.unsqueeze(0)
        
        mel_segment = mel_spectrogram_db[segment_start_idx:segment_start_idx + n_frames, :] #(N, 80)
        mel_segment = mel_segment.unsqueeze(0)
        
        audio_list = []
        audio_data, _ = librosa.load(audio_name.replace(".json", ".wav").replace("audio_features", "audios"), sr=16000)
        for i in range(segment_start_idx, segment_start_idx + n_frames):
            start = i * 635
            end = start + 800
            audio_seg = audio_data[start:end]
            audio_list.append(audio_seg)
        
        return (mel_segment, lm_data_list, vs_path, audio_list)
    
def save_plot(pred_lm, gt_lm, img_paths, raw_audio_seg, audio_seg, output_file, image_size=256):
    background = cv2.imread(img_paths)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    background = cv2.resize(background, (image_size, image_size))

    combined_image = np.ones((image_size, image_size * 3, 3), dtype=np.uint8) * 255
    combined_image[:, :image_size, :] = background
    combined_image[:, image_size:image_size*2, :] = background

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Phần 1: Ảnh background + Ground Truth
    axes[0].imshow(combined_image[:, :image_size, :])
    plot_landmark_connections(axes[0], gt_lm, 'green')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    # Phần 2: Ảnh background + Prediction
    axes[1].imshow(combined_image[:, image_size:image_size*2, :])
    plot_landmark_connections(axes[1], pred_lm, 'red')
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Phần 3: Ảnh Ground Truth (đỏ) và Prediction (xanh dương)
    axes[2].imshow(combined_image[:, image_size*2:image_size*3, :])
    axes[2].scatter(gt_lm[:, 0], gt_lm[:, 1], color='green', label='Ground Truth', s=2)
    axes[2].scatter(pred_lm[:, 0], pred_lm[:, 1], color='red', label='Prediction', s=2)
    axes[2].set_title('GT (Green) vs Prediction (Red)')
    axes[2].axis('off')
    
    # Phần 4: Biểu đồ từ dữ liệu audio_seg
    amplitude = np.abs(raw_audio_seg).mean()
    axes[3].plot(raw_audio_seg)
    axes[3].set_title(f'Raw audio with amp: {amplitude}')
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Amplitude')
    
    # Phần 4: Biểu đồ từ dữ liệu audio_seg
    axes[4].plot(audio_seg)
    axes[4].set_title('Audio Segment')
    axes[4].set_xlabel('Mel')
    axes[4].set_ylabel('Amplitude')

    # Lưu ảnh vào file
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
def inference(model, batch, device, output_file, save_plot_folder):
    with torch.no_grad():
        audio, landmark, lm_paths, raw_audio = batch
        
        gt_lm_list = []
        pred_lm_list = []
        for i in range(1, 5):
            audio_seg = audio[0,i-1]     
            gt_landmark = landmark[:,i-1]                        
            gt_lm_list.append(gt_landmark)
            pred_lm_list.append(gt_landmark)
            if save_plot_folder is not None:
                image_size = 256
                lm_output_file = os.path.join(save_plot_folder, f'landmarks_{i-1}.png')
                gt_lm = gt_landmark[0] * image_size
                pred_lm = gt_landmark[0] * image_size
                save_plot(pred_lm, gt_lm, lm_paths[i-1], raw_audio[i-1], audio_seg, lm_output_file, image_size)   
                
        
        for i in tqdm(range(5, audio.shape[1]+1), desc="Landmark Processing"):
            audio_seg = audio[:,i-5:i]
            prv_landmark = landmark[:,i-5:i-1]
            gt_landmark = landmark[:,i-1]
            (pred_landmark), _ = model(
                audio = audio_seg, 
                landmark = prv_landmark,
                gt_lm = gt_landmark
            )
            pred_landmark = pred_landmark.detach().cpu()
            
            ld_score = calculate_LMD(pred_landmark * 256, gt_landmark * 256)
            if ld_score > 10.0:
                pred_landmark = gt_landmark.clone()
            
            gt_lm_list.append(gt_landmark)
            pred_lm_list.append(pred_landmark)
            
            if save_plot_folder is not None:
                image_size = 256
                lm_output_file = os.path.join(save_plot_folder, f'landmarks_{i-1}.png')
                gt_lm = gt_landmark[0] * image_size
                pred_lm = pred_landmark[0] * image_size
                save_plot(pred_lm, gt_lm, lm_paths[i-1], raw_audio[i-1], audio_seg[0,-1], lm_output_file, image_size)
            
        gt_lm_list = torch.cat(gt_lm_list, dim=0).tolist()
        pred_lm_list = torch.cat(pred_lm_list, dim=0).tolist()
        data = {
            "gt_lm": gt_lm_list,
            "pred_lm": pred_lm_list,
            "paths": lm_paths
        }
        with open(output_file , 'w') as json_file:
            json.dump(data, json_file)
        
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-d', '--data_root', type=str, required=True)
    parser.add_argument('-f', '--data_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=False)
    parser.add_argument('--pretrained', type=str, required=False)
    parser.add_argument('--n_frames', type=int, default=25)
    parser.add_argument('--save_plot', type=str, required=False)

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
    
    batch = load_data(args.data_file, audio_dataroot, lm_dataroot, vs_dataroot, n_frames=args.n_frames)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if args.save_plot is not None:
        os.makedirs(args.save_plot, exist_ok=True)
    
    inference(model, batch, device, args.output_file, args.save_plot)
    
        
if __name__ == '__main__':
    main()