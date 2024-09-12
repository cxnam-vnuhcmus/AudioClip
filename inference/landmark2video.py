import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import argparse
import moviepy.editor as mv
from moviepy.editor import *
from glob import glob
import json
import numpy as np
import torch
import cv2
import librosa
import soundfile as sf
from utility import FACEMESH_ROI_IDX, ALL_GROUPS

def draw_landmark(data_file, output_file, vs_dataroot, audio_dataroot, fps=25., sr=16000):
    with open(data_file, "r") as f:
        data = json.load(f)
    
    gt_lm = data["gt_lm"]
    pred_lm = data["pred_lm"]
    vs_paths = data["paths"]
    
    print("Loading datas...")
    images_list = []
    for i in range(len(pred_lm)):
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
        images_list.append(image_lm)
    
    audio_path = vs_paths[0].replace(vs_dataroot, audio_dataroot)
    audio_file_name = f'{os.path.dirname(audio_path)}.wav'
    start_seg_idx = int(os.path.basename(audio_path).replace(".jpg","")) - 1
    audio_data, _ = librosa.load(audio_file_name, sr=sr)
    audio_seg = audio_data[start_seg_idx * 635 : (start_seg_idx + len(pred_lm) - 1) * 635 + 800 + 1]
    audio_seg = np.asarray(audio_seg, dtype=np.float32)
    sf.write('./audio.wav', audio_seg, sr)
    audioclip = AudioFileClip("./audio.wav", fps=sr)
    
    video = ImageSequenceClip(images_list, fps=fps)
    video = video.set_audio(audioclip)
    video.write_videofile(output_file, audio=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--data_file', type=str, required=True)
    parser.add_argument('--log_samples', type=str, required=True)
    parser.add_argument('--vs_dataroot', type=str, required=False)
    parser.add_argument('--audio_dataroot', type=str, required=False)

    args, unknown_args = parser.parse_known_args()

    os.makedirs(os.path.dirname(args.log_samples), exist_ok=True)
    
    draw_landmark(args.data_file, args.log_samples, args.vs_dataroot, args.audio_dataroot)
    
        
if __name__ == '__main__':
    main()