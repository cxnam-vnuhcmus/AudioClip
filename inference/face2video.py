import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# import cv2
# import subprocess
import argparse
import moviepy.editor as mv
from moviepy.editor import *
from glob import glob

def merge_audio_image_to_video(folder_input, output_file='video.mp4', fps=25., sr=16000):
    audioclip = AudioFileClip(os.path.join(folder_input, "audio.wav"), fps=sr)
    
    images_list = []
    for frame in sorted(glob(f'{folder_input}/pred_*.*')):
        if frame.endswith(".jpg") or frame.endswith(".png"):
            images_list.append(frame)
    video = ImageSequenceClip(images_list, fps=fps)
    video = video.set_audio(audioclip)
    video.write_videofile(output_file, audio=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_samples', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=False)

    args, unknown_args = parser.parse_known_args()

    os.makedirs(args.log_samples, exist_ok=True)
    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    merge_audio_image_to_video(folder_input = args.log_samples, output_file = args.output_file)
    
        
if __name__ == '__main__':
    main()