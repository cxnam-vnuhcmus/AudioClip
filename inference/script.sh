python inference/audio2landmark.py \
  --config config/model-base.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --output_file ./inference/samples/lm_data.json \
  --pretrained ./assets/checkpoints/checkpoint_MEAD_base_checkpoint_20.pt \
  --n_frames 25 \
  --save_plot ./inference/samples/a2lm \

python inference/landmark2face.py \
  --config config/model-lm-vae-2.json \
  --data_root /home/cxnam/Documents/MEAD/M003 \
  --data_file ./inference/samples/lm_data.json \
  --log_samples ./inference/samples/lm2face \
  --pretrained ./assets/checkpoints/checkpoint_M003_lm_vae_checkpoint_93.pt \

python inference/face2video.py \
  --log_samples ./inference/samples/lm2face \
  --output_file ./inference/samples/video.mp4 \

python inference/landmark2video.py \
  --data_file ./inference/samples/lm_data.json \
  --log_samples ./inference/samples/lm_video.mp4 \
  --vs_dataroot /home/cxnam/Documents/MEAD/M003/images \
  --audio_dataroot /home/cxnam/Documents/MEAD/M003/audios \
