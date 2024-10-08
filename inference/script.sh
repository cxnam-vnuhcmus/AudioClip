python inference/audio2landmark.py \
  --config config/model-apl.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --output_file ./inference/samples/lm_data.json \
  --pretrained ./assets/checkpoints/checkpoint_M003_apl_mse_mfcc_checkpoint_19.pt \
  --n_frames -1 \
  --save_plot ./inference/samples/a2lm \

python inference/landmark2face.py \
  --config /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/config/model-lm-vae-gan.json \
  --data_root /home/cxnam/Documents/MEAD/M003 \
  --data_file ./inference/samples/lm_data.json \
  --log_samples ./inference/samples/lm2face \
  --pretrained /home/cxnam/Documents/MyWorkingSpace/LM2F_VAE/assets/checkpoints/best_M003_lm_vae_v3_checkpoint_1_MSE=-0.0040.pt \


python inference/face2video.py \
  --log_samples ./inference/samples/lm2face \
  --output_file ./inference/samples/video.mp4 \

python inference/landmark2video.py \
  --data_file ./inference/samples/lm_data.json \
  --log_samples ./inference/samples/lm_video.mp4 \
  --vs_dataroot /home/cxnam/Documents/MEAD/M003/images \
  --audio_dataroot /home/cxnam/Documents/MEAD/M003/audios \
