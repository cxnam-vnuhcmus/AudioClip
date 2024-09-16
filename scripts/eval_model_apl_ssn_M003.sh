python main.py \
  --config config/model-base.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M030.txt \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_M003_base_mse_checkpoint_20.pt \
  --evaluation \
  --n_folders 10 \
  --log_samples ./assets/samples/MEAD/samples_base_mse

python main.py \
  --config config/model-apl.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_MEAD_apl_mse_checkpoint_20.pt \
  --evaluation \
  --n_folders 10 \
  --log_samples ./assets/samples/MEAD/samples_apl_mse

python main.py \
  --config config/model-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_MEAD_ssn_mse_checkpoint_20.pt \
  --evaluation \
  --n_folders 10 \
  --log_samples ./assets/samples/M003/samples_ssn_mse


python main.py \
  --config config/model-apl-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_apl_ssn \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_apl_speechsyncnet.pt \
  --evaluation

python main.py \
  --config ./config/model-apl-ssn-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_apl_concate \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_apl_ssn_concate.pt \
  --evaluation

python main.py \
  --config ./config/model-apl-ssn-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_apl_ssn_contrastive \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_apl_ssn_contrastive.pt \
  --evaluation

python main.py \
  --config ./config/model-apl-ssn-concate-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_apl_ssn_concate_contrastive \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_apl_ssn_concate_contrastive.pt \
  --evaluation

python main.py \
  --config ./config/model-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_concate \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_concate.pt \
  --evaluation

python main.py \
  --config ./config/model-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_contrastive \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_contrastive.pt \
  --evaluation

python main.py \
  --config ./config/model-contrastive-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_concate_contrastive \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_contrastive_concate.pt \
  --evaluation

python main.py \
  --config ./config/model-mouth.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_M003_mouth_checkpoint_20.pt \
  --evaluation \
  --n_folders 10 \
  --log_samples ./assets/samples/MEAD/samples_multimodel_mse

python main.py \
  --config ./config/model-llfs-only.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_llfs_only.pt \
  --evaluation \
  --n_folders 10 \
  --log_samples ./assets/samples/M003/samples_llf_only

python main.py \
  --config ./config/model-base.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M030.txt \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_MEAD_base_mse_mfcc_checkpoint_20.pt \
  --evaluation \
  --n_folders 10 \
  --log_samples ./assets/samples/M003/samples_base_mse_mfcc