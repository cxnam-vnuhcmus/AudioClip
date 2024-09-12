python main.py \
  --config config/model-base.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_base \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_base.pt \
  --evaluation

python main.py \
  --config config/model-apl.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_apl \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_apl_v2.pt \
  --evaluation

python main.py \
  --config config/model-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_ssn \
  --skip-train-val \
  --pretrained ./assets/checkpoints/cp_20_speechsyncnet.pt \
  --evaluation

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
  --config ./config/model-llfs-apl-ssn-contrastive-new.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_llfs_apl_ssn_contrastive_new_2 \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_M003_llfs_apl_ssn_contrastive_new_2_checkpoint_20.pt \
  --evaluation
