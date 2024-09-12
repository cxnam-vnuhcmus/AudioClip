python main.py \
  --config config/model-base.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_base \
  --suffix CREMAD_base \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_base_checkpoint_20.pt \
  --evaluation


python main.py \
  --config config/model-apl.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_apl \
  --suffix CREMAD_apl \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_apl_checkpoint_20.pt \
  --evaluation
  

python main.py \
  --config config/model-speechsyncnet.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_ssn \
  --suffix CREMAD_ssn \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_ssn_checkpoint_20.pt \
  --evaluation
  

python main.py \
  --config config/model-apl-speechsyncnet.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_apl_ssn \
  --suffix CREMAD_apl_ssn \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_apl_ssn_checkpoint_20.pt \
  --evaluation


python main.py \
  --config ./config/model-apl-ssn-concate.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_apl_ssn_concate \
  --suffix CREMAD_apl_ssn_concate \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_apl_ssn_concate_checkpoint_20.pt \
  --evaluation


python main.py \
  --config ./config/model-apl-ssn-contrastive.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_apl_ssn_contrastive \
  --suffix CREMAD_apl_ssn_contrastive \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_apl_ssn_contrastive_checkpoint_20.pt \
  --evaluation


python main.py \
  --config ./config/model-apl-ssn-concate-contrastive.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_apl_ssn_concate_contrastive \
  --suffix CREMAD_apl_ssn_concate_contrastive \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_apl_ssn_concate_contrastive_checkpoint_20.pt \
  --evaluation


python main.py \
  --config ./config/model-concate.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_concate \
  --suffix CREMAD_concate \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_concate_checkpoint_20.pt \
  --evaluation


python main.py \
  --config ./config/model-contrastive.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_contrastive \
  --suffix CREMAD_contrastive \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_contrastive_checkpoint_20.pt \
  --evaluation


python main.py \
  --config ./config/model-contrastive-concate.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_concate_contrastive \
  --suffix CREMAD_concate_contrastive \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_concate_contrastive_checkpoint_20.pt \
  --evaluation

python main.py \
  --config ./config/model-llfs-apl-ssn-contrastive-new.json \
  --data_root /home/cxnam/Documents/CREMAD \
  --data_file ./assets/datas_CREMAD \
  --log_samples ./assets/samples/CREMAD/samples_llfs_apl_ssn_contrastive_new_3 \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_CREMAD_llfs_apl_ssn_contrastive_new_3_checkpoint_20.pt \
  --evaluation


