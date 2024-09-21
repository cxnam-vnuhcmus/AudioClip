python main.py \
  --config config/model-base.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_base \
  --suffix MEAD_base \
  --skip-train-val \
  --n_folders 10



python main.py \
  --config config/model-apl.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_apl \
  --suffix MEAD_apl \
  --skip-train-val \
  --n_folders 10
  

python main.py \
  --config config/model-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_ssn \
  --suffix MEAD_ssn \
  --skip-train-val \
  --n_folders 10
  

python main.py \
  --config config/model-apl-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_apl_ssn \
  --suffix MEAD_apl_ssn \
  --skip-train-val \
  --n_folders 10
  

python main.py \
  --config ./config/model-apl-ssn-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_apl_ssn_concate \
  --suffix MEAD_apl_ssn_concate \
  --skip-train-val \
  --n_folders 10


python main.py \
  --config ./config/model-apl-ssn-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_apl_ssn_contrastive \
  --suffix MEAD_apl_ssn_contrastive \
  --skip-train-val \
  --n_folders 10


python main.py \
  --config ./config/model-apl-ssn-concate-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_apl_ssn_concate_contrastive \
  --suffix MEAD_apl_ssn_concate_contrastive \
  --skip-train-val \
  --n_folders 10


python main.py \
  --config ./config/model-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_concate \
  --suffix MEAD_concate \
  --skip-train-val \
  --n_folders 10


python main.py \
  --config ./config/model-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_contrastive \
  --suffix MEAD_contrastive \
  --skip-train-val \
  --n_folders 10


python main.py \
  --config ./config/model-contrastive-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --log_samples ./assets/samples/MEAD/samples_MEAD_concate_contrastive \
  --suffix MEAD_concate_contrastive \
  --skip-train-val \
  --n_folders 10


#Train

python main.py \
  --config ./config/model-llfs-apl-ssn-contrastive-new.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --suffix MEAD_multimodel_mse \
  --skip-train-val \
  --n_folders 10 \
  --batch-train 32 \
  --batch-test 32 \
  --pretrained ./assets/checkpoints/checkpoint_MEAD_multimodel_mse_checkpoint_1.pt \

python main.py \
  --config config/model-base.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --suffix MEAD_base_mse \
  --skip-train-val \
  --n_folders 10 \
  --pretrained ./assets/checkpoints/checkpoint_MEAD_base_mse_checkpoint_5.pt \
  --batch-train 32 \
  --batch-test 32

python main.py \
  --config config/model-apl.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --suffix MEAD_apl_mse \
  --skip-train-val \
  --n_folders 10 \
  --batch-train 32 \
  --batch-test 32

python main.py \
  --config config/model-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --suffix MEAD_ssn_mse \
  --skip-train-val \
  --n_folders 10 \
  --batch-train 32  \
  --batch-test 32

python main.py \
  --config config/model-apl-ssn-kdloss.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --suffix MEAD_apl_ssn_kdloss \
  --skip-train-val \
  --n_folders 10 \
  --pretrained ./assets/checkpoints/checkpoint_MEAD_apl_ssn_kdloss_checkpoint_18.pt
  
  