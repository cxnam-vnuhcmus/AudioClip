python main.py \
  --config config/model-fullmesh.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --suffix M003_fullmesh_mfcc \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_M003_fullmesh_mfcc_checkpoint_5.pt

python main.py \
  --config config/model-apl-ssn.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas_norm/M003.txt \
  --suffix M003_ssn \
  --model model_apl_ssn.model_ssn.Model \
  --skip-train-val \
  --n_folders 10




python main.py \
  --config config/model-apl.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --suffix M003_apl_mse_mfcc \
  --skip-train-val \
  --n_folders 10 \
  --log_samples ./assets/samples/M003/samples_apl \
  

python main.py \
  --config config/model-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --suffix M003_ssn_mse_mfcc_av1_lmv31 \
  --skip-train-val \
  --log_samples ./assets/samples/M003/samples_ssn_v2 \
  

python main.py \
  --config config/model-apl-ssn-add.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --suffix M003_apl_ssn_mfcc \
  --skip-train-val \
  --log_samples ./assets/samples/M003/samples_apl_ssn \
  

python main.py \
  --config ./config/model-apl-ssn-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --suffix M003_apl_ssn_mfcc_concate \
  --skip-train-val \
  --log_samples ./assets/samples/M003/samples_apl_ssn_concate \


python main.py \
  --config ./config/model-apl-ssn-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --suffix M003_apl_ssn_add_kdloss \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_M003_apl_ssn_add_kdloss_checkpoint_4.pt \


python main.py \
  --config ./config/model-apl-ssn-concate-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --suffix M003_apl_ssn_concate_kdloss \
  --skip-train-val \
  --log_samples ./assets/samples/M003/samples_apl_ssn_concate_contrastive \


python main.py \
  --config ./config/model-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_concate \
  --suffix M003_concate \
  --skip-train-val \


python main.py \
  --config ./config/model-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_contrastive \
  --suffix M003_contrastive \
  --skip-train-val \


python main.py \
  --config ./config/model-contrastive-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_concate_contrastive \
  --suffix M003_concate_contrastive \
  --skip-train-val \



python main.py \
  --config ./config/model-multimodel.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --suffix MEAD_multimodel_v1 \
  --skip-train-val \
  --n_folders 10

python main.py \
  --config config/model-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --suffix MEAD_ssn_v2 \
  --skip-train-val \
  --n_folders 10

python main.py \
  --config config/model-apl-ssn-kdloss.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --suffix M003_apl_ssn_kdloss_celoss_add \
  --skip-train-val \
  --pretrained ./assets/checkpoints/checkpoint_M003_apl_ssn_kdloss_celoss_add_checkpoint_9.pt \
  --n_folders 10

