python main.py \
  --config config/model-base.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --suffix MEAD_base_mse_mfcc \
  --skip-train-val \
  --n_folders 10 




python main.py \
  --config config/model-apl.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas \
  --suffix MEAD_apl_mfcc \
  --skip-train-val \
  --n_folders 10 \
  --log_samples ./assets/samples/M003/samples_apl \
  

python main.py \
  --config config/model-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --suffix M003_ssn_v2 \
  --skip-train-val \
  --log_samples ./assets/samples/M003/samples_ssn_v2 \
  

python main.py \
  --config config/model-apl-speechsyncnet.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_apl_ssn \
  --suffix M003_apl_ssn \
  --skip-train-val \
  

python main.py \
  --config ./config/model-apl-ssn-concate.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_apl_ssn_concate \
  --suffix M003_apl_ssn_concate \
  --skip-train-val \


python main.py \
  --config ./config/model-apl-ssn-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_apl_ssn_contrastive \
  --suffix M003_apl_ssn_contrastive \
  --skip-train-val \


python main.py \
  --config ./config/model-apl-ssn-concate-contrastive.json \
  --data_root /home/cxnam/Documents/MEAD \
  --data_file ./assets/datas/M003.txt \
  --log_samples ./assets/samples/M003/samples_apl_ssn_concate_contrastive \
  --suffix M003_apl_ssn_concate_contrastive \
  --skip-train-val \


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

