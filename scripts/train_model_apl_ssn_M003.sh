python main.py \
  --config config/model-base.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_base \
  --suffix base \
  --skip-train-val \



python main.py \
  --config config/model-apl.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_apl \
  --suffix apl \
  --skip-train-val \
  

python main.py \
  --config config/model-speechsyncnet.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_ssn \
  --suffix ssn \
  --skip-train-val \
  

python main.py \
  --config config/model-apl-speechsyncnet.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_apl_ssn \
  --suffix apl_ssn \
  --skip-train-val \
  

python main.py \
  --config Trainer/config/model-apl-ssn-concate.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_apl_ssn_concate \
  --suffix apl_ssn_concate \
  --skip-train-val \


python main.py \
  --config Trainer/config/model-apl-ssn-contrastive.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_apl_ssn_contrastive \
  --suffix apl_ssn_contrastive \
  --skip-train-val \


python main.py \
  --config Trainer/config/model-apl-ssn-concate-contrastive.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_apl_ssn_concate_contrastive \
  --suffix apl_ssn_concate_contrastive \
  --skip-train-val \


python main.py \
  --config Trainer/config/model-concate.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_concate \
  --suffix concate \
  --skip-train-val \


python main.py \
  --config Trainer/config/model-contrastive.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_contrastive \
  --suffix contrastive \
  --skip-train-val \


python main.py \
  --config Trainer/config/model-contrastive-concate.json \
  --data_root /home/cxnam/Documents/M003 \
  --data_file ./assets/datas_M003 \
  --log_samples ./assets/samples/M003/samples_concate_contrastive \
  --suffix concate_contrastive \
  --skip-train-val \

