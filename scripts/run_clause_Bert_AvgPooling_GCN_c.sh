# CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 --wait-for-client run_glue_no_trainer_Bert_AvgPooling_GCN_c.py \
CUDA_VISIBLE_DEVICES=1 python run_glue_no_trainer_Bert_AvgPooling_GCN_c.py \
  --model_name_or_path /data10T/yangcaihua/download/bert-base-chinese  \
  --per_device_train_batch_size 128 \
  --learning_rate 1e-5 \
  --num_train_epochs 50 \
  --output_dir output/Bert_AvgPooling_GCN_c/ \
  --train_file ./data/sentc_train.csv \
  --validation_file ./data/sentc_dev.csv \
  --test_file ./data/sentc_test.csv \
  --graph_type dep_sentic
