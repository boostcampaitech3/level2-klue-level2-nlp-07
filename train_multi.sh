# # How to use: type 'sh train.sh' on your CLI
# python train.py \
# --load_data_filename load_data \
# --load_data_func_load load_data \
# --load_data_func_tokenized tokenized_dataset \
# --load_data_class RE_Dataset \
# --metric_for_best_model 'eval_micro f1 score' \
# --gradient_accumulation_steps 1 \
# --seed 42 \
# --model klue/roberta-large \
# --train_data ../dataset/train/train.csv \
# --num_labels 30 \
# --output_dir ./results \
# --save_total_limit 5 \
# --save_steps 500 \
# --num_train_epochs 3 \
# --learning_rate 2e-5 \
# --per_device_train_batch_size 32 \
# --per_device_eval_batch_size 32 \
# --warmup_steps 0.5 \
# --weight_decay 0.0 \
# --logging_dir ./logs \
# --logging_steps 500 \
# --evaluation_strategy steps \
# --eval_steps  500 \
# --load_best_model_at_end True \
# --save_pretrained ./best_model \
# --tokenize temp \
# --n_splits 1 \
# --test_size 0.2 \
# --report_to wandb \
# --project_name FAST_TEST \
# --entity_name growing_sesame \
# --run_name "roberta-large focal 0.5"

python inference.py \
--test_dataset ../dataset/test/test_data.csv \
--model  klue/roberta-large \
--model_dir ./results/checkpoint-2000 \
--tokenize punct \
--load_data_filename load_data \
--load_data_func_load load_data \
--load_data_func_tokenized tokenized_dataset \
--file_name roberta_large_f05_2500 \
--load_data_class RE_Dataset

python inference.py \
--test_dataset ../dataset/test/test_data.csv \
--model  klue/roberta-large \
--model_dir ./results/checkpoint-1500 \
--tokenize punct \
--load_data_filename load_data \
--load_data_func_load load_data \
--load_data_func_tokenized tokenized_dataset \
--file_name roberta_large_f05_3000 \
--load_data_class RE_Dataset

python inference.py \
--test_dataset ../dataset/test/test_data.csv \
--model  klue/roberta-large \
--model_dir ./results/checkpoint-1000 \
--tokenize punct \
--load_data_filename load_data \
--load_data_func_load load_data \
--load_data_func_tokenized tokenized_dataset \
--file_name roberta_large_f05_3500 \
--load_data_class RE_Dataset
