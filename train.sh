# How to use: type 'sh train.sh' on your CLI
python train.py \
--load_data_filename load_data \
--load_data_func_load load_data \
--load_data_func_tokenized tokenized_dataset \
--load_data_class RE_Dataset \
--metric_for_best_model 'eval_micro f1 score' \
--gradient_accumulation_steps 1 \
--seed 42 \
--model klue/roberta-large \
--train_data ../dataset/train/train.csv \
--num_labels 30 \
--output_dir ./results \
<<<<<<< HEAD
--save_total_limit 10 \
--save_steps 500 \
--num_train_epochs 4 \
=======
--save_total_limit 5 \
--save_steps 500 \
--num_train_epochs 5 \
>>>>>>> 4f011b80b7ca29a070f0e86fab12c3b4d31b57d1
--learning_rate 5e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--warmup_steps 500 \
--weight_decay 0.01 \
--logging_dir ./logs \
--logging_steps 500 \
--evaluation_strategy steps \
--eval_steps  500 \
--load_best_model_at_end True \
--save_pretrained ./best_model \
--tokenize temp \
--n_splits 1 \
--test_size 0.2 \
--report_to wandb \
<<<<<<< HEAD
--project_name TAPT \
--entity_name growing_sesame \
--run_name robert_large_adaptive_5500
=======
--project_name baseline \
--entity_name growing_sesame \
--run_name testRun
>>>>>>> 4f011b80b7ca29a070f0e86fab12c3b4d31b57d1
