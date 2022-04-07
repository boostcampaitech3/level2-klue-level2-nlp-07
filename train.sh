# How to use: type 'sh train.sh' on your CLI
python train.py \
--load_data_filename load_data_junejae \
--load_data_func_load load_data \
--load_data_func_tokenized tokenized_dataset \
--load_data_func_tokenized_train tokenized_dataset \
--load_data_class RE_Dataset \
--metric_for_best_model 'eval_loss' \
--gradient_accumulation_steps 1 \
--use_augmentation True \
--aug_data ../dataset/train/augmented_phonologicalProcess.csv \
--seed 42 \
--model klue/roberta-large \
--train_data ../dataset/train/train_finalCorrection.csv \
--num_labels 30 \
--output_dir ./results \
--save_total_limit 10 \
--save_steps 500 \
--num_train_epochs 2 \
--learning_rate 1e-5 \
--per_device_train_batch_size 34 \
--per_device_eval_batch_size 64 \
--warmup_steps 500 \
--weight_decay 0.0 \
--logging_dir ./logs \
--logging_steps 500 \
--evaluation_strategy steps \
--eval_steps  500 \
--load_best_model_at_end True \
--save_pretrained ./best_model \
--tokenize punct \
--n_splits 1 \
--test_size 0.2 \
--report_to wandb \
--project_name [junejae]eval_aug_test \
--entity_name growing_sesame \
--run_name robertaL_batch34_spellNoise_qustionFront_1e-5_warmup0.0_epoch2_st_sentenceToken_withClue_dropout0.0_decay0.0_corrected_focal_minQuestion3

# --train_data ../dataset/train/train.csv \
# --train_data ../dataset/train/train_finalCorrection.csv \