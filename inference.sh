# How to use: type 'sh inference.sh' on your CLI
python inference.py \
--test_dataset ../dataset/test/test_data.csv \
--model  xlm-roberta-large \
--model_dir ./results/checkpoint-2500 \
--tokenize punct \
--load_data_filename load_data \
--load_data_func_load load_data \
--load_data_func_tokenized tokenized_dataset \
--load_data_class RE_Dataset


# --model_dir ./hp_search/checkpoint-4500 \
# --model_dir ./results/checkpoint-5000 \
# --model_dir ./best_model \
# --model klue/roberta-large \
# --model klue/bert-base \
#  xlm-roberta-large