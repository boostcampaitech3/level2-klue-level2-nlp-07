# How to use: type 'sh inference.sh' on your CLI
python inference.py \
--test_dataset ../dataset/test/test_data.csv \
--model  klue/roberta-large \
--model_dir ./results/checkpoint-4000 \
--special_entity_type default \
--preprocess False \
--load_data_filename load_data \
--load_data_func_load load_data \
--load_data_func_tokenized tokenized_dataset \
--load_data_class RE_Dataset
