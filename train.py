import pickle as pickle
import os
import pandas as pd
import torch
import random
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
import wandb
import argparse
from importlib import import_module
from loss import FocalLoss

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")  
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = FocalLoss(gamma=0.5)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train(args):
  seed_everything(args.seed)
  # load model and tokenizer
  MODEL_NAME = args.model
  # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, additional_special_tokens=["#", "@","^","*", "<S:PER>", "</S:PER>", "<S:ORG>", "</S:ORG>", "<O:DAT>", "</O:DAT>", "<O:LOC>", "</O:LOC>", "<O:NOH>", "</O:NOH>", "<O:ORG>", "</O:ORG>", "<O:PER>", "</O:PER>", "<O:POH>", "</O:POH>"])
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  load = getattr(import_module(args.load_data_filename), args.load_data_func_load)
  dataset = load(args.train_data)

  split = StratifiedShuffleSplit(n_splits=args.n_splits, test_size=args.test_size, random_state=args.seed)

  for train_idx, test_idx in split.split(dataset, dataset["label"]):
      train_dataset = dataset.loc[train_idx]
      dev_dataset = dataset.loc[test_idx]
  
  dev_index = dev_dataset['id'].tolist() # added for augmentation
  if args.use_augmentation: # added for augmentation
    aug_dataset1 = load('../dataset/train/augmented_phonologicalProcess.csv')
    aug_dataset2 = load('../dataset/train/augmented_vowelNoise.csv')
    temp = pd.concat([train_dataset, aug_dataset1, aug_dataset2]).drop_duplicates(['sentence', 'subject_entity', 'object_entity', 'label'])
    train_dataset = temp[~temp['id'].isin(dev_index)]

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenize = getattr(import_module(args.load_data_filename), args.load_data_func_tokenized)
  tokenize_train = getattr(import_module(args.load_data_filename), args.load_data_func_tokenized_train)
  tokenized_train = tokenize_train(train_dataset, tokenizer, args.tokenize)
  tokenized_dev = tokenize(dev_dataset, tokenizer, args.tokenize)


  # make dataset for pytorch.
  re_data = getattr(import_module(args.load_data_filename), args.load_data_class)
  RE_train_dataset = re_data(tokenized_train, train_label)
  RE_dev_dataset = re_data(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  # model_config =  AutoConfig.from_pretrained('./TAPT/adaptive/checkpoint-5500')
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = args.num_labels

  model_config.classifier_dropout = 0.0 # gives dropout to classifier layer

  # model =  AutoModelForSequenceClassification.from_pretrained('./TAPT/adaptive/checkpoint-5500', config=model_config)
  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  model.resize_token_embeddings(len(tokenizer))
  model.parameters
  model.to(device)

  wandb.init(project=args.project_name, entity=args.entity_name)
  wandb.run.name = args.run_name
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=args.output_dir,          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    save_steps=args.save_steps,            # model saving step.
    num_train_epochs=args.num_train_epochs,         # total number of training epochs
    learning_rate=args.learning_rate, # learning rate
    per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.per_device_eval_batch_size,   # batch size for evaluation
    #warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
    warmup_ratio=0.1,
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir=args.logging_dir,            # directory for storing logs
    logging_steps=args.logging_steps,              # log saving step.
    evaluation_strategy=args.evaluation_strategy, # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = args.eval_steps,            # evaluation step.
    load_best_model_at_end = args.load_best_model_at_end,
    report_to=args.report_to,
    metric_for_best_model=args.metric_for_best_model,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=True,
  )

  trainer = CustomTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  wandb.finish()

  model.save_pretrained(args.save_pretrained)

def main(args):
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # Data and model checkpoints directories
  parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
  parser.add_argument("--model", type=str, default="klue/bert-base", help="model to train (default: klue/bert-base)")
  parser.add_argument("--train_data", type=str, default="../dataset/train/train.csv", help="train_data directory (default: ../dataset/train/train.csv)")
  parser.add_argument("--num_labels", type=int, default=30, help="number of labels (default: 30)")
  parser.add_argument("--output_dir", type=str, default="./results", help="directory which stores various outputs (default: ./results)")
  parser.add_argument("--save_total_limit", type=int, default=5, help="max number of saved models (default: 5)")
  parser.add_argument("--save_steps", type=int, default=500, help="interval of saving model (default: 500)")
  parser.add_argument("--num_train_epochs", type=int, default=20, help="number of train epochs (default: 20)")
  parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate (default: 5e-5)")
  parser.add_argument("--per_device_train_batch_size", type=int, default=16, help=" (default: 16)")
  parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help=" (default: 16)")
  parser.add_argument("--warmup_steps", type=int, default=500, help=" (default: 500)")
  parser.add_argument("--weight_decay", type=float, default=0.01, help=" (default: 0.01)")
  parser.add_argument("--logging_dir", type=str, default="./logs", help=" (default: ./logs)")
  parser.add_argument("--logging_steps", type=int, default=100, help=" (default: 100)")
  parser.add_argument("--evaluation_strategy", type=str, default="steps", help=" (default: steps)")
  parser.add_argument("--eval_steps", type=int, default=500, help=" (default: 500)")
  parser.add_argument("--load_best_model_at_end", type=bool, default=True, help=" (default: True)")
  parser.add_argument("--save_pretrained", type=str, default="./best_model", help=" (default: ./best_model)")

  # updated
  parser.add_argument('--run_name', type=str, default="baseline")
  parser.add_argument('--tokenize', type=str, default="punct")
  parser.add_argument("--n_splits", type=int, default=1, help=" (default: )")
  parser.add_argument("--test_size", type=float, default=0.1, help=" (default: )")
  parser.add_argument("--project_name", type=str, default="Model_Test", help=" (default: )")
  parser.add_argument("--entity_name", type=str, default="growing_sesame", help=" (default: )")
  parser.add_argument("--report_to", type=str, default="wandb", help=" (default: )")
  parser.add_argument("--metric_for_best_model", type=str, default="eval_micro f1 score", help=" (default: )")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help=" (default: )")
  parser.add_argument("--use_augmentation", type=bool, default=False, help=" (default: False)")
  parser.add_argument("--aug_data", type=str, default="../dataset/train/augmented_vowelNoise.csv", help="(default: )")

  # load_data module
  parser.add_argument('--load_data_filename', type=str, default="load_data")
  parser.add_argument('--load_data_func_load', type=str, default="load_data")
  parser.add_argument('--load_data_func_tokenized', type=str, default="tokenized_dataset")
  parser.add_argument('--load_data_class', type=str, default="RE_Dataset")
  parser.add_argument('--load_data_func_tokenized_train', type=str, default="tokenized_dataset")
  
  args = parser.parse_args()
  print(args)

  seed_everything(args.seed)
  
  main(args)
