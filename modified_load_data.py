import pickle as pickle
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from typing import Tuple
from sklearn.model_selection import StratifiedShuffleSplit
import re


class RE_Dataset(Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)
    
def preprocessing_dataset(dataset):
  subject_entity = []
  subject_start = []
  subject_end = []


  object_entity = []
  object_start = []
  object_end = []


  # sentences = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
      dict_i = eval(i) # str을 코드화
      dict_j = eval(j)
      print(dict_i)
      sub = dict_i['word'] # subj
      sub_start_idx = dict_i['start_idx'] # subj
      sub_end_idx = dict_i['end_idx'] # subj
      
      obj = dict_j['word'] # obj
      obj_start_idx = dict_j['start_idx'] # obj
      obj_end_idx = dict_j['end_idx'] # obj

      subject_entity.append(sub)
      subject_start.append(sub_start_idx)
      subject_end.append(sub_end_idx)
      
      object_entity.append(obj)
      object_start.append(obj_start_idx)
      object_end.append(obj_end_idx)
      
      
      

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],
                              'subject_entity':subject_entity,'subject_start':subject_start,'subject_end':subject_end,
                              'object_entity':object_entity,'object_start':object_start,'object_end':object_end,
                              'label':dataset['label']})
  return out_dataset


def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer, type):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  if type == "punct":
    sentences = list()
    
    for sent, sub, sub_start, sub_end, obj, obj_start, obj_end in zip(dataset['sentence'], 
                                                                      dataset['subject_entity'], dataset['subject_start'], dataset['subject_end'], 
                                                                      dataset['object_entity'],dataset['object_start'], dataset['object_end']):

      sub = sub[1:-1]
      obj = obj[1:-1]
      
      if sub_start > obj_start:
        sent = sent[:sub_start-1] + " @ " + sent[sub_start:sub_end+1] + " @ " + sent[sub_end+1:]
        sent = sent[:obj_start-1] + " # " + sent[obj_start:obj_end+1] + " # " + sent[obj_end+1:]
      else:
        sent = sent[:obj_start-1] + " # " + sent[obj_start:obj_end+1] + " # " + sent[obj_end+1:]
        sent = sent[:sub_start-1] + " @ " + sent[sub_start:sub_end+1] + " @ " + sent[sub_end+1:]
    
      # 두개 이상의 공백 지우기 + 앞 뒤 공백 지우기
      sent = re.sub("[^a-zA-Z가-힣0-9\@\#\<\>\:\/\"\'\,\.\?\!\-\+\%\$\(\)\~\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff ]", "", sent)

      sent = re.sub(r"\"+", '\"', sent).strip()
      sent = re.sub(r"\'+", "\'", sent).strip()
      sent = re.sub(r"\s+", " ", sent).strip()
      sentences.append(sent)
    
    tokenized_sentences = tokenizer(
      sentences,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      ) 
    
  else: # baseline
    concat_entity = []
    
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)
      
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        )
  

  return tokenized_sentences
