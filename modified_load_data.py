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
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  subject_start = []
  subject_end = []
  subject_type = []

  object_entity = []
  object_start = []
  object_end = []
  object_type = []

  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
      i_entity = i[1:-1].split(", '")[0].split(':')[1]
      i_start = int(i[1:-1].split(", '")[1].split(':')[1])
      i_end = int(i[1:-1].split(", '")[2].split(':')[1])
      i_type = i[1:-1].split(", '")[3].split(':')[1]
      
      j_entity = j[1:-1].split(", '")[0].split(':')[1]
      j_start = int(j[1:-1].split(", '")[1].split(':')[1])
      j_end = int(j[1:-1].split(", '")[2].split(':')[1])
      j_type = j[1:-1].split(", '")[3].split(':')[1]

      subject_entity.append(i_entity)
      subject_start.append(i_start)
      subject_end.append(i_end)
      subject_type.append(i_type)


      object_entity.append(j_entity)
      object_start.append(j_start)
      object_end.append(j_end)
      object_type.append(j_type)
      
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],
                              'subject_entity':subject_entity,'subject_start':subject_start,'subject_end':subject_end,'subject_type':subject_type,
                              'object_entity':object_entity,'object_start':object_start,'object_end':object_end,'object_type':object_type,
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

      sub = sub.replace("'", "").strip()
      obj = obj.replace("'", "").strip()
      
      if sub_start > obj_start:
        sent = sent[:sub_start-1] + " @ " + sent[sub_start:sub_end+1] + " @ " + sent[sub_end+1:]
        sent = sent[:obj_start-1] + " # " + sent[obj_start:obj_end+1] + " # " + sent[obj_end+1:]
      else:
        sent = sent[:obj_start-1] + " # " + sent[obj_start:obj_end+1] + " # " + sent[obj_end+1:]
        sent = sent[:sub_start-1] + " @ " + sent[sub_start:sub_end+1] + " @ " + sent[sub_end+1:]
    
      # 두개 이상의 공백 지우기 + 앞 뒤 공백 지우기
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
  elif type == "entity":
    sentences = list()
    
    for sent, sub, sub_start, sub_end, sub_type, obj, obj_start, obj_end, obj_type in zip(dataset['sentence'], 
                                                                                      dataset['subject_entity'], dataset['subject_start'], dataset['subject_end'], dataset['subject_type'], 
                                                                                      dataset['object_entity'],dataset['object_start'], dataset['object_end'], dataset['object_type']):
      sub = sub.replace("'", "").strip()
      special_sub = "<S:%s> " % (sub_type.replace("'", "").strip()) + sub + " </S:%s> " % (sub_type.replace("'", "").strip())
      obj = obj.replace("'", "").strip()
      special_obj = "<O:%s> " % (obj_type.replace("'", "").strip()) + obj + " </O:%s> " % (obj_type.replace("'", "").strip())

      if sub_start > obj_start:
          # subject token 달기
          sent = sent[:int(sub_start)-1] + special_sub + sent[int(sub_end)+1:]
          
          # object token 달기
          sent = sent[:int(obj_start)-1] + special_obj + sent[int(obj_end)+1:]
      else:
          # object token 달기
          sent = sent[:int(obj_start)-1] + special_obj + sent[int(obj_end)+1:]
          
          # subject token 달기
          sent = sent[:int(sub_start)-1] + special_sub + sent[int(sub_end)+1:]
      
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
  else:
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
