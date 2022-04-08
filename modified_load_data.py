import pickle as pickle
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from typing import Tuple
from sklearn.model_selection import StratifiedShuffleSplit
import re

type_dict = {"PER": "사람", "ORG": "기관", "LOC": "위치", "DAT": "시간", "POH": "명사", "NOH": "수사"}

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

def get_entity_position_embedding(tokenizer, input_ids):
    special_token2id = {k:v for k,v in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)}

    sub_token_id = 36 #special_token2id['@'] # 36
    obj_token_id = 7 # special_token2id['#'] # 7
    
    pos_embeddings = []

    for y in input_ids:
      ss_embedding = [0]
      os_embedding = []
      for j in range(0, len(y)):
          if len(ss_embedding) + len(os_embedding) == 5:
              break
          if y[j] == sub_token_id:
              ss_embedding.append(j)
          if y[j] == obj_token_id:
              os_embedding.append(j)
          
      pos = ss_embedding + os_embedding
       
      pos_embeddings.append(pos)
      
    return torch.tensor(pos_embeddings, dtype=torch.int)
  
      
def preprocessing_dataset(dataset):
  subject_entity = []
  subject_start = []
  subject_end = []
  subject_type = []


  object_entity = []
  object_start = []
  object_end = []
  object_type = []


  # sentences = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
      dict_i = eval(i) # str을 코드화
      dict_j = eval(j)
      sub = dict_i['word'] # subj
      sub_start_idx = dict_i['start_idx'] # subj
      sub_end_idx = dict_i['end_idx'] # subj
      #sub_type = type_dict[dict_i['type']] # subj
      sub_type = dict_i['type'] # subj
      
      obj = dict_j['word'] # obj
      obj_start_idx = dict_j['start_idx'] # obj
      obj_end_idx = dict_j['end_idx'] # obj
      #obj_type = type_dict[dict_j['type']] # obj
      obj_type = dict_j['type'] # obj


      subject_entity.append(sub)
      subject_start.append(sub_start_idx)
      subject_end.append(sub_end_idx)
      subject_type.append(sub_type)
      
      object_entity.append(obj)
      object_start.append(obj_start_idx)
      object_end.append(obj_end_idx)
      object_type.append(obj_type)
      
      
      

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
    entity = list()
    
    for sent, sub, sub_start, sub_end, obj, obj_start, obj_end in zip(dataset['sentence'], 
                                                                      dataset['subject_entity'], dataset['subject_start'], dataset['subject_end'], 
                                                                      dataset['object_entity'],dataset['object_start'], dataset['object_end']):

      temp = sub + '[SEP]' + obj
      
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
      entity.append(temp)
    
    tokenized_sentences = tokenizer(
      #entity,
      sentences,
      entity,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      ) 
  
  elif type == "entity":
    sentences = list()
    entity = list()
    
    for sent, sub, sub_start, sub_end, sub_type, obj, obj_start, obj_end, obj_type in zip(dataset['sentence'], 
                                                                                      dataset['subject_entity'], dataset['subject_start'], dataset['subject_end'], dataset['subject_type'], 
                                                                                      dataset['object_entity'],dataset['object_start'], dataset['object_end'], dataset['object_type']):
      
      special_sub = "<S:%s> " % (sub_type.replace("'", "").strip()) + sub + " </S:%s> " % (sub_type.replace("'", "").strip())
      special_obj = "<O:%s> " % (obj_type.replace("'", "").strip()) + obj + " </O:%s> " % (obj_type.replace("'", "").strip())
      temp = '이 문장에서' + sub + '와(과) ' + obj + '은(는) 어떤 관계일까?'

      if sub_start > obj_start:
          # subject token 달기
          sent = sent[:int(sub_start)] + special_sub + sent[int(sub_end)+1:]
          
          # object token 달기
          sent = sent[:int(obj_start)] + special_obj + sent[int(obj_end)+1:]
      else:
          # object token 달기
          sent = sent[:int(obj_start)] + special_obj + sent[int(obj_end)+1:]
          
          # subject token 달기
          sent = sent[:int(sub_start)] + special_sub + sent[int(sub_end)+1:]
      
      sent = re.sub(r"\s+", " ", sent).strip()
      sentences.append(sent)
      entity.append(temp)
      
      
    tokenized_sentences = tokenizer(
      #entity,
      sentences,
      entity,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
    ) 
  
  elif type == "typed_entity":
    # Typed entity marker (punct)

    sentences = list()
    entity = list()

    for sent, sub, sub_start, sub_end, sub_type, obj, obj_start, obj_end, obj_type in zip(dataset['sentence'], 
                                                                                        dataset['subject_entity'], dataset['subject_start'], dataset['subject_end'], dataset['subject_type'], 
                                                                                        dataset['object_entity'],dataset['object_start'], dataset['object_end'], dataset['object_type']):

      special_sub = " @ * %s * " % (sub_type.replace("'", "").strip()) + sub + " @ "
      special_obj = " # ^ %s ^ " % (obj_type.replace("'", "").strip()) + obj + " # "
      #temp = sub + '[SEP]' + obj

      if sub_start > obj_start:
          # subject token 달기
          sent = sent[:int(sub_start)] + special_sub + sent[int(sub_end)+1:]
          
          # object token 달기
          sent = sent[:int(obj_start)] + special_obj + sent[int(obj_end)+1:]
      else:
          # object token 달기
          sent = sent[:int(obj_start)] + special_obj + sent[int(obj_end)+1:]
          
          # subject token 달기
          sent = sent[:int(sub_start)] + special_sub + sent[int(sub_end)+1:]
      
      sent = re.sub("[^a-zA-Z가-힣0-9\@\#\<\>\:\/\"\'\,\.\?\!\-\+\*\^\%\$\(\)\~\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff ]", "", sent)

      sent = re.sub(r"\"+", '\"', sent).strip()
      sent = re.sub(r"\'+", "\'", sent).strip()
      sent = re.sub(r"\s+", " ", sent).strip()

      temp = 'in ' + sub + ' and ' + obj + ' are?'
      # temp = "에서 @와 #의 관계는?"
      sentences.append(sent)
      entity.append(temp)
        
    tokenized_sentences = tokenizer(
      #entity,
      sentences,
      entity,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
    )
    
    # tokenized_sentences['entity_position_embedding'] = get_entity_position_embedding(tokenizer, tokenized_sentences['input_ids'])
              
        
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
