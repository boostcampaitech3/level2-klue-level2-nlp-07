import pandas as pd
import torch
from torch.utils.data import Dataset
import re

def preprocess(sent):
  return re.sub("[^a-zA-Z가-힣0-9\@\#\<\>\:\/\"\'\,\.\?\!\-\+\%\$\(\)\~\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff ]", "", sent)

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
      dict_i = eval(i)
      dict_j = eval(j)

      sub = dict_i['word'] # subj
      sub_start_idx = dict_i['start_idx'] # subj
      sub_end_idx = dict_i['end_idx'] # subj
      sub_type = dict_i['type'] # subj
      
      obj = dict_j['word'] # obj
      obj_start_idx = dict_j['start_idx'] # obj
      obj_end_idx = dict_j['end_idx'] # obj
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


def tokenized_dataset(dataset, tokenizer, special_entity_type, preprocess, clue_type):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  sentences = list()
  clues = list()
    
  for sent, sub, sub_start, sub_end, sub_type, obj, obj_start, obj_end, obj_type in zip(dataset['sentence'], 
                                                                                      dataset['subject_entity'], dataset['subject_start'], dataset['subject_end'], dataset['subject_type'], 
                                                                                      dataset['object_entity'],dataset['object_start'], dataset['object_end'], dataset['object_type']):

    # Special Entity Token
    if special_entity_type == "punct":
      special_sub = " @ " + sub + " @ "
      special_obj = " # " + obj + " # "
    
    elif special_entity_type == "entity":
      special_sub = " <S:%s> " % (sub_type.replace("'", "").strip()) + sub + " </S:%s> " % (sub_type.replace("'", "").strip())
      special_obj = " <O:%s> " % (obj_type.replace("'", "").strip()) + obj + " </O:%s> " % (obj_type.replace("'", "").strip())

    elif special_entity_type == "typed_entity":
      special_sub = " @ * %s * " % (sub_type.replace("'", "").strip()) + sub + " @ "
      special_obj = " # ^ %s ^ " % (obj_type.replace("'", "").strip()) + obj + " # "

    if special_entity_type != "baseline":
      if sub_start > obj_start:
          sent = sent[:int(sub_start)] + special_sub + sent[int(sub_end)+1:]
          sent = sent[:int(obj_start)] + special_obj + sent[int(obj_end)+1:]
      else:
          sent = sent[:int(obj_start)] + special_obj + sent[int(obj_end)+1:]
          sent = sent[:int(sub_start)] + special_sub + sent[int(sub_end)+1:]

    if preprocess:
      sent = preprocess(sent)

    if clue_type == "question":
      clue = '에서' + sub + '와(과) ' + obj + '은(는)?'
    else:
      clue = sub + '[SEP]' + obj

    sent = re.sub(r"\"+", '\"', sent).strip()
    sent = re.sub(r"\'+", "\'", sent).strip()
    sent = re.sub(r"\s+", " ", sent).strip()

    sentences.append(sent)
    clues.append(clue)

  if clue_type == "question":
    return tokenizer(
      sentences,
      clue,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
    ) 

  else:
    return tokenizer(
      clue,
      sentences,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
    ) 
