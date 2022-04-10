import pandas as pd
import torch
from torch.utils.data import Dataset
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

def get_entity_position_embedding(tokenizer, input_ids):
  special_token2id = {k:v for k,v in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)}

  sub_token_id = special_token2id['@']
  obj_token_id = special_token2id['#']
  
  pos_embeddings = []

  for y in input_ids:
    pos = []
      for j in range(0, len(y)):
        if len(pos) == 4:
          break
        if y[j] == start_token_id:
          pos.append(j)

        if y[j] == end_token_id:
          pos.append(j)
      pos_embeddings.append(pos)

  return torch.tensor(pos_embeddings, dtype=torch.int)


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

def tokenized_dataset(dataset, tokenizer, special_entity_type, preprocess):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  if special_entity_type == "punct":
    sentences = list()
    clue = list()
    
    for sent, sub, sub_start, sub_end, obj, obj_start, obj_end in zip(dataset['sentence'], 
                                                                      dataset['subject_entity'], dataset['subject_start'], dataset['subject_end'], 
                                                                      dataset['object_entity'],dataset['object_start'], dataset['object_end']):

      sent = list(sent)
      if sub_start > obj_start:
        sent.insert(sub_end + 1, ' @ ')
        sent.insert(sub_start, ' @ ')
        sent.insert(obj_end + 1, ' # ')
        sent.insert(obj_start, ' # ')
      else:
        sent.insert(obj_end + 1, ' # ')
        sent.insert(obj_start, ' # ')
        sent.insert(sub_end + 1, ' @ ')
        sent.insert(sub_start, ' @ ')
      sent = ''.join(sent)
    
      if preprocess:
        sent = re.sub("[^a-zA-Z가-힣0-9\@\#\<\>\:\/\"\'\,\.\?\!\-\+\%\$\(\)\~\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff ]", "", sent)

      question = '에서' + sub + '와(과) ' + obj + '은(는)?'

      sent = re.sub(r"\"+", '\"', sent).strip()
      sent = re.sub(r"\'+", "\'", sent).strip()
      sent = re.sub(r"\s+", " ", sent).strip()
      sentences.append(sent)
      clue.append(question)
    
    tokenized_sentences = tokenizer(
      sentences,
      clue,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      ) 
    
    tokenized_sentences['entity_position_embedding'] = get_entity_position_embedding(tokenizer, tokenized_sentences['input_ids'])
  
  elif special_entity_type == "entity":
    sentences = list()
    clue = list()
    
    for sent, sub, sub_start, sub_end, sub_type, obj, obj_start, obj_end, obj_type in zip(dataset['sentence'], 
                                                                                      dataset['subject_entity'], dataset['subject_start'], dataset['subject_end'], dataset['subject_type'], 
                                                                                      dataset['object_entity'],dataset['object_start'], dataset['object_end'], dataset['object_type']):
      
      special_sub = "<S:%s> " % (sub_type.replace("'", "").strip()) + sub + " </S:%s> " % (sub_type.replace("'", "").strip())
      special_obj = "<O:%s> " % (obj_type.replace("'", "").strip()) + obj + " </O:%s> " % (obj_type.replace("'", "").strip())

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
      
      if preprocess:
        sent = re.sub("[^a-zA-Z가-힣0-9\@\#\<\>\:\/\"\'\,\.\?\!\-\+\%\$\(\)\~\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff ]", "", sent)

      question = '에서' + sub + '와(과) ' + obj + '은(는)?'

      sent = re.sub(r"\"+", '\"', sent).strip()
      sent = re.sub(r"\'+", "\'", sent).strip()
      sent = re.sub(r"\s+", " ", sent).strip()

      sentences.append(sent)
      clue.append(question)
      
      
    tokenized_sentences = tokenizer(
      sentences,
      clue,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
    ) 

    tokenized_sentences['entity_position_embedding'] = get_entity_position_embedding(tokenizer, tokenized_sentences['input_ids'])
  
  elif special_entity_type == "typed_entity":
    sentences = list()
    clue = list()

    for sent, sub, sub_start, sub_end, sub_type, obj, obj_start, obj_end, obj_type in zip(dataset['sentence'], 
                                                                                        dataset['subject_entity'], dataset['subject_start'], dataset['subject_end'], dataset['subject_type'], 
                                                                                        dataset['object_entity'],dataset['object_start'], dataset['object_end'], dataset['object_type']):

      special_sub = " @ * %s * " % (sub_type.replace("'", "").strip()) + sub + " @ "
      special_obj = " # ^ %s ^ " % (obj_type.replace("'", "").strip()) + obj + " # "

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
      
      if preprocess:
        sent = re.sub("[^a-zA-Z가-힣0-9\@\#\<\>\:\/\"\'\,\.\?\!\-\+\%\$\(\)\~\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff ]", "", sent)

      question = '에서' + sub + '와(과) ' + obj + '은(는)?'

      sent = re.sub(r"\"+", '\"', sent).strip()
      sent = re.sub(r"\'+", "\'", sent).strip()
      sent = re.sub(r"\s+", " ", sent).strip()

      sentences.append(sent)
      clue.append(question)
        
    tokenized_sentences = tokenizer(
      sentences,
      clue,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
    )

    tokenized_sentences['entity_position_embedding'] = get_entity_position_embedding(tokenizer, tokenized_sentences['input_ids'])
              
        
  else: # baseline
    concat_entity = []
    
    for sub, obj in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = sub + '[SEP]' + obj
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
  
    tokenized_sentences['entity_position_embedding'] = get_entity_position_embedding(tokenizer, tokenized_sentences['input_ids'])
  
  return tokenized_sentences
