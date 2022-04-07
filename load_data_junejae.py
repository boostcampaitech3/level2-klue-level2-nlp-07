import pickle as pickle
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from typing import Tuple
from sklearn.model_selection import StratifiedShuffleSplit
from ktextaug import synonym_replace
from ktextaug.tokenization_utils import Tokenizer
import random


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


def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  """ dataset = preprocessing_dataset(pd_dataset)
  
  return dataset """
  return pd_dataset

def tokenized_dataset(dataset, tokenizer, type):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  concat_sentence = []
  # concat_whole = []

  for sentence, e01, e02 in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    #temp = e01 + '[SEP]' + e02

    dict_e01 = eval(e01) # str을 코드화
    dict_e02 = eval(e02)

    e01_word = dict_e01['word']
    e02_word = dict_e02['word']
    final_sentence = sentence

    
    if dict_e01['start_idx'] <= dict_e02['start_idx']:
      sentence_left = sentence[:dict_e01['start_idx']]
      sentence_subject = sentence[dict_e01['start_idx']:dict_e01['end_idx']+1]
      sentence_middle = sentence[dict_e01['end_idx']+1:dict_e02['start_idx']]
      sentence_object = sentence[dict_e02['start_idx']:dict_e02['end_idx']+1]
      sentence_right = sentence[dict_e02['end_idx']+1:]

      sentence_subject = '@*'+dict_e01['type']+'*'+sentence_subject+'@'
      sentence_object = '#^'+dict_e02['type']+'^'+sentence_object+'#'

      final_sentence = sentence_left + sentence_subject + sentence_middle + sentence_object + sentence_right
    
    else:
      sentence_left = sentence[:dict_e02['start_idx']]
      sentence_object = sentence[dict_e02['start_idx']:dict_e02['end_idx']+1]
      sentence_middle = sentence[dict_e02['end_idx']+1:dict_e01['start_idx']]
      sentence_subject = sentence[dict_e01['start_idx']:dict_e01['end_idx']+1]
      sentence_right = sentence[dict_e01['end_idx']+1:]

      sentence_subject = '@*'+dict_e01['type']+'*'+sentence_subject+'@'
      sentence_object = '#^'+dict_e02['type']+'^'+sentence_object+'#'

      final_sentence = sentence_left + sentence_object + sentence_middle + sentence_subject + sentence_right

    temp = e01_word + '와 ' + e02_word + '의 관계는?'
    # temp = '두 객체의 관계를 추론하자.'
    # temp = e01_word + ' 그리고 ' + e02_word + ' 사이의 관계는?[SEP]'
    # temp = ''
    # temp = '이 문장에서' + e01 + '과 ' + e02 + '은 어떤 관계일까?'
    # temp = '다음 문장에서' + e01 + '과 ' + e02 + '은 어떤 관계일까?'
    concat_entity.append(temp)
    concat_sentence.append(final_sentence)

    # concat_whole.append(temp+ final_sentence)

  tokenized_sentences = tokenizer(
      concat_entity,
      concat_sentence,
      # concat_whole,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=128, # default: 256
      add_special_tokens=True,
      )

  return tokenized_sentences

def tokenized_dataset_difficult(dataset, tokenizer, type):
  random.seed(42)
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  concat_sentence = []
  concat_whole = []

  for sentence, e01, e02 in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    #temp = e01 + '[SEP]' + e02

    dict_e01 = eval(e01) # str을 코드화
    dict_e02 = eval(e02)

    e01_word = dict_e01['word']
    e02_word = dict_e02['word']
    final_sentence = sentence
    
    if dict_e01['start_idx'] <= dict_e02['start_idx']:
      sentence_left = sentence[:dict_e01['start_idx']]
      sentence_subject = sentence[dict_e01['start_idx']:dict_e01['end_idx']+1]
      sentence_middle = sentence[dict_e01['end_idx']+1:dict_e02['start_idx']]
      sentence_object = sentence[dict_e02['start_idx']:dict_e02['end_idx']+1]
      sentence_right = sentence[dict_e02['end_idx']+1:]

      sentence_subject = '@*'+dict_e01['type']+'*'+sentence_subject+'@'
      sentence_object = '#^'+dict_e02['type']+'^'+sentence_object+'#'

      final_sentence = sentence_left + sentence_subject + sentence_middle + sentence_object + sentence_right
    
    else:
      sentence_left = sentence[:dict_e02['start_idx']]
      sentence_object = sentence[dict_e02['start_idx']:dict_e02['end_idx']+1]
      sentence_middle = sentence[dict_e02['end_idx']+1:dict_e01['start_idx']]
      sentence_subject = sentence[dict_e01['start_idx']:dict_e01['end_idx']+1]
      sentence_right = sentence[dict_e01['end_idx']+1:]

      sentence_subject = '@*'+dict_e01['type']+'*'+sentence_subject+'@'
      sentence_object = '#^'+dict_e02['type']+'^'+sentence_object+'#'

      final_sentence = sentence_left + sentence_object + sentence_middle + sentence_subject + sentence_right

    temp = e01_word + ' 그리고 ' + e02_word + ' 사이의 관계는?[SEP]'
    if random.randrange(0,2) == 0:
      temp = ''
    #temp = '이 문장에서' + e01 + '과 ' + e02 + '은 어떤 관계일까?'
    #temp = '다음 문장에서' + e01 + '과 ' + e02 + '은 어떤 관계일까?'

    """ concat_entity.append(temp)
    concat_sentence.append(final_sentence) """

    concat_whole.append(temp + final_sentence)

  tokenized_sentences = tokenizer(
      #concat_entity,
      #concat_sentence,
      concat_whole,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=128, # default: 256
      add_special_tokens=True,
      )

  return tokenized_sentences
