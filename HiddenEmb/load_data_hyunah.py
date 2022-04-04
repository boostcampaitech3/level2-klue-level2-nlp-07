### load_data.py 재구성
import torch
from torch.utils.data import Dataset
import re
import numpy as np
import pandas as pd
from typing import List
from collections import Counter
from itertools import chain
from konlpy.tag import Mecab, Okt, Hannanum
from konlpy import jvm
from tokenizers import BertWordPieceTokenizer, PreTokenizedString
from tokenizers import normalizers, pre_tokenizers
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import PreTokenizer, Digits, Punctuation, WhitespaceSplit
from transformers import BertTokenizer


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

# Normalization
def add_punct(sent, sub_ids, obj_ids):
  sub_start, sub_end = sub_ids
  obj_start, obj_end = obj_ids
  sent = list(sent) # str to list

  if sub_start > obj_start: # subj가 뒤에 위치
    sent.insert(sub_end + 1, ' @ ')
    sent.insert(sub_start, ' @ ')
    sent.insert(obj_end + 1, ' # ')
    sent.insert(obj_start, ' # ')
  else:
    sent.insert(obj_end + 1, ' # ')
    sent.insert(obj_start, ' # ')
    sent.insert(sub_end + 1, ' @ ')
    sent.insert(sub_start, ' @ ')

  return ''.join(sent)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  sentences = []
  subj_pos = []
  obj_pos = []

  for sent,i,j in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    subj_entity = eval(i) # str을 코드화
    obj_entity = eval(j)
    
    subject_entity.append(subj_entity['word']) # subj
    object_entity.append(obj_entity['word']) # obj
    subj_pos.append((subj_entity['start_idx'], subj_entity['end_idx']))
    obj_pos.append((obj_entity['start_idx'], obj_entity['end_idx']))
    sentences.append(add_punct(sent, subj_pos[-1], obj_pos[-1]))
    
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':sentences, 'subject_entity':subject_entity, 'object_entity':object_entity, 'label':dataset['label'],'subject_pos':subj_pos, 'object_pos':obj_pos,})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

# Pre-tokenization
# class FinePreTokenizer(): # base class?
#     def __init__(self, tokenizer):
#         self.mecab = Mecab()
#         self.okt = Okt()
        
#     def normalize_str(self, sentence: str):
#         # noun_unit_fn = lambda x: ' '.join(mecab.nouns(x))
#         i = 0
#         words = []
#         for x in self.mecab.nouns(sentence):            
#             words.append((x, (i, i+len(x))))
#             i += len(x)+1
#         return words
    
#     def pre_tokenize(self, pretok: PreTokenizedString):
#         pretok = normalize_str(pretok) # process?


# The model
def build_bpe(
    corpus: List[str],
    max_vocab_size: int,
    WORD_END = '##'
) -> List[int]:
    vocab = list(set(chain.from_iterable(corpus)) | {WORD_END})
    corpus = {' '.join(word + WORD_END): count for word, count in Counter(corpus).items()}

    while len(vocab) < max_vocab_size:
        counter = Counter() # bigram 개수
        for word, word_count in corpus.items():
            word = word.split()
            counter.update({
                pair: count * word_count 
                for pair, count in Counter(zip(word, word[1:])).items()
            })

        if not counter:
            break
        
        pair = counter.most_common(1)[0][0]
        vocab.append(''.join(pair))
        corpus = {
            word.replace(' '.join(pair), ''.join(pair)): count
            for word, count in corpus.items()
        }

    id2token = sorted(vocab, key=len, reverse=True)    
    return id2token

def create_vocab_by_corpus(
    vocab_fname: str,
    corpus: List[str], 
    max_vocab_size: int, 
    WORD_END = '##'
):
    tokenizer = BertWordPieceTokenizer(
        None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=True, # Must be False if cased model
        lowercase=False,
        wordpieces_prefix="##",
    )
    tokenizer.train_from_iterator(
        corpus, max_vocab_size, min_frequency=2
    )
    
    text2id = {x:i+5 for i, x in enumerate(corpus)}
    id2text = sorted({v:k for k, v in text2id.items()}.items())
    vocab = list(np.array(id2text)[:, 1])
    
    with open(vocab_fname, 'w+') as f:
        f.write('\n'.join(vocab))

# Post-processing





def get_entity_position_embedding(tokenizer, input_ids):
    special_token2id = {k:v for k,v in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)}
    start_token_id = special_token2id['@']
    end_token_id = special_token2id['#']
    
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

def tokenized_dataset(dataset, tokenizer, type=""): # df, AutoTokenizer
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):

#     pos = [0]*len(x)
#     for i in range(ss, se+1):
#       pos[i] = 1 
#     for i in range(os, oe+1):
#       pos[i] = 2
        
    temp = ''
    temp = e01 + '[SEP]' + e02 #+ '[SEP]' + ''.join(pos)
    concat_entity.append(temp)

  # vocab 추가하기
  # for x in list(dataset['subject_entity']):
  #   tokenizer.add_tokens([x])
  # for y in list(dataset['object_entity']):
  #   tokenizer.add_tokens([y])

#   subj_vocab = list(set(list(dataset['subject_entity'])))
#   obj_vocab = list(set(list(dataset['object_entity'])))
#   types = ['<PER>', '<LOC>', '<POH>', '<DAT>', '<ORG>', '<NOH>']
#   tokenizer.train_new_from_iterator(subj_vocab, vocab_size=tokenizer.vocab_size+len(subj_vocab))
#   tokenizer.train_new_from_iterator(obj_vocab, vocab_size=tokenizer.vocab_size+len(obj_vocab))
#   tokenizer.train_new_from_iterator(types, vocab_size=tokenizer.vocab_size+len(types))

  # tokenizer.pre_tokenizers = tokenizer.pre_tokenizers.append(Mecab())
  tokenized_sentences = tokenizer(
      concat_entity, # subj, obj
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
  )
  tokenized_sentences['entity_position_embedding'] = get_entity_position_embedding(tokenizer, tokenized_sentences['input_ids'])


  # for x in tokenized_sentences['input_ids']:
  #   li = x.tolist()
  #   subject_token_end = li.index(tokenizer.sep_token_id)+1
  #   object_token_end = li.index(tokenizer.sep_token_id, subject_token_end)+1
    

  return tokenized_sentences
