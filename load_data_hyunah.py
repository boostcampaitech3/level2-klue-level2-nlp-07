### load_data.py 재구성
from load_data import RE_Dataset
import re
import pandas as pd
from typing import List
from collections import Counter
from itertools import chain
from konlpy.tag import Mecab


# Normalization
# !! 중복된 문장 제거
def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  # sentences = []
  for sent,i,j in zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']):
    dict_i = eval(i) # str을 코드화
    dict_j = eval(j)
    i = '<{0}>{1}'.format(dict_i['type'], dict_i['word']) # subj
    j = '<{0}>{1}'.format(dict_j['type'], dict_j['word']) # obj
    
    # sent = sent.replace(dict_i['word'], i)
    # sent = sent.replace(dict_j['word'], j)
    
    # sent = re.sub(f"[{chr(0x4e00)}-{chr(0x9fff)}]", "", sent) # 한자 제거

    subject_entity.append(i)
    object_entity.append(j)
    # sentences.append(sent)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'], 'subject_entity':subject_entity, 'object_entity':object_entity, 'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

# Pre-tokenization
def split_by_morphs(sentences):
    mecab = Mecab()
    
    morph_unit_fn = lambda x: ' '.join(mecab.morphs(x))
    noun_unit_fn = lambda x: ' '.join(mecab.nouns(x))
    
    return list(sentences.apply(noun_unit_fn))

# The model
def build_bpe(
    corpus: List[str],
    max_vocab_size: int,
    WORD_END = '##'
) -> List[int]:
    """ BPE Vocab 만들기
    Byte Pair Encoding을 통한 Vocab 생성을 구현하세요.
    단어의 끝은 '_'를 사용해 주세요.
    이때 id2token을 서브워드가 긴 길이 순으로 정렬해 주세요.
    
    Note: 만약 모든 단어에 대해 BPE 알고리즘을 돌리게 되면 매우 비효율적입니다.
          왜냐하면 대부분의 단어는 중복되기 때문에 중복되는 단어에 대해서는 한번만 연산할 수 있다면 매우 효율적이기 때문입니다.
          따라서 collections 라이브러리의 Counter를 활용해 각 단어의 빈도를 구하고,
          각 단어에 빈도를 가중치로 활용하여 BPE를 돌리면 시간을 획기적으로 줄일 수 있습니다.
          물론 이는 Optional한 요소입니다.

    Arguments:
    corpus -- Vocab을 만들기 위한 단어 리스트
    max_vocab_size -- 최대 vocab 크기

    Return:
    id2token -- 서브워드 Vocab. 문자열 리스트 형태로 id로 token을 찾는 매핑으로도 활용 가능
    """
    ### YOUR CODE HERE
    ### ANSWER HERE ###
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

def tokenized_dataset(dataset, tokenizer): # df, AutoTokenizer
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
    
  tokenized_sentences = tokenizer(
      concat_entity, # subj, obj
      split_by_morphs(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,)
  tokenized_sentences['entity_position_embedding'] = get_entity_position_embedding(tokenizer, tokenized_sentences['input_ids'])

  return tokenized_sentences
