from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from modified_load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from importlib import import_module

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def ensemble_probs(filenames, weights=None, NUM_CLASS=30, output_filename="./ensemble.csv"):
    '''
    parameter:
        filenames (list) = ['./output.csv', './submission.csv']
        weights (list) = [0.7, 0.3]
        NUM_CLASS (int) = 30 [default]
        output_filename (str) = "./ensemble.csv" [default]
    output:
        soft-voting한 output_filename file 생성
    '''
    if not weights:
        weights = [1] * len(filenames)

    output_prob = []
    for fname, w in zip(filenames, weights):
        Mat = np.array(eval(','.join(list(pd.read_csv(fname)['probs']))))
        output_prob.append(w * Mat)
    
    if not weights:
        output_prob = np.mean(output_prob, 0).tolist()
    else:
        output_prob = np.sum(output_prob, 0).tolist()

    test_id = pd.read_csv(filenames[0])['id']
    pred_answer = num_to_label(np.argmax(output_prob, axis=1))

    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
    output.to_csv(output_filename, index=False)
    print('-- done --')

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  load = getattr(import_module(args.load_data_filename), args.load_data_func_load)
  test_dataset = load(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenize = getattr(import_module(args.load_data_filename), args.load_data_func_tokenized)
  tokenized_test = tokenize(test_dataset, tokenizer, args.special_entity_type, args.preprocess, args.clue_question)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = args.model
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME, additional_special_tokens=['#', '@'])

  ## load my model
  MODEL_NAME = args.model_dir # model dir.
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
  model.resize_token_embeddings(len(tokenizer))
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset_dir = args.test_dataset
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  re_data = getattr(import_module(args.load_data_filename), args.load_data_class)
  Re_test_dataset = re_data(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/'+args.file_name+'.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--test_dataset', type=str, default="../dataset/test/test_data.csv")
  parser.add_argument('--model_dir', type=str, default="./best_model")
  parser.add_argument('--special_entity_type', type=str, default="typed_entity", choices=["baseline", "punct", "entity", "typed_entity"], help="(default: typed_entity)")
  parser.add_argument('--preprocess', type=bool, default=False, help="apply preprocess")
  parser.add_argument('--clue_type', type=str, default="question", choices=["question", "entity"], help="(default: question)")
  parser.add_argument("--model", type=str, default="klue/bert-base", help="model to train (default: klue/bert-base)")
  parser.add_argument('--file_name', type=str, default="submission")

  # load_data module
  parser.add_argument('--load_data_filename', type=str, default="modified_load_data")
  parser.add_argument('--load_data_func_load', type=str, default="load_data")
  parser.add_argument('--load_data_func_tokenized', type=str, default="tokenized_dataset")
  parser.add_argument('--load_data_class', type=str, default="RE_Dataset")

  args = parser.parse_args()
  print(args)
  main(args)
  
