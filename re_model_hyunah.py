import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification, BertForSequenceClassification
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np


class ReModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        
        MODEL_NAME = args.model
        self.config =  AutoConfig.from_pretrained(MODEL_NAME)
        self.num_labels = args.num_labels
        self.model =  AutoModel.from_pretrained(MODEL_NAME, config=self.config)
        self.model.resize_token_embeddings(len(tokenizer))
        
        hidden_size = self.config.hidden_size
        batch_size = args.per_device_train_batch_size
        dropout_prob = self.config.attention_probs_dropout_prob
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, args.num_labels)
        )
        
    
    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_position_embedding = None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[0] # logits
        # print('---- pooled_output shape: ', pooled_output.shape) # 16, 250, 768
        # print('---- entity_position_embedding.shape: ', entity_position_embedding.shape) # 16, 4

        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        entity_position_embedding = entity_position_embedding.T
        ss_emb = pooled_output[idx, entity_position_embedding[0].tolist()]
        se_emb = pooled_output[idx, entity_position_embedding[1].tolist()]
        os_emb = pooled_output[idx, entity_position_embedding[2].tolist()]
        oe_emb = pooled_output[idx, entity_position_embedding[3].tolist()]
        # print('---- ss_emb.shape: ', ss_emb.shape)
        
        h = torch.cat((ss_emb, se_emb, os_emb, oe_emb), dim=-1).to(input_ids.device)
        # print('---- h.shape: ', h.shape) # 16, 16, 3072
        
        logits = self.classifier(h)        
        outputs = (logits,)
        if labels is not None:
            # print('---- logits:', logits.shape, ', labels: ', labels)
            loss = self.loss_fn(logits.float(), labels)
            outputs = (loss, ) + outputs
        
        return outputs
