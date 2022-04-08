import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, RobertaPreTrainedModel
from torch.cuda.amp import autocast
from loss import *


class ReModel(RobertaPreTrainedModel):
    def __init__(self, args, tokenizer, emb_no=4):
        self.args = args
        
        MODEL_NAME = args.model
        self.config =  AutoConfig.from_pretrained(MODEL_NAME)
        super().__init__(self.config)

        self.num_labels = args.num_labels
        self.model = AutoModel.from_pretrained(MODEL_NAME, config=self.config)
        self.model.resize_token_embeddings(len(tokenizer))
        
        hidden_size = self.config.hidden_size
        dropout_prob = self.config.attention_probs_dropout_prob
        
        if args.loss == "focal":
            self.loss_fn = FocalLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, self.num_labels)
        )
        
    
    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_position_embedding = None,
        position_ids=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pooled_output = outputs[0]
        
        if self.emb_no == 4:
            idx = torch.arange(input_ids.size(0)).to(input_ids.device)
            entity_position_embedding = entity_position_embedding.T
            ss_emb = pooled_output[idx, entity_position_embedding[0].tolist()]
            se_emb = pooled_output[idx, entity_position_embedding[1].tolist()]
            os_emb = pooled_output[idx, entity_position_embedding[2].tolist()]
            oe_emb = pooled_output[idx, entity_position_embedding[3].tolist()]
            
            h = torch.cat((
                ss_emb, 
                se_emb, 
                os_emb, 
                oe_emb
            ), dim=-1).to(input_ids.device)
        
        elif self.emb_no == 2:
            idx = torch.arange(input_ids.size(0)).to(input_ids.device)
            entity_position_embedding = entity_position_embedding.T
            ss_emb = pooled_output[idx, entity_position_embedding[0].tolist()]
            os_emb = pooled_output[idx, entity_position_embedding[2].tolist()]
            
            h = torch.cat((
                ss_emb, 
                os_emb, 
            ), dim=-1).to(input_ids.device)
        
        logits = self.classifier(h)        
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fn(logits.float(), labels)
            outputs = (loss, ) + outputs
        
        return outputs
