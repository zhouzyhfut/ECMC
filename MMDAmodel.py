import torch
import torch.nn as nn
import lightning.pytorch as pl
from module.Qformer import BertConfig, BertLMHeadModel
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    BertTokenizer, 
    BertModel,
    LlamaTokenizer
)
from module.modeling_llama import LlamaForCausalLM
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False

class ECMC(pl.LightningModule):
    def __init__(
        self,
        text2vec_ckpt="weights"):
        super(ECMC,self).__init__()
        self.training_step_outputs  = []
        
        #path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        text2vec_ckpt = os.path.join(current_directory, text2vec_ckpt)

        #text2vec
        self.text2vec_model=BertModel.from_pretrained(text2vec_ckpt)
        self.text2vec_tokenizer=BertTokenizer.from_pretrained(text2vec_ckpt)

        for p in self.parameters():
            p.requires_grad = False

        #Qformer-audio-emo
        self.audio_Qformer,self.audio_query_tokens=self.init_Qformer(num_query_token=32,vision_width=768)
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.audio_llama_project=nn.Linear(768,4096)

        #Qformer-video-emo
        self.video_Qformer,self.video_query_tokens=self.init_Qformer(num_query_token=32,vision_width=768)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        #Qformer-text-emo
        self.text_Qformer,self.text_query_tokens=self.init_Qformer(num_query_token=32,vision_width=768)
        self.text_Qformer.cls = None
        self.text_Qformer.bert.embeddings.word_embeddings = None
        self.text_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.text_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        #Qformer-audio-cong
        self.audio_Qformer_cong,self.audio_query_tokens_cong=self.init_Qformer(num_query_token=32,vision_width=768)
        self.audio_Qformer_cong.cls = None
        self.audio_Qformer_cong.bert.embeddings.word_embeddings = None
        self.audio_Qformer_cong.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer_cong.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        #Qformer-video-cong
        self.video_Qformer_cong,self.video_query_tokens_cong=self.init_Qformer(num_query_token=32,vision_width=768)
        self.video_Qformer_cong.cls = None
        self.video_Qformer_cong.bert.embeddings.word_embeddings = None
        self.video_Qformer_cong.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer_cong.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        #Qformer-text-cong
        self.text_Qformer_cong,self.text_query_tokens_cong=self.init_Qformer(num_query_token=32,vision_width=768)
        self.text_Qformer_cong.cls = None
        self.text_Qformer_cong.bert.embeddings.word_embeddings = None
        self.text_Qformer_cong.bert.embeddings.position_embeddings = None
        for layer in self.text_Qformer_cong.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        

    def init_Qformer(self,num_query_token, vision_width, cross_attention_freq=2):
        path=os.path.dirname(os.path.abspath(__file__))
        config_path=os.path.join(path,"weights")
        encoder_config = BertConfig.from_pretrained(config_path)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        ckpt=os.path.join(path,"pytorch_model.bin")
        Qformer.load_state_dict(torch.load(ckpt),strict=False)

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    
    def forward(self, audio, video, text, emo_category, cognition_category):
        #audio
        audio_feature=audio

        #text2vec
        with torch.no_grad():
            #describtion
            describtion=[s+"</s>" for s in text]
            describtion_input=self.text2vec_tokenizer(describtion, padding=True, truncation=True, return_tensors='pt').to(self.device)
            describtion_feature=self.text2vec_model(**describtion_input)
            describtion_feature=self.mean_pooling(describtion_feature,describtion_input['attention_mask']).unsqueeze(1)
            describtion_feature = describtion_feature.to(audio_feature.dtype)

        #Qformer-emotion-----------------------------------------------------
        audio_query_tokens=self.audio_query_tokens.expand(audio_feature.shape[0], -1, -1)
        frame_atts_audio = torch.ones(audio_feature.size()[:-1], dtype=torch.long).to(audio_feature.device)
        #print(audio_query_tokens.shape,audio_feature.shape,frame_atts.shape)
        audio_query_output=self.audio_Qformer.bert(
            query_embeds=audio_query_tokens, #[32,768]
            encoder_hidden_states=audio_feature,
            encoder_attention_mask=frame_atts_audio,
            return_dict=True,
            )
        audio_hidden=audio_query_output.last_hidden_state

        video_feature=video
        video_query_tokens=self.video_query_tokens.expand(video_feature.shape[0], -1, -1)
        frame_atts_video = torch.ones(video_feature.size()[:-1], dtype=torch.long).to(video_feature.device)

        video_query_output=self.video_Qformer.bert(
            query_embeds=video_query_tokens, #[32,768]
            encoder_hidden_states=video_feature,
            encoder_attention_mask=frame_atts_video,
            return_dict=True,
            )
        video_hidden=video_query_output.last_hidden_state

        text_feature=describtion_feature
        text_query_tokens=self.text_query_tokens.expand(text_feature.shape[0], -1, -1)
        frame_atts_text = torch.ones(text_feature.size()[:-1], dtype=torch.long).to(text_feature.device)

        text_query_output=self.text_Qformer.bert(
            query_embeds=text_query_tokens, #[32,768]
            encoder_hidden_states=text_feature,
            encoder_attention_mask=frame_atts_text,
            return_dict=True,
            )
        text_hidden=text_query_output.last_hidden_state
        combined_feature_emo = torch.cat([audio_hidden, video_hidden, text_hidden], dim=1)

        #Qformer-congnition---------------------------------------------------
        audio_query_tokens_cong=self.audio_query_tokens_cong.expand(audio_feature.shape[0], -1, -1)
        audio_query_output=self.audio_Qformer_cong.bert(
            query_embeds=audio_query_tokens_cong, 
            encoder_hidden_states=audio_feature,
            encoder_attention_mask=frame_atts_audio,
            return_dict=True,
            )
        audio_hidden=audio_query_output.last_hidden_state

        video_query_tokens_cong=self.video_query_tokens_cong.expand(video_feature.shape[0], -1, -1)
        video_query_output=self.video_Qformer_cong.bert(
            query_embeds=video_query_tokens_cong,
            encoder_hidden_states=video_feature,
            encoder_attention_mask=frame_atts_video,
            return_dict=True,
            )
        video_hidden=video_query_output.last_hidden_state

        text_query_tokens_cong=self.text_query_tokens_cong.expand(text_feature.shape[0], -1, -1)
        text_query_output=self.text_Qformer_cong.bert(
            query_embeds=text_query_tokens_cong,
            encoder_hidden_states=text_feature,
            encoder_attention_mask=frame_atts_text,
            return_dict=True,
            )
        text_hidden=text_query_output.last_hidden_state
        combined_feature_cong = torch.cat([audio_hidden, video_hidden, text_hidden], dim=1)

        #Loss---------------------------------------------------
        emo_loss = self.contrastive_loss(
        features=combined_feature_emo,
        labels=emo_category, 
        temperature=0.1
        )
    
        cog_loss = self.multilabel_contrastive_loss(
        features=combined_feature_cong,
        labels=[row.nonzero(as_tuple=True)[0].tolist() for row in cognition_category],  # 转类别索引
        temperature=0.1
        )

        return emo_loss + cog_loss, combined_feature_emo, combined_feature_cong
    
    def contrastive_loss(self, features, labels, temperature=0.1):
        features = features.mean(dim=1)
        features = F.normalize(features, p=2, dim=1) 
        logits = torch.matmul(features, features.T) / temperature  
        labels = labels.view(-1)
        assert logits.shape[0] == labels.shape[0]

        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  

        mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        logits_masked = logits.masked_select(mask).view(logits.size(0), -1)
        label_matrix_masked = label_matrix.masked_select(mask).view(logits.size(0), -1)

        log_prob = logits_masked - torch.logsumexp(logits_masked, dim=1, keepdim=True)

        positives = label_matrix_masked.float()
        mean_log_prob_pos = (positives * log_prob).sum(1) / positives.sum(1).clamp(min=1)

        intra_loss = -mean_log_prob_pos.mean()
        
        negatives = (~label_matrix_masked).float()
        inter_loss = torch.log1p(torch.exp(logits_masked) * negatives).mean()
            
        return intra_loss + inter_loss
    
    
    def multilabel_contrastive_loss(self, features, labels, temperature=0.1):
        features = features.mean(dim=1)
        features = F.normalize(features, p=2, dim=1)  
        sim_matrix = torch.matmul(features, features.T) / temperature  
    
        weight_matrix = torch.zeros_like(sim_matrix)
        for i, lbls_i in enumerate(labels):
            for j, lbls_j in enumerate(labels):
                set_i = set(lbls_i)
                set_j = set(lbls_j)

                if not set_i and not set_j:
                    sim = 1.0
                elif not set_i or not set_j:
                    sim = 0.0
                else:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    sim = intersection / union if union > 0 else 0.0

                weight_matrix[i, j] = sim
        
        pos_weight = (weight_matrix > 0).float()
        neg_weight = (weight_matrix == 0).float()
        
        exp_sim = torch.exp(sim_matrix)
        pos_term = -torch.log(
            (exp_sim * pos_weight).sum(dim=1) / 
            (exp_sim.sum(dim=1) + 1e-8)
        )
        neg_term = torch.log(1 + (exp_sim * neg_weight).sum(dim=1))
        
        return (pos_term + neg_term).mean()

    def training_step(self, batch, batch_idx):
        audio, video, text, emo_category, cognition_category = batch['audio'], batch['video'], batch['text'], batch['emo_category'], batch['cognition_category']
        loss, combined_feature_emo, combined_feature_cong = self.forward(audio, video, text, emo_category, cognition_category)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(audio), sync_dist=True)
        
        self.training_step_outputs.append({
            'emo_feature': combined_feature_emo.mean(dim=1).detach().cpu(),
            'cog_feature': combined_feature_cong.mean(dim=1).detach().cpu(),
            'emo_label': emo_category.detach().cpu(),
            'cog_label': cognition_category.detach().cpu()
        })
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, video, text, emo_category, cognition_category =batch['audio'],batch['video'],batch['text'],batch['emo_category'],batch['cognition_category']
        loss,combined_feature_emo, combined_feature_cong=self.forward(audio, video, text, emo_category, cognition_category)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(audio),sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.000013, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
        return optimizer
