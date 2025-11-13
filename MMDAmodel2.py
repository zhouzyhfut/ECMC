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
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from MMDAmodel import *
import json
from torch.cuda.amp import autocast
class ECMCLLaMA(ECMC):
    def __init__(
        self,
        llama_ckpt="weights"):
        super().__init__()
        #llama
        self.llama_model=LlamaForCausalLM.from_pretrained(llama_ckpt, torch_dtype="auto")
        self.llama_tokenizer=LlamaTokenizer.from_pretrained(llama_ckpt)
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
        for param in self.llama_model.parameters():
            param.requires_grad = False
        self.llama_model.eval() 

        self.emo_llama_project=nn.Linear(768,4096)
        self.cog_llama_project=nn.Linear(768,4096)
     
    def forward(self, audio, video, text, emo_category, cognition_category, emotion_cap, cognition_cap):
        _, combined_feature_emo, combined_feature_cog = super().forward(audio, video, text, emo_category, cognition_category)
        emo_input = self.emo_llama_project(combined_feature_emo)
        cog_input = self.cog_llama_project(combined_feature_cog)
        mm_input = torch.cat([emo_input, cog_input], dim=1)

        batchsize = emo_input.shape[0]

        prompt = '''I provide you with a conversation between a doctor and a user. Please analyze the emotional state and the signs of cognitive impairment. 

        For emotion, What are the facial expressions used by the person in the video? What is the intended meaning behind his words? Which emotion does this reflect?

        For cognition impairment, it includes four domains:1. Orientation 2. Memory 3. Attention 4. Language ability.
        Please provide a brief analysis about cognitive impairment (1–3 sentences) that considers both the user's speech and the video emotion

        Output Format:
        Emotion:
        <your description here>

        Cognition:
        1. ...
        2. ...
        3. ...
        '''

        prompts_id = self.llama_tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).input_ids.to(self.device)
        prompts_id = prompts_id.expand(batchsize, -1)
        prompts_embeds = self.llama_model.model.embed_tokens(prompts_id)

        assert emotion_cap is not None
        text_cap = [f"Emotion:\n{e}\n\nCognition:\n{c}" for e, c in zip(emotion_cap, cognition_cap)]
        text_tokens = self.llama_tokenizer(
            text_cap,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
        targets = text_tokens.input_ids.masked_fill(text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)

        bos = torch.ones([batchsize, 1], dtype=torch.long).to(self.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)

        input_embeds = torch.cat([bos_embeds, mm_input, prompts_embeds, text_embeds], dim=1)

        attn_audio = torch.ones(mm_input.shape[:-1], dtype=torch.long).to(self.device)
        attn_prompt = torch.ones(prompts_embeds.shape[:-1], dtype=torch.long).to(self.device)
        attn_text = text_tokens.attention_mask
        attn_bos = attn_audio[:, :1]
        attention_mask = torch.cat([attn_bos, attn_audio, attn_prompt, attn_text], dim=1)

        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            use_cache=False,
            attention_mask=attention_mask,
            labels=targets,
            return_dict=True
        )
        return outputs.loss

    
    def training_step(self, batch, batch_idx):
        audio, video, text, emo_category, cognition_category, emotion_cap, cognition_cap =batch['audio'],batch['video'],batch['text'],batch['emo_category'],batch['cognition_category'],batch['emotion_cap'],batch['cognition_cap']
        loss=self.forward(audio, video, text, emo_category, cognition_category,emotion_cap, cognition_cap)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(audio),sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, video, text, emo_category, cognition_category, emotion_cap, cognition_cap =batch['audio'],batch['video'],batch['text'],batch['emo_category'],batch['cognition_category'],batch['emotion_cap'],batch['cognition_cap']
        loss=self.forward(audio, video, text, emo_category, cognition_category,emotion_cap, cognition_cap)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(audio),sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, self.parameters()),
        lr=1e-4
    )

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def inference(self, audio, video, text, emo_category, cognition_category):
        self.eval()
        _, combined_feature_emo, combined_feature_cog = super().forward(audio, video, text, emo_category, cognition_category)
        emo_input = self.emo_llama_project(combined_feature_emo)
        cog_input = self.cog_llama_project(combined_feature_cog)
        mm_input = torch.cat([emo_input, cog_input], dim=1)

        batchsize = emo_input.shape[0]

        prompt = '''I provide you with a conversation between a doctor and a user. Please analyze the emotional state and the signs of cognitive impairment. 

        For emotion, What are the facial expressions used by the person in the video? What is the intended meaning behind his words? Which emotion does this reflect?

        For cognition impairment, it includes four domains:1. Orientation 2. Memory 3. Attention 4. Language ability.
        Please provide a brief analysis about cognitive impairment (1–3 sentences) that considers both the user's speech and the video emotion

        Output Format:
        Emotion:
        <your description here>

        Cognition:
        1. ...
        2. ...
        3. ...
        '''
        prompt_ids = self.llama_tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).input_ids.to(self.device)
        prompt_ids = prompt_ids.expand(batchsize, -1) 
        prompt_embeds = self.llama_model.model.embed_tokens(prompt_ids)

        bos = torch.ones([batchsize, 1], dtype=torch.long).to(self.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)

        input_embeds = torch.cat([bos_embeds, mm_input, prompt_embeds], dim=1)

        attn_bos = torch.ones((batchsize, 1), dtype=torch.long).to(self.device)
        attn_mm = torch.ones(mm_input.shape[:-1], dtype=torch.long).to(self.device)
        attn_prompt = torch.ones(prompt_embeds.shape[:-1], dtype=torch.long).to(self.device)
        attention_mask = torch.cat([attn_bos, attn_mm, attn_prompt], dim=1)
     
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                outputs = self.llama_model.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=256,
                        min_new_tokens=None,
                        do_sample=True,
                        top_k=10,
                        top_p=0.95,
                        num_beams=3,
                        repetition_penalty=9.0,
                        pad_token_id=self.llama_tokenizer.pad_token_id,
                        eos_token_id=self.llama_tokenizer.eos_token_id,
                        early_stopping=True,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2
                    )
        decoded = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_outputs=decoded
        return final_outputs

    def test_step(self, batch, batch_idx, save_path="predictions.jsonl"):
        audio = batch["audio"]
        video = batch["video"]
        text = batch["text"]
        emo_cat = batch["emo_category"]
        cog_cat = batch["cognition_category"]
        sample_id = batch["ids"]  

        predictions = self.inference(audio, video, text, emo_cat, cog_cat)  

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'a', encoding='utf-8') as f:
            for sid, pred in zip(sample_id, predictions):
                result = {
                    "id": sid,
                    "label": text,
                    "prediction": pred
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        