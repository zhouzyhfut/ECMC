from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
import torch
from lightning.pytorch import Trainer, LightningDataModule, LightningModule, Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import torch.optim as optim
import os
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())
from MMDAdataloader import *
from MMDAmodel import ECMC

pl.seed_everything(666)
model=ECMC()
    
batch_size=64
#create the train and val set
train_set = MMDADataset(
        split_file="train.csv",
        modalities=("text", "audio", "video"),
        max_seq_len={"audio": 1024, "video": 512},
    )
val_set = MMDADataset(
        split_file="val.csv",
        modalities=("text", "audio", "video"),
        max_seq_len={"audio": 1024, "video": 512},
    )

train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_batch,prefetch_factor=2,persistent_workers=True,num_workers=8)
val_loader=DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch,prefetch_factor=2,persistent_workers=True,num_workers=8)

#put your own checkpoint dir here
checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='mymodel-{epoch:02d}-{train_loss:.5f}',
        save_top_k=20,
        every_n_epochs=5,
        monitor='val_loss',
        mode='min'
    )

trainer = pl.Trainer(
    profiler="simple",
    logger=TensorBoardLogger(name='MMDA_model',save_dir='./logger'),
    accelerator='gpu',
    max_epochs=10000,
    devices=1,
    log_every_n_steps=50,
    precision="16-mixed",
    callbacks=[checkpoint_callback],
    #accumulate_grad_batches=4,
    #strategy="ddp_find_unused_parameters_true"
    )

trainer.fit(model, train_loader, val_loader)