from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import torch
from lightning.pytorch.strategies import DeepSpeedStrategy
from MMDAdataloader import MMDADataset, collate_multimodal_batch
from MMDAmodel2 import ECMCLLaMA

pl.seed_everything(666)

# Load the model
model = ECMCLLaMA()
# Load stage-1 parameters
model.load_state_dict(torch.load("./checkpoints/your_ckpt")["state_dict"], strict=False)
model.llama_model.gradient_checkpointing_enable()
# Create datasets
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

train_loader = DataLoader(
    train_set, batch_size=4, shuffle=True, collate_fn=collate_multimodal_batch,
    prefetch_factor=2, persistent_workers=True, num_workers=8
)
val_loader = DataLoader(
    val_set, batch_size=4, shuffle=False, collate_fn=collate_multimodal_batch,
    prefetch_factor=2, persistent_workers=True, num_workers=8
)

# Trainer
checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints_stage2',
    filename='stage2-{epoch:02d}-{val_loss_epoch:.4f}',
    save_top_k=30,
    every_n_epochs=1,
    monitor='val_loss_epoch',
    mode='min'
)

trainer = pl.Trainer(
    strategy=DeepSpeedStrategy(config="ds_config.json"),
    accelerator="gpu",
    devices=2,
    precision="16-mixed",
    max_epochs=30,
    accumulate_grad_batches=8,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(name='MMDA_stage2', save_dir='./logger'),
    log_every_n_steps=20
)

trainer.fit(model, train_loader, val_loader)
