import os
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

model = ECMCLLaMA()
model = model.eval()
test_set = MMDADataset(
    split_file="test.csv",
    modalities=("text", "audio", "video"),
    max_seq_len={"audio": 1024, "video": 512},
)

test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_multimodal_batch,
    prefetch_factor=2,
    persistent_workers=True,
    num_workers=8
)

trainer = pl.Trainer(
    strategy=DeepSpeedStrategy(config="ds_config.json",load_full_weights=True),
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
)

trainer.test(
    model=model,
    dataloaders=test_loader,
    ckpt_path="your.ckpt"
)

