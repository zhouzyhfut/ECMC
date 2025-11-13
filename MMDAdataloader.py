import os
import numpy as np
import pandas as pd
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Dict, Union

class MMDADataset(Dataset):
    def __init__(
        self,
        root_dir: str = "MMDA",
        split_file: str = "./train.csv",
        audio_dir: str = "split_audio_f",
        video_dir: str = "split_video_f",
        modalities: tuple = ("text", "audio", "video"),
        max_seq_len: Dict[str, int] = {"audio": 1024, "video": 1568},
    ):
        """
        Multimodal dataset loader
        
        Args:
            root_dir: Root directory of the dataset
            split_file: Data split file (CSV format)
            audio_dir: Subdirectory for audio features
            video_dir: Subdirectory for video features
            modalities: Modalities to load
            max_seq_len: Maximum sequence length for each modality
        """
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, audio_dir)
        self.video_dir = os.path.join(root_dir, video_dir)
        self.modalities = modalities
        self.max_seq_len = max_seq_len

        # Read CSV file and parse multi-labels
        self.df = pd.read_csv(split_file)

        def safe_parse_int(x):
            try:
                return int(float(x))
            except:
                return 0
        self.df["emotion_bin"] = self.df["emotion_bin"].fillna(0).apply(safe_parse_int)

        self.df["cognition_bin"] = self.df["cognition_bin"].apply(ast.literal_eval)  
        
        # Build category indices
        self._build_indices()
        self._validate_data()

    def _build_indices(self):
        """Build sample indices for emotion and cognition categories"""
        # Emotion categories (-1,0,1)
        self.emo_indices = defaultdict(list)
        for idx, emo in enumerate(self.df["emotion_bin"]):
            self.emo_indices[emo].append(idx)
        
        # Cognition categories (multi-label)
        self.cog_indices = defaultdict(list)
        for idx, cog in enumerate(self.df["cognition_bin"]):
            for cls_idx, val in enumerate(cog):
                if val == 1:
                    self.cog_indices[cls_idx].append(idx)
        
        # Label statistics
        print("Emotion category distribution:", {k: len(v) for k, v in self.emo_indices.items()})
        print("Cognition category distribution:", {k: len(v) for k, v in self.cog_indices.items()})

    def _validate_data(self):
        """Validate and filter samples with missing feature files"""
        valid_indices = []
        for idx, row in self.df.iterrows():
            base_name = f"{row['id']}_{row['hdTimeStart']}_{row['hdTimeEnd']}"
            valid = True
            
            if "audio" in self.modalities:
                audio_path = os.path.join(self.audio_dir, f"{base_name}.npy")
                if not os.path.exists(audio_path):
                    valid = False
                    
            if "video" in self.modalities:
                video_path = os.path.join(self.video_dir, f"{base_name}.npy")
                if not os.path.exists(video_path):
                    valid = False
            

                    
            if valid:
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Number of remaining samples after filtering: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, np.ndarray, List[int]]]:
        row = self.df.iloc[idx]
        base_name = f"{row['id']}_{row['hdTimeStart']}_{row['hdTimeEnd']}"
        
        # Base data
        data = {
            "text": row["content"],
            "emo_category": int(row.get("emotion_bin", 0)),
            "cognition_category": list(row.get("cognition_bin", [0]*4)),
            "emotion_cap": row.get("emotion", ""),         
            "cognition_cap": row.get("cognition", ""),     
            "id": base_name,
        }
        
        # Load audio features
        if "audio" in self.modalities:
            audio_path = os.path.join(self.audio_dir, f"{base_name}.npy")
            if os.path.exists(audio_path):
                audio_feat = np.load(audio_path)
                data["audio"] = self._pad_feature(audio_feat, self.max_seq_len["audio"])
            else:
                pass
        
        # Load video features
        if "video" in self.modalities:
            video_path = os.path.join(self.video_dir, f"{base_name}.npy")
            if os.path.exists(video_path):
                video_feat = np.load(video_path)
                data["video"] = self._pad_feature(video_feat, self.max_seq_len["video"])
            else:
                pass

        return data
    
    def _pad_feature(self, feature: np.ndarray, max_len: int) -> np.ndarray:
        """Pad or truncate features"""
        if len(feature) > max_len:
            return feature[:max_len]
        elif len(feature) < max_len:
            return np.pad(feature, ((0, max_len - len(feature)), (0, 0)))
        return feature

class MultilabelBalancedSampler:
    """Multi-label balanced sampler"""
    def __init__(self, dataset: MMDADataset, batch_size: int, strategy: str = "emotion"):
        """
        Args:
            strategy: 
                'emotion' - balance by emotion category
                'cognition' - balance by main cognition category
                'multilabel' - balance by multi-label combinations
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        
        if strategy == "emotion":
            self.groups = dataset.emo_indices
        elif strategy == "cognition":
            self.groups = dataset.cog_indices
        else:  # multilabel
            self.groups = defaultdict(list)
            for idx, cog in enumerate(dataset.df["cognition_bin"]):
                self.groups[tuple(cog)].append(idx)
        
        self.min_samples = min(len(v) for v in self.groups.values())
        self.num_batches = self.min_samples // (self.batch_size // len(self.groups))

    def __iter__(self):
        # Create sampling pools for each group
        pools = {
            k: torch.randperm(len(v))[: self.min_samples].tolist()
            for k, v in self.groups.items()
        }
        
        for _ in range(self.num_batches):
            batch = []
            samples_per_group = max(1, self.batch_size // len(self.groups))
            
            for group in self.groups:
                selected = np.random.choice(
                    pools[group],
                    size=samples_per_group,
                    replace=len(pools[group]) < samples_per_group,
                )
                batch.extend(selected)
            
            yield batch[: self.batch_size]  # Ensure batch_size is not exceeded

    def __len__(self):
        return self.num_batches

def collate_multimodal_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate a multimodal batch"""
    texts = [item["text"] for item in batch]
    
    cog_labels = torch.tensor([item["cognition_category"] for item in batch], dtype=torch.float32)
    emo_labels = torch.tensor([item["emo_category"] for item in batch], dtype=torch.long)
    
    if "audio" in batch[0]:
        audios = torch.stack([torch.from_numpy(item["audio"]) for item in batch])
    else:
        audios = None
    
    if "video" in batch[0]:
        videos = torch.stack([torch.from_numpy(item["video"]) for item in batch])
    else:
        videos = None

    # Captions
    emotion_cap = [item["emotion_cap"] for item in batch]
    cognition_cap = [item["cognition_cap"] for item in batch]
    
    return {
        "text": texts,
        "audio": audios,
        "video": videos,
        "emotion_cap": emotion_cap,
        "cognition_cap": cognition_cap,
        "emo_category": emo_labels,
        "cognition_category": cog_labels,
        "ids": [item["id"] for item in batch],
    }

# Usage example
if __name__ == "__main__":
    # Initialize dataset
    train_set = MMDADataset(
        split_file="./train.csv",
        modalities=("text", "audio", "video"),
        max_seq_len={"audio": 1024, "video": 512},
    )
    
    # Create sampler (three strategies available)
    sampler = MultilabelBalancedSampler(
        dataset=train_set,
        batch_size=64,
        strategy="multilabel",  # finest multi-label balancing
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        collate_fn=collate_multimodal_batch,
        num_workers=4,
    )
    
    # Verify batch data
    for batch in train_loader:
        print(f"Batch size: {len(batch['text'])}")
        print(f"Emotion distribution: {batch['emo_category'].unique(return_counts=True)}")
        print(f"Cognition matrix sum: {batch['cognition_category'].sum(dim=0)}")  # counts per category
        break
