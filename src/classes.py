import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

class amazon_dataset(Dataset):
    def __init__(self, encoded_data, labels):
        self.input_ids = encoded_data["input_ids"]
        self.attention_mask = encoded_data["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

class amazon_dataset_run(Dataset):
    def __init__(self, encoded_data, labels):
        self.input_ids = encoded_data["input_ids"]
        self.attention_mask = encoded_data["attention_mask"]

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }