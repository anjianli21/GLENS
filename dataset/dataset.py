from torch.utils.data import Dataset
import torch

class BaseDataset(Dataset):
    
    def __init__(self):
        pass
            
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError
    
    @staticmethod
    def collate_fn(batch):
        # Get all keys from the first batch item
        keys = batch[0].keys()
        
        merged_batch = {}
        
        # Process each key in the batch
        for key in keys:
            # Check if the value is a tensor that can be stacked
            if torch.is_tensor(batch[0][key]):
                merged_batch[key] = torch.stack([item[key] for item in batch], dim=0)
            else:
                merged_batch[key] = [item[key] for item in batch]
        
        return merged_batch