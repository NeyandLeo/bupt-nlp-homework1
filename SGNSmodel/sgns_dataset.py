from torch.utils.data import Dataset, DataLoader
import pickle
import torch

class TrainingDataset(Dataset):
    def __init__(self, file_path):
        # Load the training data from the file
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_index, context_indices = self.data[idx]
        return torch.tensor(word_index), torch.tensor(context_indices)#, torch.tensor(negative_indices)