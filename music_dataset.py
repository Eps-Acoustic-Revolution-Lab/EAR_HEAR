import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.neighbors import KernelDensity
import random
from tqdm import tqdm
import os
import re

# Track 1: Single label (Musicality)
LABEL_NAMES_TRACK_1 = ["Musicality"]

# Track 2: Multiple labels (5 dimensions)
LABEL_NAMES_TRACK_2 = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]

class MusicDataset(Dataset):
    def __init__(self, txt_path, mode, label_names):
        """
        Args:
            txt_path: Path to the pickle file containing dataset
            mode: "train" or "test"
            label_names: List of label names to use (e.g., LABEL_NAMES_TRACK_1 or LABEL_NAMES_TRACK_2)
        """
        self.data = []
        self.mode = mode
        self.label_names = label_names
        
        with open(txt_path, "rb") as f:
            self.data = pickle.load(f)
        print("Loaded dataset:", len(self.data))
        labels_list = []
        for item in self.data:
            tmp = []
            for label_name in self.label_names:
                tmp.append(item[label_name])
            labels_list.append(torch.tensor(tmp, dtype=torch.float32))
        self.labels = torch.stack(labels_list, dim=0).cpu().numpy()
        self.npy_filenames = [
            re.sub(r'(.*?)(_aug\d+)?\.npy$', r'\1', os.path.basename(item['npy_path'])) 
            for item in self.data
        ]

        if mode == "train":
            N = self.labels.shape[0]
            self.weights = np.zeros((N, N), dtype=np.float32)
            for i in tqdm(range(N), total=N, desc="Computing KDE-based probability matrix"):
                label_i = self.labels[i].reshape(1, -1)
                kd = KernelDensity(
                    kernel="gaussian",
                    bandwidth=1.0
                ).fit(label_i)
                each_rate = np.exp(kd.score_samples(self.labels))
                
                # 将相同 npy_filenames 的位置设为 0
                same_filename_mask = np.array([self.npy_filenames[j] == self.npy_filenames[i] for j in range(N)])
                each_rate[same_filename_mask] = 0
                each_rate_sum = np.sum(each_rate)
                if each_rate_sum > 0:
                    each_rate /= each_rate_sum
                else:
                    # 如果所有位置都是0（极端情况，不应该发生），使用均匀分布但排除相同文件名
                    num_valid = N - np.sum(same_filename_mask)
                    if num_valid > 0:
                        each_rate = np.ones(N, dtype=np.float32) / num_valid
                        each_rate[same_filename_mask] = 0
                    else:
                        # 极端情况：所有样本都有相同文件名，使用均匀分布（包括自己）
                        each_rate = np.ones(N, dtype=np.float32) / N

                self.weights[i] = each_rate
 
            self.weights = torch.tensor(self.weights, dtype=torch.float32)
            print("KDE probability matrix computed successfully")
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = np.load(self.data[idx]['npy_path'])
        feature = torch.from_numpy(feature).squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.mode == "train":
            # 训练模式：返回原始样本 + 匹配样本
            partner_idx = self.sample_partner(idx)
            partner_feature = np.load(self.data[partner_idx]['npy_path'])
            partner_feature = torch.from_numpy(partner_feature).squeeze(0)
            partner_label = torch.tensor(self.labels[partner_idx], dtype=torch.float32)
            return feature, label, partner_feature, partner_label
        else:
            # 测试模式：只返回原始样本
            return feature, label
    
    def sample_partner(self, idx):
        """根据预计算的概率矩阵为样本idx选择配对样本j"""
        probs = self.weights[idx]
        return torch.multinomial(probs, num_samples=1).item()
    
def music_collate_fn(batch):
    """
    Custom collate function for batching variable-length audio features.
    
    Handles two modes:
    - Training mode: batches contain original samples and their matched partner samples
    - Test mode: batches contain only original samples
    
    Args:
        batch: List of tuples, where each tuple contains:
            - Training mode: (feature: Tensor[T, dim], label: Tensor[num_labels], 
                             partner_feature: Tensor[T', dim], partner_label: Tensor[num_labels])
            - Test mode: (feature: Tensor[T, dim], label: Tensor[num_labels])
            T and T' are variable sequence lengths, dim is the feature dimension
    
    Returns:
        Training mode:
            feature_batch: Tensor[B, T_max, dim] - Padded original features
            labels: Tensor[B, num_labels] - Original labels
            audio_mask: Tensor[B, T_max] - Binary mask for original features (1 for valid, 0 for padding)
            partner_feature: Tensor[B, T'_max, dim] - Padded partner features
            partner_labels: Tensor[B, num_labels] - Partner labels
            partner_mask: Tensor[B, T'_max] - Binary mask for partner features (1 for valid, 0 for padding)
        
        Test mode:
            feature_batch: Tensor[B, T_max, dim] - Padded features
            labels: Tensor[B, num_labels] - Labels
            audio_mask: Tensor[B, T_max] - Binary mask (1 for valid, 0 for padding)
    """
    if len(batch[0]) == 4:  
        audios = [item[0] for item in batch]  
        labels = torch.stack([item[1] for item in batch], dim=0)  
        partner_audios = [item[2] for item in batch]
        partner_labels = torch.stack([item[3] for item in batch], dim=0)  
        batch_size = len(audios)
        feature_dim = audios[0].shape[-1]

        max_len_ori = max(a.shape[0] for a in audios)  
        feature_batch = torch.zeros(batch_size, max_len_ori, feature_dim, dtype=audios[0].dtype)
        audio_mask = torch.zeros(batch_size, max_len_ori, dtype=torch.long)

        for i, audio in enumerate(audios):
            cur_len = audio.shape[0]
            feature_batch[i, :cur_len, :] = audio 
            audio_mask[i, :cur_len] = 1

        max_len_partner = max(p.shape[0] for p in partner_audios)
        partner_feature = torch.zeros(batch_size, max_len_partner, feature_dim, dtype=audios[0].dtype)
        partner_mask = torch.zeros(batch_size, max_len_partner, dtype=torch.long)

        for i, audio in enumerate(partner_audios):
            cur_len = audio.shape[0]
            partner_feature[i, :cur_len, :] = audio 
            partner_mask[i, :cur_len] = 1 

        return feature_batch, labels, audio_mask, partner_feature, partner_labels, partner_mask
    
    else:
        # 测试集：只处理原始样本
        audios = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch], dim=0)

        max_len = max(a.shape[0] for a in audios)
        batch_size = len(audios)
        feature_dim = audios[0].shape[-1]

        feature_batch = torch.zeros(batch_size, max_len, feature_dim, dtype=audios[0].dtype)
        audio_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, audio in enumerate(audios):
            cur_len = audio.shape[0]
            feature_batch[i, :cur_len, :] = audio
            audio_mask[i, :cur_len] = 1 
        
        return feature_batch, labels, audio_mask

def train_data_loader(pkl_path=None, label_names=None):
    """
    Create training dataset loader.
    
    Args:
        pkl_path: Path to the pickle file (default: 'data_pipeline/dataset_pkl/train_set.pkl')
        label_names: List of label names to use (default: LABEL_NAMES_TRACK_1)
    """
    if pkl_path is None:
        pkl_path = 'data_pipeline/dataset_pkl/train_set.pkl'
    if label_names is None:
        label_names = LABEL_NAMES_TRACK_1
    train_data = MusicDataset(
        txt_path=pkl_path,
        mode="train",
        label_names=label_names
    )
    return train_data

def test_data_loader(pkl_path=None, label_names=None):
    """
    Create test dataset loader.
    
    Args:
        pkl_path: Path to the pickle file (default: 'data_pipeline/dataset_pkl/test_set.pkl')
        label_names: List of label names to use (default: LABEL_NAMES_TRACK_1)
    """
    if pkl_path is None:
        pkl_path = 'data_pipeline/dataset_pkl/test_set.pkl'
    if label_names is None:
        label_names = LABEL_NAMES_TRACK_1
    test_data = MusicDataset(
        txt_path=pkl_path,
        mode="test",
        label_names=label_names
    )
    return test_data


if __name__ == "__main__":
    pkl_path = 'data_pipeline/dataset_pkl/test_set.pkl'
    # Test Track 1
    # ds = train_data_loader(pkl_path, LABEL_NAMES_TRACK_1)
    ds = test_data_loader(pkl_path, LABEL_NAMES_TRACK_1)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True, collate_fn=music_collate_fn)
    for batch in dl:
        print(batch)
        break

