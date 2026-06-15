print('✅ connected', flush=True)

import os
print("✅ imported os", flush=True)

import numpy as np
print("✅ imported numpy", flush=True)

import pandas as pd
print("✅ imported pandas", flush=True)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
print("✅ imported torch", flush=True)

from dataloader import MyDataloader
print("✅ imported DataLoader", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ using device: {device}", flush=True)

bootstrap_seed = 0
torch.manual_seed(bootstrap_seed)
np.random.seed(bootstrap_seed)
dataloader = MyDataloader(roi='all', bootstrap_seed=bootstrap_seed)
dataset, loocv_lookup = dataloader.dataset, dataloader.loocv_lookup

X, y = [], []
for participant, participant_data in dataset.items():
    for run_id, run_data in participant_data.items():
        X_prearousal = run_data['prearousal']
        X_sleep = run_data['sleep']
        X_wake = run_data['wake']
        
        n = list(set([X_prearousal.shape[0], X_sleep.shape[0], X_wake.shape[0]]))
        if len(n) != 1:
            print(f"Warning: Mismatched sample sizes for {participant} {run_id}")
            continue
        n = n[0]

        X.append(np.concatenate([X_prearousal, X_sleep, X_wake], axis=0))
        y.append(run_data['true'])

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

# LSTM expects input of shape (batch_size, sequence_length, num_features)
X = X.transpose(0, 2, 1)

# convert to tensors of the correct data types
X = torch.tensor(X, dtype=torch.float32) # for nn.LSTM need float32 inputs
y = torch.tensor(y, dtype=torch.long) # for nn.CrossEntropyLoss need long integer labels

# training
batch_size = 128

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(X.shape)
print(y.shape)