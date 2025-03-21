import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt 
import wandb

import os 
import json
import argparse

from typing import List
from collections import Counter

DATA_PATH = "/Volumes/New Volume/malware-detection-dataset/opcodes/processed-data"
VOCAB_PATH = "/Users/henrywilliams/Documents/uni/masters-thesis/src/classifiers/static/opcode-transformer/vocab_table.json"

parser = argparse.ArgumentParser() 

parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--max-len', type=int, default=512)
parser.add_argument('--embedding-dim', type=int, default=128)
parser.add_argument('--num-heads', type=int, default=4)
parser.add_argument('--num-layers', type=int, default=4)
parser.add_argument('--hidden-dim', type=int, default=256)
parser.add_argument('--learning-rate', type=float, default=2e-4)
parser.add_argument('--dropout-rate', type=float, default=0.1)
args = parser.parse_args()

VOCAB_SIZE = 1293
EPOCHS = 20

vocab_lookup = json.load(open(VOCAB_PATH, 'r'))

class OpcodeDataset(Dataset): 
    def __init__(self, paths, labels):
        assert len(paths) == len(labels), "Mismatch between number of files and labels"
        self.paths = paths 
        self.labels = labels

    def __len__(self):
        return len(self.paths)        


    def __getitem__(self, idx):
        assert 0 <= idx <= len(self), "Index out of range"
        label = self.labels[idx]

        with open(self.paths[idx], 'r') as file: 
            content = file.readlines() 
            
        if len(content) >= args.max_len: 
            start = torch.randint(len(content) - args.max_len, (1,))
            sequence = [vocab_lookup[instr.rstrip()] for instr in content[start:start + args.max_len]]
        else: 
            sequence = [vocab_lookup[instr.rstrip()] for instr in content]
        X = torch.full((args.max_len,), -1)
        X[:len(sequence)] = torch.tensor(sequence)
        return X, label

class Classifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int, 
        num_heads: int, 
        num_layers: int, 
        hidden_dim: int, 
        max_seq_len: int, 
        dropout: float = 0.1,
    ):
        super(Classifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim, 
                dropout=dropout, 
                batch_first=True
            ), 
            num_layers
        )

        self.fc = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.embedding(x) + self.position_encoding[:, :x.shape[1], :]
        x = self.encoder(x) 
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)

def get_data(path: os.PathLike, full_path: bool = True) -> List[str]:
    all_files = os.listdir(path)
    
    if full_path:
        return [os.path.join(path, file) for file in all_files if file.endswith('.txt')]
    else: 
        return all_files

def get_labels(filenames):
    return [1 if "VirusShare" in filename else 0 for filename in filenames]


def train(model, data_loader, loss_fn, optim):
    train_loop = tqdm(data_loader, desc=f"Train {epoch + 1}/{EPOCHS}", unit="batch", leave=False)

    for sequence, label in train_loop: 
        sequence = sequence.to(device)
        label = label.float().to(device)

        optim.zero_grad() 
        output = model(sequence).squeeze(1)
        
        loss = loss_fn(output, label)
        
        loss.backward() 
        optim.step() 

        train_loop.set_postfix_str(f"Loss: {loss.item():.2f}")
        wandb.log({'loss': loss.item()})

def test(model, data_loader, loss_fn):
    total_loss = 0.0
    total_correct = 0 
    total = 0 
    model.eval() 
    
    for sequence, label in tqdm(data_loader, desc=f"Test {epoch + 1}/{EPOCHS}", leave=False, unit="batch"):
        sequence = sequence.to(device)
        label = label.float().to(device)

        with torch.no_grad():
            output = model(sequence).squeeze(1)

        loss = loss_fn(output, label)

        total_loss += loss.item() 
        predictions = (torch.sigmoid(output) > 0.5).long() 
        total_correct += (predictions == label.long()).sum().item() 
        total += label.size(0)
        
    acc = total_correct / total 
    wandb.log({'test_loss': total_loss / len(data_loader)})
    wandb.log({'acc': acc})
    print(f"Epoch {epoch + 1}/{EPOCHS}: Loss {total_loss / len(data_loader):.2f}, Acc {acc:.2f}")
    model.train()

if __name__ == "__main__":
    wandb.login()
    run = wandb.init(
        project='simple-opcode-transformer',
        config={
            "batch-size": args.batch_size,
            "max-len": args.max_len,
            "embedding-dim": args.embedding_dim * args.num_heads,
            "num-heads": args.num_heads,
            "num-layers": args.num_layers,
            "hidden-dim": args.hidden_dim, 
            "learning-rate": args.learning_rate, 
            "dropout-rate": args.dropout_rate
        }
    )

    paths = get_data(DATA_PATH)
    labels = get_labels(paths)

    train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels)
    train_data = OpcodeDataset(train_paths, train_labels)
    test_data = OpcodeDataset(test_paths, test_labels)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=True)
    
    device = torch.device('mps')

    model = Classifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=args.embedding_dim * args.num_heads,
        num_heads=args.num_heads,
        num_layers=args.num_layers, 
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_len,
        dropout=args.dropout_rate
    ).to(device)
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(EPOCHS):
        train(model, train_loader, loss_fn, optim)
        test(model, test_loader, loss_fn)