import json
import time

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForMaskedLM


DATASET_PATH = "/Volumes/New Volume/malware-detection-dataset/opcodes/tokenized_squences.json" 
TOKENIZER_PATH = "./opcode_tokenizer.json"
MAX_LENGTH = 64

tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)

class OpcodeDataset(Dataset):
    def __init__(self, tokenized_sequences, vocab_size: int, mask_prob: float = 0.15):
        self.sequences = tokenized_sequences
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx].copy()

        processed = np.zeros(MAX_LENGTH, dtype=np.int64)
        attention_mask = np.ones(MAX_LENGTH, dtype=np.int64)
        labels = np.full(MAX_LENGTH, -100, dtype=torch.int64)

        start = np.random.randint(0, len(tokens))

        selected = tokens[start : start + MAX_LENGTH]
        processed[: len(selected)] = selected
        attention_mask[processed == 0] = 0

        for i, token in enumerate(processed):
            if token == 0:
                continue
            if np.random.random() < self.mask_prob:
                # Token selected for masking
                labels[i] = token
                q = np.random.random()

                if q < 0.8:
                    # Replace token with mask (80%)
                    processed[i] = tokenizer.encode("[MASK]")[0]
                elif 0.8 < q < 0.9:
                    # Replace token with random token (10%)
                    processed[i] = np.random.randint(2, self.vocab_size)
                else:
                    # Leave as is
                    pass
        return {
            "input_ids": torch.tensor(processed),
            "attention_mask": torch.tensor(attention_mask),
        }, torch.tensor(labels)

if __name__ == "__main__":
    data = json.load(open(DATASET_PATH, "r"))

    device = torch.device("cuda:2")

    train, test = train_test_split(data)
    
    train = OpcodeDataset(train, tokenizer.vocab_size)
    train = DataLoader(train, batch_size=64, shuffle=True)
    test = OpcodeDataset(test, tokenizer.vocab_size)
    test = DataLoader(test, batch_size=64, shuffle=True)

    start_time = time.time()
    print("Creating model... ", end="")
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=MAX_LENGTH,
    )

    model = RobertaForMaskedLM(config)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(20):

        for input, labels in train:
            input_ids, attention_mask = input.values()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask, labels=labels)
            loss = output.loss
            
            loss.backward()
            optimizer.step()
