import json
import sys
import time

import wandb
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForMaskedLM
import pandas as pd

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--dataset-path",
    default="/Volumes/New Volume/malware-detection-dataset/opcodes/tokenized_sequences.json",
    help="Path to the dataset",
    type=str,
)
argparser.add_argument("--batch-size", default=128, help="Batch size", type=int)
argparser.add_argument(
    "--learning-rate", default=5e-5, help="Learning rate", type=float
)
argparser.add_argument(
    "--max-length", default=64, help="Maximum sequence length", type=int
)
argparser.add_argument(
    "--tokenizer-path",
    default="/Users/henrywilliams/Documents/uni/masters-thesis/notebooks/classifiers/static/opcode-transformer/opcode_tokenizer.json",
    help="Path to the tokenizer file",
    type=str,
)
argparser.add_argument(
    "--num-attention", default=6, help="Number of attention heads", type=int
)
argparser.add_argument("--hidden-size", default=42, help="Hidden size", type=int)
argparser.add_argument(
    "--num-hidden", default=3, help="Number of hidden layers", type=int
)
argparser.add_argument(
    "--intermediate-size", default=512, help="Size of intermediate layer", type=int
)
argparser.add_argument(
    "--post-train-size",
    default=5,
    help="Number of samples from test set to show in post-train results",
    type=int,
)
argparser.add_argument(
    "--epochs", default=20, help="Number of epochs to train for", type=int
)
args = argparser.parse_args(sys.argv[1:])

tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)


class OpcodeDataset(Dataset):
    def __init__(self, tokenized_sequences, vocab_size: int, mask_prob: float = 0.15):
        self.sequences = tokenized_sequences
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx].copy()

        processed = np.zeros(args.max_length, dtype=np.int64)
        attention_mask = np.ones(args.max_length, dtype=np.int64)
        labels = np.full(args.max_length, -100)

        start = np.random.randint(0, len(tokens))

        selected = tokens[start : start + args.max_length]
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


def eval(model, test_loader):
    model.eval()
    test_loss = 0.0

    test_loop = tqdm(
        test_loader,
        desc=f"Test Epoch {epoch + 1}/{args.epochs}",
        leave=False,
        unit="batch",
    )

    for i, (tokens, labels) in enumerate(test_loop):
        input_ids, attention_mask = tokens.values()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask, labels=labels)

        test_loss += output.loss.item()
        test_loop.set_postfix_str(f"Test Loss: {test_loss / (i + 1):.2f}")
    model.train()
    wandb.log({"test_loss": test_loss / len(test_loader)})
    return test_loss / len(test_loader)


def post_train(model, test):
    model.eval()

    idxs = np.random.choice(len(test.dataset), replace=False, size=args.post_train_size)

    test_loop = tqdm(test, desc="Evaluation", leave=False, unit="batch")

    actual = []
    predicted = []
    was_masked = []
    for input, labels in test_loop:
        input_ids, attention_mask = input.values()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask, labels=labels)

        input_ids = input_ids.cpu()
        labels = labels.cpu()
        output = output.logits.cpu()

        gt = np.zeros_like(labels)
        gt[labels == -100] = input_ids[labels == -100]
        gt[labels != -100] = labels[labels != -100]
        mask = labels != -100

        actual.append(gt)
        predicted.append(output.argmax(dim=-1))
        was_masked.append(mask)

    actual = np.array([gt for batch in actual for gt in batch])
    predicted = np.array([pred for batch in predicted for pred in batch])
    was_masked = np.array([mask for batch in was_masked for mask in batch])

    df = pd.DataFrame(
        {
            "ID": idxs,
            "Actual": [tokenizer.decode(gt) for gt in actual[idxs]],
            "Predicted": [tokenizer.decode(y_hat) for y_hat in predicted[idxs]],
            "Mask": [str(np.nonzero(mask)[0].tolist()) for mask in was_masked[idxs]],
            "Correct %": [
                np.count_nonzero((y == y_hat) & mask) / np.count_nonzero(mask)
                for y, y_hat, mask in zip(
                    actual[idxs], predicted[idxs], was_masked[idxs]
                )
            ],
        }
    )
    correct = np.array(
        [
            [np.count_nonzero((y == y_hat) & mask), np.count_nonzero(mask)]
            for y, y_hat, mask in zip(actual, predicted, was_masked)
            if np.count_nonzero(mask) > 0
        ]
    )

    table = wandb.Table(dataframe=df)
    wandb.log({"eval_sample": table})
    wandb.log({"test_acc": np.mean(correct[:, 0] / correct[:, 1])})


if __name__ == "__main__":
    args.hidden_size = args.num_attention * args.hidden_size
    wandb.init(
        project="opcode-roberta",
        config=args.__dict__,
    )
    print("Loading data... ", end="")
    start_time = time.time()
    data = json.load(open(args.dataset_path, "r"))
    print(f"done, took {time.time() - start_time:.2f}s")

    device = torch.device("mps")

    start_time = time.time()
    print("Creating datasets... ", end="")
    train, test = train_test_split(data)
    del data
    train = OpcodeDataset(train, tokenizer.vocab_size)
    train = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test = OpcodeDataset(test, tokenizer.vocab_size)
    test = DataLoader(test, batch_size=args.batch_size, shuffle=True)
    print(f"done, took {time.time() - start_time:.2f}s")

    start_time = time.time()
    print("Creating model... ", end="")
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_length,
        num_attention_heads=args.num_attention,
        num_hidden_layers=args.num_hidden,
        type_vocab_size=1,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )
    print(config.max_position_embeddings)
    print(config.vocab_size)

    model = RobertaForMaskedLM(config)
    model = model.to(device)
    print(f"done, took {time.time() - start_time:.2f}s")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    losses = []
    test_losses = []

    print("Starting training...")
    for epoch in range(args.epochs):
        test_losses.append(eval(model, test))
        train_loop = tqdm(
            train,
            desc=f"Train Epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=False,
        )
        for input, labels in train_loop:
            input_ids, attention_mask = input.values()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask, labels=labels)
            loss = output.loss
            wandb.log({"loss": loss.item()})
            loss.backward()
            optimizer.step()
            train_loop.set_postfix_str(
                f"Loss: {loss.item():.2f}, Last Test Loss: {test_losses[-1]:.2f}"
            )
            losses.append(loss.item())
    post_train(model, test)
