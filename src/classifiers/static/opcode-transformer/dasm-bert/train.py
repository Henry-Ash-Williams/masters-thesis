import os 
import torch 
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast, 
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments,
    LineByLineTextDataset,
)
from sklearn.model_selection import train_test_split 

BASE_PATH = "/Volumes/New Volume/malware-detection-dataset/opcodes/data"

if __name__ == "__main__":
    files = [os.path.join(BASE_PATH, filename) for filename in os.listdir(BASE_PATH) if not filename.startswith("._")]
    train_files, test_files = train_test_split(files)

    tokenizer = RobertaTokenizerFast.from_pretrained("/Users/henrywilliams/Documents/uni/masters-thesis/src/classifiers/static/opcode-transformer/dasm-bert")
    input()