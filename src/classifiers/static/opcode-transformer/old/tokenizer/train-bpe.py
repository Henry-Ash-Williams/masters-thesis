import os
from typing import List 

from tokenizers import ByteLevelBPETokenizer 

BASE = "/Volumes/New Volume/malware-detection-dataset/opcodes/processed-data"

def read_file(path) -> List[str]:
    with open(path, "r") as fp: 
        return fp.read().split(' ')

if __name__ == "__main__":
    files = [
        os.path.join(BASE, filename) 
        for filename in os.listdir(BASE) 
        if not (filename.startswith("._") or filename == "labels.json")
    ]

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files, 
        vocab_size=2000, 
        min_frequency=1, 
        special_tokens=[
            '<s>',
            '<pad>',
            '</s>',
            '<unk>',
            '<mask>',
        ]
    )
    tokenizer.save('mnembert-bpe')