import json 
from tqdm import tqdm
import os
from tokenizers import ByteLevelBPETokenizer

BASE_PATH = "/Volumes/New Volume/malware-detection-dataset/opcodes/data"

def read_file(path):
    if not os.path.exists(path): 
        raise Exception("File not found")

    return json.load(open(path, 'r'))

def file_iter(files):
    for file in files: 
        for line in read_file(file):
            yield line

if __name__ == "__main__":
    files = [os.path.join(BASE_PATH, filename) for filename in os.listdir(BASE_PATH) if not filename.startswith("._")]
    tokenizer = ByteLevelBPETokenizer() 
    tokenizer.train_from_iterator(file_iter(files), vocab_size=15000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<mask>", "<unk>"])    
    tokenizer.save_model(".")
