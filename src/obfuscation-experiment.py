import json
import torch
import numpy as np 
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torchvision.models.efficientnet import EfficientNet
import torchvision.transforms.functional as TF

import os
import random
import tempfile
import subprocess
from os import PathLike
from argparse import ArgumentParser
from typing import Dict, List, Tuple

device = torch.device('mps')

IMAGE_SIZE = 320

RUNS = {
    'base': "/Users/henrywilliams/Documents/programming/python/ai/cnn-malware-detector/models/full-ds.pt",
    'filtered': "/Users/henrywilliams/Documents/programming/python/ai/cnn-malware-detector/models/tmp.pt",
}

def get_args():
    parser = ArgumentParser(
        prog="Obfuscation Experiment",
        description="TODO: Write a description",
    )
    
    parser.add_argument(
        "target", 
        choices=['malicious', 'benign', 'all'],
        default='all',
        type=str,
        help="The class of executables to be obfuscated"
    )
    
    parser.add_argument(
        '-m', '--model', 
        choices=list(RUNS.keys()), 
        default='base', 
        type=str, 
        help='The model used by the experiment'
    )
    
    parser.add_argument(
        '-o', '--output', 
        type=str,
        help='Results output location', 
    )

    parser.add_argument(
        '-s', '--steps', 
        type=int, 
        default=10, 
        help='Number of experiment steps',
    )
    
    parser.add_argument(
        '-f', '--files', 
        type=str, 
        help="Path to a JSON file containing the locations and labels of files used by the experiment", 
        default="/Users/henrywilliams/Documents/programming/python/ai/malbert-test/notebooks/obfuscation-experiment-files.json",
    )
    
    args = parser.parse_args()    
    return args
    
    
def make_image(file_path):
    binary = open(file_path, "rb").read()

    if not binary: 
        raise Exception("Zero data read")

    num_bytes = len(binary)
    shape = np.ceil(np.sqrt(num_bytes)).astype(np.int32)

    image = np.zeros((shape**2,))
    image[:num_bytes] = list(binary)
    image = torch.tensor(image).reshape((1, 1, shape, shape))
    return TF.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])


def pipeline(model: EfficientNet, path: os.PathLike) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    with torch.no_grad():
        return model(make_image(path).expand(-1, 3, -1, -1).to(torch.float32).to(device))

def get_model(path: PathLike) -> EfficientNet:
    return torch.load(path, weights_only=False).to(device)

def obfuscate(
    file: os.PathLike,
    output: os.PathLike
) -> os.PathLike:
    obfuscated_path = os.path.join(output, file.split('/')[-1].replace('.exe', '.obfs.exe'))
    subprocess.run(['upx', '-o', obfuscated_path, file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return obfuscated_path
    
def experiment_step(
    args,
    model: EfficientNet,
    file: os.PathLike,
    label: int,
    tdir: str,
    p: float = 1.0
) -> Tuple[torch.Tensor, bool]:
    ''' 
    p is the likelihood of the file being obfuscated
    '''
    obfuscated = False
    q = random.random()

    if args.target == 'all' or \
        (args.target == "malicious" and label == 1) or \
        (args.target == "benign" and label == 0): 

        if q < p: 
            path = obfuscate(file, tdir)
            obfuscated = True
        else: 
            path = file
    else: 
        path = file 

    return pipeline(model=model, path=path), obfuscated

def run_experiment(args, files: List[os.PathLike], labels: List[int], model: EfficientNet, step: int, p: float = 0.0) -> Dict[str, Dict[str, float] | float]:
    predicted = []
    actual = []

    os.makedirs(os.path.join(args.output, f"step-{step}/logits"), exist_ok=True)

    with tempfile.TemporaryDirectory() as tdir: 
        for file, label in tqdm(zip(files, labels), total=len(files), leave=False, desc=f'p = {p:.2}', position=1): 
            try: 
                logits, was_obfuscated = experiment_step(
                    args=args,
                    model=model, 
                    file=file, 
                    label=label, 
                    tdir=tdir, 
                    p=p
                )
            except Exception as e: 
                # print(e)
                continue 
            
            output_logits_path = os.path.join(
                args.output,
                f"step-{i}/logits", f"{file.split('/')[-1].replace('.exe', '.obfs.pt' if was_obfuscated else '.pt')}"
            )

            torch.save(logits, output_logits_path)
            class_likelihood = F.softmax(logits, dim=-1)

            predicted.append(class_likelihood.argmax().item())
            actual.append(label)

        torch.save(torch.tensor(predicted), os.path.join(args.output, f"step-{step}/predicted.pt"))
        torch.save(torch.tensor(actual), os.path.join(args.output, f"step-{step}/actual.pt"))

    return classification_report(actual, predicted, output_dict=True)

if __name__ == "__main__":
    print("Hello, World!")
    args = get_args()
    model = get_model(RUNS[args.model])

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, 'config.json'), 'w') as f: 
        json.dump(args.__dict__, f)
    
    with open(args.files, 'r') as file: 
        data = json.load(file)

    files = list(data.keys())
    labels = list(data.values())
    
    obfuscation_likelihood = np.linspace(0.0, 1.0, args.steps)
    
    results = []

    for i, p in tqdm(enumerate(obfuscation_likelihood), position=0, desc='Running...', total=len(obfuscation_likelihood)):
        results.append(run_experiment(args, files, labels, model, i, p))

    with open(os.path.join(args.output, 'results.json'), 'w') as f: 
        json.dump(results, f)