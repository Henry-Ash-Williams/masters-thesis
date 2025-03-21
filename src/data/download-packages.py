import json
from os import PathLike
from time import sleep
from typing import List

from tqdm import tqdm

DATA_LOC = "/Volumes/New Volume/malware-detection-dataset/benign"


def load_package_names(path: PathLike) -> List[str]:
    with open(path, "r") as file:
        return json.load(file)


def download(name: str) -> None:
    sleep(1)


if __name__ == "__main__":
    package_names = load_package_names(
        "/Users/henrywilliams/Documents/uni/masters-thesis/notebooks/data/choco/install-commands.json"
    )
    loop = tqdm(package_names, unit="package")

    for package in loop:
        loop.set_description(package)
        download(package)
