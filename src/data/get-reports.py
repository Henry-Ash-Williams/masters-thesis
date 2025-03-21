import json
import os

import requests
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

load_dotenv("/Users/henrywilliams/Documents/uni/masters-thesis/.env")
BASE_PATH = "/Volumes/New Volume/malware-detection-dataset/malware"
API_KEY = os.environ["VS_API_KEY"]


def get_hash(fp: str) -> str:
    return fp.split("_")[-1]


@sleep_and_retry
@limits(calls=4, period=60)
def make_request(hash):
    url = f"https://virusshare.com/apiv2/file?apikey={API_KEY}&hash={hash}"
    res = requests.get(url)

    if res.status_code != 200:
        raise Exception(f"API Response Code: {res.status_code}")

    return json.loads(res.content)


def process_item(item):
    hash = get_hash(item)
    report = make_request(hash)
    save_report(report, hash)


def save_report(report, hash):
    with open(
        f"{BASE_PATH}/reports/report-{hash}.json",
        "w",
    ) as file:
        json.dump(report, file)


if __name__ == "__main__":
    files = [
        os.path.join(BASE_PATH, file)
        for file in os.listdir(BASE_PATH)
        if file.startswith("VirusShare_")
    ]
    pbar = tqdm(files, desc="Gathering Reports", total=len(files), unit="reqs")

    for file in pbar:
        process_item(file)
