import os
import json
import argparse
import concurrent.futures
import requests
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(
    filename="download.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_file(url: str, output_dir: Path, progress: tqdm) -> None:
    """Downloads a file from the given URL and saves it in the output directory."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        filename = os.path.basename(urlparse(url).path) or f"file_{hash(url)}.dat"
        filepath = output_dir / filename
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        
        progress.update(1)
        logging.info(f"Downloaded: {url} -> {filepath}")
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        progress.update(1)

def main():
    parser = argparse.ArgumentParser(description="Download files from a JSON list of URLs in parallel.")
    parser.add_argument("--input" , help="Path to JSON file containing a list of URLs", default="/Volumes/New Volume/malware-detection-dataset/winget-urls.json")
    parser.add_argument("--output",  help="Directory to save downloaded files", default="/Volumes/New Volume/malware-detection-dataset/more-benign-installers/")
    parser.add_argument("--jobs", type=int, default=4, help="Number of concurrent downloads")
    args = parser.parse_args()
    
    # Load URLs from JSON file
    with open(args.input, "r") as f:
        urls = json.load(f)
        
    if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
        raise ValueError("JSON file must contain a list of URLs.")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup progress bar
    with tqdm(total=len(urls), desc="Downloading", unit="file") as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(download_file, url, output_dir, progress) for url in urls]
            concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
