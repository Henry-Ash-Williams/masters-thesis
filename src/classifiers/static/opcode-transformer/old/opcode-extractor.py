# Disassembles each executable within a directory,
# and saves it to a json file

import datetime
import json
import logging
import os
import time
from os import PathLike

import r2pipe

import utils.data as data

BASE = "/Volumes/New Volume/malware-detection-dataset/opcodes/"
MAX_FILE_SIZE_MB = 1024 * 1024
MAX_INSTR_COUNT = 10000
DEBUG = False
DATA_DIR = os.path.join(BASE, "data-2/")

logging.basicConfig(
    filename=os.path.join(
        BASE,
        f"{datetime.datetime.now().isoformat()}_{'debug_' if DEBUG else ''}opcode_extraction.log",
    ),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_opcodes(path: PathLike):
    if not os.path.exists(path):
        raise Exception(f"Could not find specified file at {path}")

    r2 = r2pipe.open(path)
    r2.cmd("aaa")

    info = r2.cmdj("ij")

    if info["bin"]["arch"] != "x86":
        logging.info(f"Skipping {path.split('/')[-1]}, not x86")
        return []

    section_info = r2.cmdj("iSj")
    executable_sections = [
        section for section in section_info if "x" in section.get("perm", "")
    ]

    full_disassembly = []

    for section in executable_sections:
        start = section["vaddr"]
        size = section["vsize"]

        disassembly = r2.cmdj(f"pdaj {size} @ {start}")

        valid = [instr for instr in disassembly if set(instr["bytes"]) != {"0"}]
        full_disassembly.extend(valid)

    return full_disassembly


if __name__ == "__main__":
    dataset = data.MalwareDataset()
    labels = {}
    logging.info("Beginning opcode extraction")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for path, label in dataset:
        ppath = path.split("/")[-1]
        opcode_path = os.path.join(DATA_DIR, f"{path.split('/')[-1]}.json")
        if os.path.exists(opcode_path):
            labels[opcode_path] = int(label)
            logging.warning(f"Skipping {ppath}: already processed")
            continue

        if os.path.getsize(path) > MAX_FILE_SIZE_MB:
            logging.warning(
                f"Skipping {ppath}: File size exceeds limit ({MAX_FILE_SIZE_MB / (1024 * 1024)} MB)."
            )
            continue

        try:
            logging.info(f"Started processing {ppath}")
            start_time = time.time()
            opcode_info = [
                instruction["inst"]
                for instruction in get_opcodes(path)
                if instruction["inst"] != "invalid"
            ]
            elapsed = time.time() - start_time

            if not DEBUG:
                with open(opcode_path, "w") as fp:
                    json.dump(opcode_info, fp)
            else:
                logging.warning(f"Saving {ppath} was skipped as debug mode is enabled")

            labels[opcode_path] = int(label)
            logging.info(f"Finished processed {ppath} in {elapsed:.2f}s.")
        except Exception as e:
            logging.error(f"Error processing {ppath}: {e}")
            continue

    logging.info("Saving label information to disk")

    if not DEBUG:
        with open(os.path.join(BASE, "labels.json"), "w") as fp:
            json.dump(labels, fp)

    logging.info("Opcode extraction complete.")
