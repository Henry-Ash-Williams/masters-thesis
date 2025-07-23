# Disassembles each executable within a directory,
# and saves it to a json file

import json
import logging
from multiprocessing import Process

import os
from os import PathLike

import r2pipe
from tqdm import tqdm

BASE = "/Volumes/malware-dataset/obfuscated-benign/"
MAX_FILE_SIZE_MB = 1024 * 1024
MAX_INSTR_COUNT = 10000
DEBUG = False
DATA_DIR = os.path.join(BASE, "disassembly")

def get_opcodes(path: PathLike):
    if not os.path.exists(path):
        raise Exception(f"Could not find specified file at {path}")

    r2 = r2pipe.open(path, ['-12'])
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

def run(input_path, output_path):
    opcode_info = [
        instruction["inst"]
        for instruction in get_opcodes(input_path)
        if instruction["inst"] != "invalid"
    ]

    if not DEBUG:
        with open(output_path, "w") as fp:
            json.dump(opcode_info, fp)

    
def run_with_timeout(input_path, output_path, timeout=60):
    p = Process(target=run, args=(input_path, output_path)) 
    
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join() 
        raise Exception("Skipped due to timeout")
 
if __name__ == "__main__":
    labels = {}

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    files = tqdm(os.listdir(BASE))
    for path in files:
        full_path = os.path.join(BASE, path)
        opcode_path = os.path.join(DATA_DIR, path.replace('.exe', '.json'))

        try:

            if os.path.exists(opcode_path):
                raise Exception("Already Processed")

            if os.path.getsize(full_path) > MAX_FILE_SIZE_MB:
                raise Exception("Too Big")

            labels[opcode_path] = 0
            files.set_description(path)
            run_with_timeout(full_path, opcode_path, timeout=10)
        except Exception as e:
            files.write(f"Error processing {path}:\n{e}")

    if not DEBUG:
        with open(os.path.join(BASE, "labels.json"), "w") as fp:
            json.dump(labels, fp)