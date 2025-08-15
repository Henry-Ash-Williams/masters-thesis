import json
import logging
import os
import signal
import sys
import time
from multiprocessing import Process, set_start_method
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
        raise FileNotFoundError(f"{path}")

    try:
        r2 = r2pipe.open(path, ["-2"])
        r2.cmd("aaa")
        info = r2.cmdj("ij")
        if not info or info["bin"].get("arch") != "x86":
            return []

        section_info = r2.cmdj("iSj")
        executable_sections = [s for s in section_info if "x" in s.get("perm", "")]

        full_disassembly = []
        for section in executable_sections:
            start = section["vaddr"]
            size = section["vsize"]

            disassembly = r2.cmdj(f"pdaj {size} @ {start}")
            valid = [
                instr for instr in disassembly if set(instr.get("bytes", [])) != {"0"}
            ]
            full_disassembly.extend(valid)
            time.sleep(0.01)  # Yield to scheduler

        return full_disassembly

    except Exception:
        return []


def restricted_run(input_path, output_path):
    try:
        import resource

        # Limit memory to 1.5GB
        # soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # resource.setrlimit(resource.RLIMIT_AS, (int(1.5e9), hard))
        # Limit CPU time to 30s
        resource.setrlimit(resource.RLIMIT_CPU, (30, 35))
    except ImportError:
        pass  # macOS only supports partial resource limits

    opcodes = [
        instr["inst"]
        for instr in get_opcodes(input_path)
        if instr.get("inst") != "invalid"
    ]

    if not DEBUG:
        with open(output_path, "w") as fp:
            json.dump(opcodes, fp)


def run_with_timeout(input_path, output_path, timeout=60):
    p = Process(target=restricted_run, args=(input_path, output_path))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"{input_path} exceeded {timeout}s")


def main():
    set_start_method("spawn", force=True)
    labels = {}

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    files = tqdm(os.listdir(BASE))
    for fname in files:
        full_path = os.path.join(BASE, fname)
        opcode_path = os.path.join(DATA_DIR, fname.replace(".exe", ".json"))

        try:
            if os.path.exists(opcode_path):
                raise FileExistsError("Already Processed")

            if os.path.getsize(full_path) > MAX_FILE_SIZE_MB:
                raise ValueError("Too Big")

            labels[opcode_path] = 0
            files.set_description(fname)
            run_with_timeout(full_path, opcode_path, timeout=10)

        except Exception as e:
            files.write(f"Error processing {fname}: {e}")

    if not DEBUG:
        with open(os.path.join(BASE, "labels.json"), "w") as fp:
            json.dump(labels, fp)


if __name__ == "__main__":
    main()
