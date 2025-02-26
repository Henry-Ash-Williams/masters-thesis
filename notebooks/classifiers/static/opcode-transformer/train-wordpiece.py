import os
import json
import argparse
from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import Strip
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from typing import Iterator


def yield_texts_from_json(directory: str) -> Iterator[str]:
    """
    Yields lines of disassembled instructions from JSON files in the given directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".json") and not filename.startswith("._"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    if isinstance(data, list) and all(
                        isinstance(item, str) for item in data
                    ):
                        for line in data:
                            yield line
                    else:
                        print(f"Skipping {filename}, expected an array of strings.")
                except json.JSONDecodeError:
                    print(f"Skipping {filename}, could not parse JSON.")


def train_tokenizer(directory: str, vocab_size: int, output_path: str):
    """
    Trains a WordPiece tokenizer on disassembled instructions from JSON files.
    """
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([Strip()])
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.train_from_iterator(yield_texts_from_json(directory), trainer)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a Hugging Face WordPiece tokenizer from disassembled instructions JSON files."
    )
    parser.add_argument(
        "directory", type=str, help="Path to the directory containing JSON files."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30000,
        help="Vocabulary size for the tokenizer.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer.json",
        help="Output path for the trained tokenizer.",
    )
    args = parser.parse_args()

    train_tokenizer(args.directory, args.vocab_size, args.output)


if __name__ == "__main__":
    main()
