import os
from pathlib import Path

import torch
from tqdm import tqdm
import csv
from torch.nn.utils.rnn import pad_sequence


def build_vocab(file_path):
    with open(file_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)  # skip the headers
        for line in tsvreader:
            line = line[0]
            line = line.replace(']', 'X').replace('(', '').replace(')', '')
            seq = line.split(' ')
            seq = list(filter(None, seq))
            vocab = list(set(seq))
            vocab.sort()
            return vocab


def preprocess_data(part: str, vocab: list, ch2idx: dict, max_seq: int, input_file: Path, output_data_file: Path,
                    output_target_file: Path) -> None:
    """Preprocess the data and save the tensors."""
    sources = []
    targets = []
    sources.append(torch.tensor([0 for _ in range(max_seq)], dtype=torch.long))  # Dummy tensor
    with open(input_file) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)  # Skip the headers
        total = 96000 if part == 'train' else 2000
        for line in tqdm(tsvreader, total=total):
            targ = line[1]
            line = line[0]
            line = line.replace(']', 'X').replace('(', '').replace(')', '')
            seq = line.split(' ')
            seq = list(filter(None, seq))
            mapped_seq = [ch2idx[token] for token in seq]
            sources.append(torch.tensor(mapped_seq, dtype=torch.long))  # Save as LongTensor
            targets.append(int(targ))

        final_tensor = pad_sequence(sources, padding_value=ch2idx['<PAD>']).T[1:]  # Remove dummy tensor
        final_targets = torch.tensor(targets, dtype=torch.long)  # Save targets as LongTensor
    torch.save(final_tensor, output_data_file)
    torch.save(final_targets, output_target_file)


if __name__ == "__main__":
    # change dir to root
    root_dir = Path(__file__).resolve().parent.parent
    raw_data_dir = root_dir / 'data/raw/listops-1000/'
    preprocessed_data_dir = root_dir / 'data/preprocessed/listops-1000/'
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = raw_data_dir / 'basic_test.tsv'

    vocab = build_vocab(vocab_file)
    vocab_size = len(vocab) + 1  # Including <PAD>
    ch2idx = {x: i for i, x in enumerate(vocab)}
    ch2idx['<PAD>'] = vocab_size - 1
    max_seq = 1999

    for part in ['test', 'val', 'train']:
        print(f'Starting {part}')
        input_file = raw_data_dir / f'basic_{part}.tsv'
        output_data_file = preprocessed_data_dir / f'{part}_clean.pt'
        output_target_file = preprocessed_data_dir / f'target_{part}_clean.pt'
        preprocess_data(part, vocab, ch2idx, max_seq, input_file, output_data_file, output_target_file)

    # save the vocab size
    torch.save(torch.tensor(vocab_size), preprocessed_data_dir / 'vocab_size.pt')
