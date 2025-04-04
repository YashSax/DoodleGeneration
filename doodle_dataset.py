from torch.utils.data import Dataset
import numpy as np
import torch
import clip
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
import re


def encode_stroke_data(stroke_data: np.array, pad_length=None):
    # Going from [x, y, lift_pen] to [delta_x, delta_y, pen_on_paper, pen_off_paper, finished]
    new_doodle = np.zeros((stroke_data.shape[0], 5))

    # Handling delta_x, delta_y
    new_row = np.zeros((1, 3))
    temp = np.vstack([new_row, stroke_data])
    new_doodle[:, :2] = temp[1:, :2] - temp[:-1, :2]

    # Handling pen_on_paper and pen_off_paper
    new_doodle[:, 2] = stroke_data[:, 2] == 0
    new_doodle[:, 3] = stroke_data[:, 2] == 1

    # Handling `finished`
    new_doodle[-1, 2] = 0
    new_doodle[-1, 3] = 0
    new_doodle[-1, 4] = 1

    if pad_length is not None and new_doodle.shape[0] < pad_length:
        padding = np.zeros((pad_length - new_doodle.shape[0], 5))
        padding[:, 4] = 1  # Set finished flag to 1 for padding
        new_doodle = np.vstack([new_doodle, padding])

    return new_doodle


def decode_stroke_data(stroke_data: np.array):
    # Going from [delta_x, delta_y, pen_on_paper, pen_off_paper, finished] to [x, y, lift_pen]
    new_doodle = np.zeros((stroke_data.shape[0], 3))
    new_doodle[:, :2] = np.cumsum(stroke_data[:, :2], axis=0)
    new_doodle[:, 2] = np.logical_or(stroke_data[:, 3], stroke_data[:, 4])
    return new_doodle


class DoodleDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, block_size: int, device=str):
        self.data_dir = data_dir
        self.block_size = block_size
        self.data = {}
        self.class_embeddings = {}
        self.device = device
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

        print("Preprocessing Data:")
        for filepath in tqdm(self.data_dir.glob("*.npz")):
            class_name = self._extract_class_name(filepath)
            class_doodles = self._extract_data(
                filepath, split, pad_length=self.block_size + 1
            )  # Pad to block_size + 1 because we need x and y
            self.data[class_name] = class_doodles

        print("Calculating CLIP Embeddings")
        class_names = list(self.data.keys())
        tokenized_classnames = clip.tokenize(class_names)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokenized_classnames.to(device))

        for class_name, class_features in zip(class_names, text_features):
            self.class_embeddings[class_name] = class_features

        self.class_order_counts = []
        self.class_doodle_lengths = {}  # Key = classname, Value = List[int], # blocks
        for class_name, class_doodles in self.data.items():
            self.class_doodle_lengths[class_name] = [
                (idx, self.get_num_blocks(doodle))
                for idx, doodle in enumerate(class_doodles)
            ]
            num_blocks_in_doodles = sum(
                i[1] for i in self.class_doodle_lengths[class_name]
            )
            self.class_order_counts.append((class_name, num_blocks_in_doodles))

        # print("Class Order counts:", self.class_order_counts)
        # print("Class Doodle lengths:", self.class_doodle_lengths["apple"][:10])

    def get_num_blocks(self, doodle):
        return doodle.shape[0] - self.block_size

    def __len__(self):
        return sum(i[1] for i in self.class_order_counts)

    def get_category_from_index(self, idx: int, category_list: List[Tuple]):
        for category, count in category_list:
            idx -= count
            if idx < 0:
                return category, idx + count
        return category_list[-1][0], idx

    def __getitem__(self, idx):
        assert idx < len(self), f"Index {idx} out of bounds with length {len(self)}!"

        # Get the class name
        class_name, idx_remainder = self.get_category_from_index(
            idx, self.class_order_counts
        )
        # print("Class name:", class_name)

        # Get the doodle index
        # print("Postprocessed idx:", idx_remainder)
        doodle_index, block_idx = self.get_category_from_index(
            idx_remainder, self.class_doodle_lengths[class_name]
        )

        # print("Doodle index:", doodle_index)
        # print("Block idx:", block_idx)

        # Extract the block
        full_doodle = self.data[class_name][doodle_index]
        # print("Full doodle length:", full_doodle.shape)

        x = full_doodle[block_idx : block_idx + self.block_size]
        y = full_doodle[block_idx + 1 : block_idx + self.block_size + 1]
        classname_embedding = self.class_embeddings[class_name]

        return torch.from_numpy(x), torch.from_numpy(y), classname_embedding

    def _extract_class_name(self, file: Path):
        # Example filename: `sketchrnn_apple.full.npz`
        pattern = r"sketchrnn_([^.]+)\.full\.npz"
        match = re.match(pattern, file.name)
        assert match, f"Regex for detecting classname failed on {file}"
        return match.group(1)

    def _extract_data(self, file: Path, split: str, pad_length: int):
        assert split in [
            "train",
            "test",
            "valid",
        ], f"Split {split} is not one of: train, test, valid!"

        raw_data = np.load(file, encoding="latin1", allow_pickle=True)[split]
        encoded_doodles = []
        for doodle in raw_data:
            encoded_doodles.append(encode_stroke_data(doodle, pad_length=pad_length))

        return encoded_doodles
