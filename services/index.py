import io
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class Index:
    """Append-only pickle store with a byte-offset index for random access."""

    def __init__(self, base_filename, verbose=False):
        """Initializes paths and load an existing index file if present.

        Args:
            base_filename: Path prefix; creates ``{prefix}_index.pkl`` and ``{prefix}_data.pkl``.
            verbose: Reserved for future logging (currently unused).
        """
        self.base_filename = base_filename
        self.index_filename = f"{base_filename}_index.pkl"
        self.data_filename = f"{base_filename}_data.pkl"
        self.offsets = []
        self.iterations = []
        self.__load_index()

    def __load_index(self):
        """Loads ``offsets`` and ``iterations`` from the index pickle if it exists."""
        if os.path.exists(self.index_filename):
            with open(self.index_filename, 'rb') as f:
                index_data = pickle.load(f)
                self.offsets = index_data.get('offsets', [])
                self.iterations = index_data.get('iterations', [])
            
    def __save_index(self):
        """Persists current ``offsets`` and ``iterations`` to the index pickle."""
        index_data = {
            'offsets': self.offsets,
            'iterations': self.iterations
        }
        with open(self.index_filename, 'wb') as f:
            pickle.dump(index_data, f)
            
    def save_data(self, data, iter=-1, logging=False):
        """Appends one pickled record and update the offset index.

        Args:
            data: Arbitrary Python object to pickle.
            iter: Iteration id stored alongside the offset (-1 if unknown).
            logging: If True, print a short confirmation message.
        """
        mode = "ab" if os.path.exists(self.data_filename) else "wb"
        with open(self.data_filename, mode) as f:
            offset = f.tell()
            pickle.dump(data, f)
            
            self.offsets.append(offset)
            self.iterations.append(iter)
            self.__save_index()
                 
        if logging:
            print("Saved data" + f": iteration {iter}" if iter != -1 else "")
            
    def load_data(self, start_iter=0, end_iter=None, verbose=False):
        """Loads a contiguous slice of records by iteration index.

        Args:
            start_iter: First offset index to read (inclusive).
            end_iter: Last index (exclusive); ``None`` means through the end.
            verbose: If True, show a tqdm progress bar.

        Returns:
            List of unpickled objects.
        """
        data = []
        with open(self.data_filename, "rb") as f:
            for i in tqdm(
                    range(start_iter, end_iter if end_iter else len(self.offsets)),
                    desc="Loading index...",
                    disable=not verbose
                ):
                f.seek(self.offsets[i])
                data.append(pickle.load(f))
            return data 

    def load_data_generator(self, start_iter=0, end_iter=None, batch_size=1):
        """Yields lists of records in batches of ``batch_size``.

        Args:
            start_iter: First offset index to read (inclusive).
            end_iter: Last index (exclusive); ``None`` means through the end.
            batch_size: Number of records per yielded batch.

        Yields:
            Lists of unpickled objects of length up to ``batch_size``.
        """
        data = []
        with open(self.data_filename, "rb") as f:
            for i in range(start_iter, end_iter if end_iter else len(self.offsets)):
                f.seek(self.offsets[i])
                data.append(pickle.load(f))
                if len(data) == batch_size or i == end_iter - 1:
                    yield data
    
    def __len__(self):
        """Returns the number of stored records."""
        return len(self.offsets)
    
    def clear(self):
        """Deletes index and data files and reset in-memory offsets."""
        files_to_remove = [self.index_filename, self.data_filename]
        
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"File is deleted: {file_path}")
                except OSError as e:
                    print(f"Error in deleting file {file_path}: {e}")
            else:
                print(f"Файл не существует: {file_path}")
        
        self.offsets = []
        self.iterations = []
        
        print("Index is cleared successfully.")


class IndexDataset(Dataset):
    """PyTorch-style view over an ``Index`` with split and optional preprocessing."""

    def __init__(
        self,
        index: Index,
        process_elements=lambda x, y: (x, y),
        split="train",
        load_all_data=False,
        train_split=0.8,
        val_split=0.9,
        verbose=False
    ):
        """Configures a train/val/test slice and optional eager loading.

        Args:
            index: Backing ``Index`` instance.
            process_elements: Callable ``(index_data, **kwargs) -> dict`` of tensors used by ``get`` / ``_load_data`` function.
            split: One of ``"train"``, ``"val"``, or ``"test"``.
            load_all_data: If True, preprocess the full split at init time.
            train_split: Fraction of records for training (default 0.8).
            val_split: Upper fraction bound for validation (default 0.9).
            verbose: Passed to ``Index.load_data`` and ``process_elements``.
        """
        self.index = index
        self.split = split
        self.data = None
        self.verbose = verbose
        self.process_elements = process_elements

        if split == "train":
            self.indices = list(range(int(len(index) * train_split)))
        elif split == "val":
            self.indices = list(
                range(
                    int(len(index) * train_split), int(len(index) * val_split)
                )
            )
        else:
            self.indices = list(range(int(len(index) * val_split), len(index)))

        if load_all_data:
            self._load_data()

    def __len__(self):
        """Number of records in this split."""
        return len(self.indices)

    def _map_end_index(self, end: int) -> int:
        """Map exclusive split position ``end`` to ``Index.load_data`` end (exclusive)."""
        if end < len(self.indices):
            return self.indices[end]
        return self.indices[-1] + 1

    def _load_data(self):
        """Eagerly load and preprocess the full split into ``self.data``."""
        self.data = self.process_elements(
            np.array(self.index.load_data(
                self.indices[0],
                self._map_end_index(len(self.indices)),
                verbose=self.verbose
            )),
            verbose=self.verbose
        )

    def get(self, start=0, end=None):
        """Return a batch dict, from self.data or by loading a sub-range of the index.

        Args:
            start: Start position within the split (inclusive).
            end: End position (exclusive); ``None`` loads through the last record.

        Returns:
            Dict of tensors from ``process_elements`` function.
        """
        if end is None:
            end = len(self.indices)

        if self.data:
            return {
                k: v[start:end]
                    for k, v in self.data.items()
            }

        return self.process_elements(
            self.index.load_data(
                self.indices[start],
                self._map_end_index(end),
                verbose=self.verbose
            ),
            verbose=self.verbose
        )
        
    # TODO: def get_generator(self, start=0, end=None):