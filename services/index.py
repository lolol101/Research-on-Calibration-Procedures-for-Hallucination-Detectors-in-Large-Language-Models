import io
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class Index:
    def __init__(self, base_filename, verbose=False):
        self.base_filename = base_filename
        self.index_filename = f"{base_filename}_index.pkl"
        self.data_filename = f"{base_filename}_data.pkl"
        self.offsets = []
        self.iterations = []
        self.__load_index()

    def __load_index(self):
        if os.path.exists(self.index_filename):
            with open(self.index_filename, 'rb') as f:
                index_data = pickle.load(f)
                self.offsets = index_data.get('offsets', [])
                self.iterations = index_data.get('iterations', [])
            
    def __save_index(self):
        index_data = {
            'offsets': self.offsets,
            'iterations': self.iterations
        }
        with open(self.index_filename, 'wb') as f:
            pickle.dump(index_data, f)
            
    def save_data(self, data, iter=-1, logging=False):
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
        data = []
        with open(self.data_filename, "rb") as f:
            for i in range(start_iter, end_iter if end_iter else len(self.offsets)):
                f.seek(self.offsets[i])
                data.append(pickle.load(f))
                if len(data) == batch_size or i == end_iter - 1:
                    yield data
    
    def __len__(self):
        return len(self.offsets)
    
    def clear(self):
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
        return len(self.indices) - 1

    def _load_data(self):
        self.data = self.process_elements(
            np.array(self.index.load_data(
                self.indices[0], 
                self.indices[-1], 
                verbose=self.verbose
            )),
            verbose=self.verbose
        )

    def get(self, start=0, end=None):
        if self.data:
            return {
                k: v[start:end] 
                    for k, v in self.data.items()
            }

        return self.process_elements(
            self.index.load_data(
                self.indices[start], 
                self.indices[end] if end else self.indices[-1],
                verbose=self.verbose
            ),
            verbose=self.verbose
        )
        
    # TODO: def get_generator(self, start=0, end=None):