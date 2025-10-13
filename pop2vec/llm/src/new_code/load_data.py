import json
import logging
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

import h5py
from torch.utils.data import IterableDataset
import os
import numpy as np

import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

class CustomIterableDataset(IterableDataset):
    def __init__(self, file_path, validation, num_val_items=None, val_split=0.1, mlm_encoded=True, inference=False):
        self.file_path = file_path
        self.validation = validation
        self.num_val_items = num_val_items
        self.val_split = val_split
        self.inference = inference
        self.set_mlm_encoded(mlm_encoded)

    def set_mlm_encoded(self, mlm_encoded, return_index=None):
        self.mlm_encoded = mlm_encoded
        if return_index is None:
          self.return_index = not self.mlm_encoded
        else:
          self.return_index = return_index


    def __len__(self):
        with h5py.File(self.file_path, 'r') as hdf5:
            return hdf5['input_ids'].shape[0] 


    def __iter__(self):
        with h5py.File(self.file_path, 'r') as hdf5:
            num_val_items = self.num_val_items
            if num_val_items is None:
              num_val_items = int(hdf5['input_ids'].shape[0] * self.val_split)
            
            n_items = hdf5['input_ids'].shape[0]
            num_train_items = n_items - num_val_items
            rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))    
            if self.validation:
                per_worker = num_val_items // world_size
                start_index = rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else num_val_items
            elif self.inference:
                per_worker = n_items // world_size
                start_index = rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else n_items
            else:
                per_worker = num_train_items // world_size
                start_index = num_val_items + rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else num_train_items + num_val_items


            for index in range(start_index, end_index):
                ret_dict = {
                    "input_ids": hdf5['input_ids'][index],
                    "padding_mask": hdf5['padding_mask'][index],
                }

                if self.mlm_encoded:
                    neg_one_index = np.where(hdf5['target_tokens'][index] == -1)[0]
                    target_tokens = hdf5['target_tokens'][index][:neg_one_index[0] if neg_one_index.size > 0 else None]
                    target_pos = hdf5['target_pos'][index][:neg_one_index[0] if neg_one_index.size > 0 else None]
                    
                    ret_dict.update({
                        "original_sequence": hdf5['original_sequence'][index],
                        "target_tokens": target_tokens,
                        "target_pos": target_pos,
                        "target_cls": hdf5['target_cls'][index],
                    })

                if self.return_index:
                    ret_dict["sequence_id"] = hdf5['sequence_id'][index]

                yield ret_dict

class CustomDataset(Dataset):
    def __init__(self, file_path, validation, num_val_items=None, val_split=0.1, mlm_encoded=True, inference=False):
        self.file_path = file_path
        self.validation = validation
        self.num_val_items = num_val_items
        self.val_split = val_split
        self.inference = inference
        self.set_mlm_encoded(mlm_encoded)
        self.load_data()

    def set_mlm_encoded(self, mlm_encoded, return_index=None):
        self.mlm_encoded = mlm_encoded
        if return_index is None:
            self.return_index = not self.mlm_encoded
        else:
            self.return_index = return_index

    def load_data(self):
        with h5py.File(self.file_path, 'r') as hdf5:
            n_items = hdf5['input_ids'].shape[0]
            num_val_items = self.num_val_items
            if num_val_items is None:
                num_val_items = int(n_items * self.val_split)
            num_train_items = n_items - num_val_items

            # Get rank and world_size from environment variables
            rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            if self.validation:
                per_worker = num_val_items // world_size
                start_index = rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else num_val_items
                indices = np.arange(start_index, end_index)
            elif self.inference:
                per_worker = n_items // world_size
                start_index = rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else n_items
                indices = np.arange(start_index, end_index)
            else:
                per_worker = num_train_items // world_size
                start_index = num_val_items + rank * per_worker
                end_index = start_index + per_worker if rank < world_size - 1 else n_items
                indices = np.arange(start_index, end_index)

            # Load datasets into memory
            self.input_ids = hdf5['input_ids'][indices]
            self.padding_mask = hdf5['padding_mask'][indices]

            if self.mlm_encoded:
                self.original_sequence = hdf5['original_sequence'][indices]
                self.target_tokens = hdf5['target_tokens'][indices]
                self.target_pos = hdf5['target_pos'][indices]
                self.target_cls = hdf5['target_cls'][indices]

            if self.return_index:
                self.sequence_id = hdf5['sequence_id'][indices]

            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ret_dict = {
            "input_ids": self.input_ids[idx],
            "padding_mask": self.padding_mask[idx],
        }

        if self.mlm_encoded:
            # Process 'target_tokens' and 'target_pos' as in the original code
            neg_one_index = np.where(self.target_tokens[idx] == -1)[0]
            end_slice = neg_one_index[0] if neg_one_index.size > 0 else None
            target_tokens = self.target_tokens[idx][:end_slice]
            target_pos = self.target_pos[idx][:end_slice]

            ret_dict.update({
                "original_sequence": self.original_sequence[idx],
                "target_tokens": target_tokens,
                "target_pos": target_pos,
                "target_cls": self.target_cls[idx],
            })

        if self.return_index:
            ret_dict["sequence_id"] = self.sequence_id[idx]

        return ret_dict

class CustomInMemoryDataset(Dataset):
    def __init__(self, file_path, validation=False, num_val_items=None, val_split=0.1, mlm_encoded=True, inference=False):
        self.file_path = file_path
        self.validation = validation
        self.num_val_items = num_val_items
        self.val_split = val_split
        self.inference = inference
        self.mlm_encoded = mlm_encoded
        self.set_mlm_encoded(mlm_encoded)
        self.load_data()

    def set_mlm_encoded(self, mlm_encoded, return_index=None):
        self.mlm_encoded = mlm_encoded
        if return_index is None:
            self.return_index = not self.mlm_encoded
        else:
            self.return_index = return_index

    def load_data(self):
        with h5py.File(self.file_path, 'r') as hdf5:
            n_items = hdf5['input_ids'].shape[0]
            num_val_items = self.num_val_items
            if num_val_items is None:
                num_val_items = int(n_items * self.val_split)
            num_train_items = n_items - num_val_items

            if self.validation:
                indices = np.arange(0, num_val_items)
            elif self.inference:
                indices = np.arange(0, n_items)
            else:
                indices = np.arange(num_val_items, n_items)

            # Load datasets into memory and move to shared memory
            self.input_ids = torch.from_numpy(hdf5['input_ids'][indices])#.share_memory_()
            self.padding_mask = torch.from_numpy(hdf5['padding_mask'][indices])#.share_memory_()

            if self.mlm_encoded:
                self.original_sequence = torch.from_numpy(hdf5['original_sequence'][indices])#.share_memory_()
                self.target_tokens = torch.from_numpy(hdf5['target_tokens'][indices])#.share_memory_()
                self.target_pos = torch.from_numpy(hdf5['target_pos'][indices])#.share_memory_()
                self.target_cls = torch.from_numpy(hdf5['target_cls'][indices])#.share_memory_()

            if self.return_index:
                self.sequence_id = torch.from_numpy(hdf5['sequence_id'][indices])#.share_memory_()

            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ret_dict = {
            "input_ids": self.input_ids[idx],
            "padding_mask": self.padding_mask[idx],
        }

        if self.mlm_encoded:
            # Handle variable-length target sequences
            target_tokens = self.target_tokens[idx]
            target_pos = self.target_pos[idx]
            neg_one_index = (target_tokens == -1).nonzero(as_tuple=True)[0]
            end_slice = neg_one_index[0] if len(neg_one_index) > 0 else target_tokens.size(0)
            target_tokens = target_tokens[:end_slice]
            target_pos = target_pos[:end_slice]

            ret_dict.update({
                "original_sequence": self.original_sequence[idx],
                "target_tokens": target_tokens,
                "target_pos": target_pos,
                "target_cls": self.target_cls[idx],
            })

        if self.return_index:
            ret_dict["sequence_id"] = self.sequence_id[idx]

        return ret_dict

# class CustomDataset(Dataset):
#     def __init__(self, data, mlm_encoded=True):
#       self.data = data
#       self.set_mlm_encoded(mlm_encoded)

#     def set_mlm_encoded(self, mlm_encoded):
#       self.mlm_encoded = mlm_encoded
#       self.return_index = not self.mlm_encoded
    
#     def __len__(self):
#         return self.data["input_ids"].shape[0]
#     def __reduce__(self):
#         return (self.__class__, (self.data,))

#     def __getitem__(self, index):
#         ret_dict = {            
#             "input_ids": self.data["input_ids"][index],
#             "padding_mask": self.data["padding_mask"][index],
#         }

#         if self.mlm_encoded:
#           ret_dict.update(
#             {
#               "original_sequence": self.data["original_sequence"][index],
#               "target_tokens": self.data["target_tokens"][index],
#               "target_pos": self.data["target_pos"][index],
#               "target_cls": self.data["target_cls"][index],
#             }
#           )

#         if self.return_index:
#           ret_dict["sequence_id"] = self.data["sequence_id"][index]

#         return ret_dict


class FineTuneInMemoryDataset(Dataset):
    """
    Dataset class for fine-tuning.
    Merges sequences.h5 data with a CSV or Parquet file
    that contains labels (or other targets).
    """
    def __init__(
        self,
        h5_file_path,
        train_file_path,
        target_col='target_label',  # name of the column containing the label
        phase='train',
        num_val_items=None,
        val_split=0.1,
        return_sequence_id=False,  # Whether to return the sequence_id
        task_type='classification',  # 'classification' or 'regression'
    ):
        """
        :param h5_file_path: Path to the .h5 file, e.g. 'sequences.h5'
        :param train_file_path: Path to the train file, e.g. 'train.csv' or 'train.parquet'
        :param target_col: Column name in the train file containing the target label
        :param phase: one of "train", "validation", "test"
        :param num_val_items: Optional fixed number of validation items
        :param val_split: If num_val_items is None, use this fraction for validation
        :param return_sequence_id: If True, return the sequence_id in __getitem__
        :param task_type: 'classification' or 'regression'; used to set label dtype
        """
        super().__init__()

        self.h5_file_path = h5_file_path
        self.train_file_path = train_file_path
        self.target_col = target_col
        self.phase = phase
        self.num_val_items = num_val_items
        self.val_split = val_split
        self.return_sequence_id = return_sequence_id
        self.task_type = task_type

        # 1) Read the label file (CSV or Parquet)
        self.label_df = self._load_label_file(self.train_file_path)

        # 2) Load data from the HDF5 and intersect with label_df
        self._load_h5_and_intersect()

    def _load_label_file(self, path):
        """
        Load the CSV or Parquet into a DataFrame with columns:
          RINPERSOON  target_label
        (Your actual column names may differ, but must contain a unique ID
         that matches 'sequence_id' in the HDF5.)
        """
        ext = os.path.splitext(path)[1]
        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f'Unsupported training file extension: {ext}')

        if 'RINPERSOON' not in df.columns:
            raise ValueError("Train file must have a 'RINPERSOON' column")

        df.set_index('RINPERSOON', inplace=True)
        return df

    def _load_h5_and_intersect(self):
        """
        Reads sequences.h5 and retains only those IDs that also exist in the label file.
        Splits data for train/validation/test as requested.
        """
        with h5py.File(self.h5_file_path, 'r') as hdf5:
            # 1) Get sequence IDs
            all_seq_ids = hdf5['sequence_id'][:]  # shape: (N,)
            n_items = len(all_seq_ids)

            # 2) Figure out split
            if self.num_val_items is None:
                self.num_val_items = int(n_items * self.val_split)
            full_indices = np.arange(n_items)

            if self.phase == 'test':
                selected_indices = full_indices
            elif self.phase == 'validation':
                selected_indices = full_indices[:self.num_val_items]
            else:  # 'train'
                selected_indices = full_indices[self.num_val_items:]

            selected_seq_ids = all_seq_ids[selected_indices]

            # 3) Create a set for fast membership checks
            label_id_set = set(self.label_df.index)

            # 4) Identify intersection: which IDs appear in both?
            mask = np.array([sid in label_id_set for sid in selected_seq_ids])
            final_indices = np.where(mask)[0]
            
            # 5) Build final subset of indices & sequence IDs
            final_h5_indices = selected_indices[final_indices]
            self.sequence_id = torch.from_numpy(selected_seq_ids[final_indices])

            # 6) Load the relevant data from HDF5 only once
            self.input_ids = torch.from_numpy(hdf5['input_ids'][final_h5_indices])
            self.padding_mask = torch.from_numpy(hdf5['padding_mask'][final_h5_indices])

        # 7) Gather the labels in the correct order
        #    We know each self.sequence_id is in label_id_set, so no NaN.
        raw_labels = self.label_df.loc[self.sequence_id.numpy(), self.target_col].values

        if self.task_type == 'classification':
            self.labels = torch.tensor(raw_labels, dtype=torch.long)
        elif self.task_type == 'regression':
            self.labels = torch.tensor(raw_labels, dtype=torch.float)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        # 8) Our final local indices array
        self.indices = np.arange(len(final_indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        arr_idx = self.indices[idx]
        ret_dict = {
            "input_ids": self.input_ids[arr_idx],
            "padding_mask": self.padding_mask[arr_idx],
            "label": self.labels[arr_idx],
        }
        if self.return_sequence_id:
            ret_dict["sequence_id"] = self.sequence_id[arr_idx]
        return ret_dict

class CustomLazyHDF5Dataset(Dataset):
    """
    Lazy HDF5 reader that mirrors the behaviour of `CustomInMemoryDataset`
    without loading the entire dataset into RAM.
    """

    def __init__(
        self,
        file_path: str | Path,
        validation: bool = False,
        num_val_items: Optional[int] = None,
        val_split: float = 0.10,
        mlm_encoded: bool = True,
        return_index: Optional[bool] = False,
        inference: bool = False,
        needed_id_set = None,
    ) -> None:
        """
        Parameters
        ----------
        file_path : str | Path
            Path to the `.h5` file.
        validation : bool, default False
            If True, yield the `val_split` fraction of samples.
        num_val_items: int | None, default None
            Exact number of validation items. Overrides `val_split`.
        val_split : float, default 0.10
            Fraction of the file that belongs to the validation split.
        mlm_encoded : bool, default True
            Whether the file contains masked-LM target columns.
        return_index : bool | None, default None
            If None, the index is returned only when `mlm_encoded` is False.
            Otherwise forces returning the "sequence_id" column.
        inference : bool, default False
            If True, return the *whole* dataset irrespective of the split.
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.validation = validation
        self.val_split = val_split
        self.inference = inference

        # masked-LM columns
        self.set_mlm_encoded(mlm_encoded, return_index)

        # Compute split indices once (cheap, uses only the header)
        with h5py.File(self.file_path, "r") as f:
            n_items = len(f["input_ids"])
        
        if needed_id_set is not None:
            self._indices = self._keep_needed_indices(
                needed_id_set, num_val_items, val_split
            )
        else:
            n_val = int(n_items * val_split) if num_val_items is None else num_val_items
            if self.inference:
                self._indices = np.arange(0, n_items, dtype=np.int64)
            elif self.validation:
                self._indices = np.arange(0, n_val, dtype=np.int64)
            else:
                self._indices = np.arange(n_val, n_items, dtype=np.int64)

        # This attribute will hold the *worker-local* handle
        self._h5: Optional[h5py.File] = None

    def _keep_needed_indices(
        self,
        needed_id_set,
        num_val_items: Optional[int],
        val_split: float,
    ) -> np.ndarray:
        """determine which rows of the HDF5 we need *and* only keep those."""
        with h5py.File(self.file_path, "r") as h5:
            n_items = len(h5["sequence_ids"])
            n_val = int(n_items * val_split) if num_val_items is None else num_val_items

            # 1) choose the coarse split (train/val/test)
            if self.inference:
                candidate_rows = np.arange(0, n_items, dtype=np.int64)
            elif self.validation:
                candidate_rows = np.arange(0, n_val, dtype=np.int64)
            else: # training
                candidate_rows = np.arange(n_val, n_items, dtype=np.int64)
            
            # 2) slice the `sequence_id` column just once
            seq_ids = h5["sequence_id"][candidate_rows]

            # 3) keep only rows that also appear in the needed_id_set
            mask = np.isin(seq_ids, list(needed_id_set))
            final_rows = candidate_rows[mask]

        return final_rows

    def set_mlm_encoded(
        self,
        mlm_encoded: bool,
        return_index: Optional[bool] = None
    ) -> None:
        self.mlm_encoded = mlm_encoded
        self.return_index = (not mlm_encoded) if return_index is None else return_index

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Dataset API
    # # # # # # # # # # # # # # # # # # # # # # # #
    def __len__(self) -> int:
        return len(self._indices)
        
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 1) get item, lazy HDF5 handle is initialised for *this* worker
        h5 = self._get_handle()

        # 2) translate `idx` (0, ..., len(self)-1) into the row inside the file
        row = self._indices[idx]

        # 3) mandatory columns
        sample: dict[str, torch.Tensor] = {
            "input_ids": torch.as_tensor(h5["input_ids"][row]),
            "padding_mask": torch.as_tensor(h5["padding_mask"][row]),
        }

        # 4) MLM-specific columns (optional)
        if self.mlm_encoded:
            tok = torch.as_tensor(h5["target_tokens"][row])
            pos = torch.as_tensor(h5["target_pos"][row])
            
            # cut at first -1 sentinel (variable-length targets)
            try:
                nz = (tok == -1).nonzero(as_tuple=False)
                end = nz[0, 0].item() if nz.numel() else tok.size(0)
                tok = tok[:end]
                pos = pos[:end]
            except IndexError: # no -1 found
                pass
            
            sample.update(
                original_sequence=torch.as_tensor(
                    h5["original_sequence"][row]
                ),
                target_tokens=tok,
                target_pos=pos,
                target_cls=torch.as_tensor(h5["target_cls"][row]),
            )

        # 5) optionally return the sequence id
        if self.return_index:
            sample["sequence_id"] = torch.as_tensor(h5["sequence_id"][row])

        return sample

    # # # # # # # # # # # # # # # # # # # # # # # #
    # private utilities
    # # # # # # # # # # # # # # # # # # # # # # # #
    def _get_handle(self) -> h5py.File:
        """
        Open the HDF5 file once per worker* (lazily).
        *Lightning's DataLoader forks workers, so every fork gets its own
        copy of `self._h5` - this function ensures we open it lazily
        instead of at import-time (which would break after fork/spawn).
        """
        if self._h5 is None:
            # cannot enable SWMR as many readers may coexist
            self._h5 = h5py.File(
                self.file_path,
                mode="r",
                libver="latest",
                swmr=True,
            )
        return self._h5

    # # # # # # # # # # # # # # # # # # # # # # # #
    # multiprocessing-pickling safety
    # # # # # # # # # # # # # # # # # # # # # # # #
    def __getstate__(self):
        """drop the file handle so torch.multiprocessing can pickle us."""
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        """close the HDF5 file when the worker exits."""
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


class FineTuneLazyDataset(Dataset):
    """
    Lazy-loading Dataset class for fine-tuning.
    Merges sequences.h5 data with a CSV or Parquet file
    that contains labels (or other targets).
    """
    def __init__(
        self,
        h5_file_path: str | Path,
        train_file_path: str | Path,
        target_col: str='target_label',  # name of the column containing the label
        phase: str='train',
        num_val_items: int | None=None,
        val_split: float=0.1,
        return_sequence_id: bool=False,  # Whether to return the sequence_id
        task_type: str='binary',  # 'binary' or 'regression'
        assign_weight: bool=False,  # Optional dict mapping label -> weight
        primary_key: str='RINPERSOON',  # Column name for unique ID in label file
    ):
        """
        :param h5_file_path: Path to the .h5 file, e.g. 'sequences.h5'
        :param train_file_path: Path to the train file, e.g. 'train.csv' or 'train.parquet'
        :param target_col: Column name in the train file containing the target label
        :param phase: one of "train", "validation", "test"
        :param num_val_items: Optional fixed number of validation items
        :param val_split: If num_val_items is None, use this fraction for validation
        :param return_sequence_id: If True, return the sequence_id in __getitem__
        :param task_type: 'binary' or 'regression'; used to set label dtype
        """
        super().__init__()

        self.h5_path = Path(h5_file_path)
        self.train_file_path = Path(train_file_path)
        self.target_col = target_col
        self.phase = phase
        self.return_sequence_id = return_sequence_id
        self.task_type = task_type
        self.assign_weight = assign_weight

        # 1) Read the label file (CSV or Parquet)
        self.label_df = self._load_label_file(self.train_file_path, primary_key)

        # 2) Load data from the HDF5 and intersect with label_df
        (self._indices, self._sequence_ids, self._labels) = self._build_indices_and_labels(
            num_val_items=num_val_items, 
            val_split=val_split,
        )

        if self.phase == 'test':
            self.labels_tensor = None
        elif self.task_type in [ 'categorical','binary', 'ordinal']:
            self.labels_tensor = torch.tensor(self._labels, dtype=torch.long)
            if self.assign_weight:
                counts = Counter(self.labels_tensor.tolist())
                weights = [1.0 / counts[int(label)] for label in self.labels_tensor]
                self.sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        elif self.task_type == 'numeric':
            self.labels_tensor = torch.tensor(self._labels, dtype=torch.float)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
        
        self._h5: Optional[h5py.File] = None  # Will be opened lazily

        logger.info(
            f"{self.phase} split: {len(self)} items "
            f"({len(self._labels)} labels, {len(self._indices)} rows)"
        )
    
    # -------------------------------------------------------------------- #
    # public helper methods                                                #
    # -------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h5 = self._get_handle()
        row = self._indices[idx]

        sample = {
            "input_ids": torch.as_tensor(h5['input_ids'][row]),
            "padding_mask": torch.as_tensor(h5['padding_mask'][row]),
        }
        if self.phase != 'test':
            sample["target"] = self.labels_tensor[idx]
        if self.return_sequence_id:
            sample["sequence_id"] = torch.tensor(self._sequence_ids[idx])
        
        return sample

    # -------------------------------------------------------------------- #
    # internal helper methods                                              #
    # -------------------------------------------------------------------- #

    def _load_label_file(self, path: str | Path, primary_key: str = "RINPERSOON") -> pd.DataFrame:
        """
        Load the CSV or Parquet into a DataFrame with columns:
          RINPERSOON  target_label
        (Your actual column names may differ, but must contain a unique ID
         that matches 'sequence_id' in the HDF5.)
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        if primary_key not in df.columns:
            raise ValueError(f"Train file must have a '{primary_key}' column")
        
        subset = [primary_key]
        if self.phase != 'test':
            subset.insert(0, self.target_col)

        df.dropna(subset=subset, inplace=True)
        # Ensure the primary key is set as the index
        df.set_index(primary_key, inplace=True)
        return df
    
    def _build_indices_and_labels(
        self, 
        num_val_items: Optional[int], 
        val_split: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads sequences.h5 and retains only those IDs that also exist in the label file.
        Splits data for train/validation/test as requested.

        Returns:
            - final_rows: np.ndarray of row indices in the HDF5 file
            - final_sequence_ids: np.ndarray of sequence IDs corresponding to those rows
            - labels: np.ndarray of labels corresponding to those sequence IDs
        """
        label_id_set = set(self.label_df.index)

        with h5py.File(self.h5_path, 'r') as h5:
            n_items = len(h5['sequence_id'])
            n_val = num_val_items or int(n_items * val_split)

            # 1) choose the coarse split (train/val/test)
            if self.phase == 'test':
                candidate_rows = np.arange(0, n_items, dtype=np.int64)
            elif self.phase == 'validation':
                candidate_rows = np.arange(0, n_val, dtype=np.int64)
            elif self.phase == 'train':
                candidate_rows = np.arange(n_val, n_items, dtype=np.int64)
            else:
                raise ValueError(f"Unknown phase: {self.phase}")
            
            # 2) slice the *sequence_id* column just once
            seq_ids = h5['sequence_id'][candidate_rows]

            # 3) keep only rows tha also appear in the label file
            mask = np.isin(seq_ids, list(label_id_set))
            final_rows = candidate_rows[mask]
            final_seq_ids = seq_ids[mask]

        # 4) pull labels in the same order
        if self.phase == 'test':
            labels = np.empty(len(final_seq_ids), dtype=np.float32)
        else:
            labels = self.label_df.loc[final_seq_ids, self.target_col].values

        return final_rows, final_seq_ids, labels
    
    # -------------------------------------------------------------------- #
    # HDFS handle management                                               #
    # -------------------------------------------------------------------- #
    def _get_handle(self) -> h5py.File:
        """
        Lazily open the HDF5 file handle.
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, mode='r', libver='latest', swmr=True)
        return self._h5
    
    # -------------------------------------------------------------------- #
    # multiprocessing-pickling safety                                      #
    # -------------------------------------------------------------------- #
    def __getstate__(self):
        """"Drop open file handle when pickling the Dataset."""
        state = self.__dict__.copy()
        state['_h5'] = None  # do not pickle the HDF5 handle
        return state
    
    def __del__(self):
        """Ensure the HDF5 file is closed when the Dataset is deleted."""
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass