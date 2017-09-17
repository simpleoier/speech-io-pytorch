#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.data as torch_data
import collections
import math
import sys
import re
import os
import threading
import traceback
import numpy as np
from HTK_IO import HTKFeat_read, HTKFeat_write
from sampler import FrameLevSampler, UtteranceLevSampler
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)

_use_shared_memory = False
"""Whether to use shared memory in default_collate"""


class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads."

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1)
    while (True):
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


def _pin_memory_loop(in_queue, out_queue, done_event):
    while True:
        try:
            r = in_queue.get()
        except:
            if done_event.is_set():
                return
            raise
        if r is None:
            break
        if isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
            continue
        idx, batch = r
        try:
            batch = pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            out_queue.put((idx, batch))


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def convert_utts_list_tensor(data, tensor):
    "Convert data[batch_size][uttLen][dim] to a tensor."
    uttsLength = [len(utt) for utt in data]
    maxLen = max(uttsLength)
    dim = len(data[0][0])
    dataTensor = tensor(len(data), maxLen, dim).zero_()
    for utt in range(len(data)):
        dataTensor[utt][0:len(data[utt])].copy_(tensor(data[utt]))
    return dataTensor


def default_collate(batch, frame_mode):
    "Puts each data field into a tensor with outer dimension batch size"
    if frame_mode:     # batch[batch_size][dim]
        if torch.is_tensor(batch):
            return batch
        if torch.is_tensor(batch[0]):
            out = None
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        elif type(batch[0]).__module__ == 'numpy':
            elem = batch[0]
            if type(elem).__name__ == 'ndarray':
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], collections.Mapping):
            return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [default_collate(samples) for samples in transposed]
        elif isinstance(batch[0][0], int):
            return torch.LongTensor(batch)
        elif isinstance(batch[0][0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0][0], string_classes):
            return batch
    else:               # batch[batch_size][utt_len][dim]
        if torch.is_tensor(batch):
            return batch
        elif torch.is_tensor(batch[0]):
            out = None
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        elif type(batch[0]).__module__ == 'numpy':
            elem = batch[0]
            if type(elem).__name__ == 'ndarray':
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0][0][0], int):
            return convert_utts_list_tensor(batch, torch.LongTensor)
        elif isinstance(batch[0][0][0], float):
            return convert_utts_list_tensor(batch, torch.DoubleTensor)
        elif isinstance(batch[0][0][0], string_classes):
            return batch

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                 .format(type(batch[0]))))


def pin_memory_batch(batch):
    if torch.is_tensor(batch):
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch


class HTKDataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.frame_mode = loader.frame_mode
        self.random_size = min(loader.random_size, self.dataset.inputs[0]['total_nframes'])
        self.epoch_size = min(loader.epoch_size, self.dataset.inputs[0]['total_nframes'])

        self.truncate_size = loader.truncate_size
        self.collate_fn = loader.collate_fn
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.drop_last = loader.drop_last
        self.done_event = threading.Event()

        self.epoch_samples_remaining = self.epoch_size
        self.random_samples_remaining = 0
        self.random_utts_remaining = 0

        self.random_utt_idx = 0
        self.max_utt_len = self.dataset.max_utt_len2

        self.all_keys = list(loader.dataset.inputs[0]['name2idx'])
        self.random_block_keys = None

        self.block_data = self._next_random_block()

        '''
        if self.num_workers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workders = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn)
                ) for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True     # ensure that the worker exists on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target = _pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()
        '''


    def __len__(self):
        if self.frame_mode:
            data_cnt = self.dataset.inputs[0]['total_nframes']
        else:
            data_cnt = self.dataset.inputs[0]['nUtts']
        if self.drop_last:
            return data_cnt // self.batch_size
        else:
            return (data_cnt + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.num_workers == 0:   # same_process loading
            if self.drop_last and self.epoch_samples_remaining < self.batch_size:
                raise StopIteration
            if self.epoch_samples_remaining <= 0:
                self.epoch_samples_remaining = self.epoch_size
                raise StopIteration
            if self.random_samples_remaining == 0:
                self.block_data = self._next_random_block()

            indices, lengths = self._next_batch_indices()
            inputs = [] if self.block_data[0] else None
            targets = [] if self.block_data[1] else None
            batch = [inputs, targets]
            for i in range(len(self.block_data)):
                if batch[i] is None: continue

                for j in range(len(self.block_data[i])):
                    tmp_batch = []
                    for k in indices:
                        tmp_batch.append(self.block_data[i][j][k])

                    #batch[i].append(self.collate_fn(tmp_batch, self.frame_mode))
                    defaultTensor = torch.DoubleTensor
                    if self.frame_mode:
                        if isinstance(tmp_batch[0][0], int): defaultTensor = torch.LongTensor
                        batch[i].append(defaultTensor(tmp_batch))
                    else:
                        if isinstance(tmp_batch[0][0][0], int): defaultTensor = torch.LongTensor
                        batch[i].append(convert_utts_list_tensor(tmp_batch, defaultTensor))

            if self.pin_memory:
                batch = pin_memory_batch(batch)
        return batch, lengths

    next = __next__     # Python 2 compatibility

    def __iter__(self):
        return self

    def _next_batch_indices(self):
        batch_size = min(self.epoch_samples_remaining, self.batch_size) if self.frame_mode else min(self.random_utts_remaining, self.batch_size)
        batch_indices = [next(self.perm_indices) for _ in range(batch_size)]
        batch_lengths = []
        if self.frame_mode:
            self.epoch_samples_remaining -= batch_size
            self.random_samples_remaining -= batch_size
        else:
            self.random_utts_remaining -= batch_size
            for i in range(batch_size):
                key = self.random_block_keys[batch_indices[i]]
                key2idx0 = self.dataset.inputs[0]['name2idx'][key]
                uttLength = self.dataset.inputs[0]['nframes'][key2idx0]
                self.epoch_samples_remaining  -= uttLength
                self.random_samples_remaining -= uttLength
                batch_lengths.append(uttLength)
        return batch_indices, batch_lengths


    def _next_random_block(self):
        """Load the next random block when the current block is drained."""
        self.random_block_keys = []
        while True:
            key = self.all_keys[self.random_utt_idx]
            key2idx0 = self.dataset.inputs[0]['name2idx'][key]
            if (self.random_samples_remaining + self.dataset.inputs[0]['nframes'][key2idx0]) <= self.random_size:
                self.random_block_keys.append(key)
                self.random_utt_idx = (self.random_utt_idx + 1) % self.dataset.inputs[0]['nUtts']
                self.random_samples_remaining += self.dataset.inputs[0]['nframes'][key2idx0]
            else:
                break
        data_cnt = self.random_samples_remaining if self.frame_mode else len(self.random_block_keys)
        self.random_utts_remaining = None if self.frame_mode else data_cnt

        # Read HTK Data of keys list
        data = [self.dataset.inputs, self.dataset.targets]
        inputs_block = [] if data[0] else None
        targets_block = [] if data[1] else None
        block_data = [inputs_block, targets_block]
        for i in range(len(block_data)):
            if block_data[i] is None: continue

            for j in range(len(data[i])):
                if data[i][j]['data_type'] == 'SCP':
                    tmp_block_data = self._get_SCP_block(data[i][j])
                elif data[i][j]['data_type'] == 'MLF':
                    tmp_block_data = self._get_MLF_block(data[i][j])
                block_data[i].append(tmp_block_data)

        #self.perm_indices = iter(torch.randperm(data_cnt).long()) if data_cnt > 0 else None
        self.perm_indices = iter(torch.randperm(data_cnt).long())
        return block_data


    def _get_SCP_block(self, subdataset):
        # Read the HTK feats of a list in a random block
        # :params: subdataset: one input or target in SCP format
        random_block_data = []
        dimension = subdataset['dim']

        for i in range(len(self.random_block_keys)):
            key = self.random_block_keys[i]
            key2idx0 = subdataset['name2idx'][key]
            feat_path = subdataset['data'][key2idx0]
            feat_start = subdataset['start_f'][key2idx0]
            feat_len = subdataset['nframes'][key2idx0]

            htk_reader = HTKFeat_read(feat_path)
            tmp_data = htk_reader.getsegment(feat_start, feat_start+feat_len-1)
            if (tmp_data.shape[1] != dimension):
                raise Exception("Dimension does not match, %d in configure vs. %d in data" % (dimension, tmp_data.shape[1]))
            htk_reader.close()

            tmp_data = tmp_data.tolist()

            if self.frame_mode:
                random_block_data += tmp_data
            else:
                random_block_data.append(tmp_data)
        return random_block_data


    def _get_MLF_block(self, subdataset):
        # Read the HTK feats of a list in a random block
        # :params: subdataset: one input or target in MLF format
        random_block_data = []

        for i in range(len(self.random_block_keys)):
            key = self.random_block_keys[i]
            key2idx0 = subdataset['name2idx'][key]
            tmp_data = subdataset['data'][key2idx0]

            if self.frame_mode:
                random_block_data += tmp_data
            else:
                random_block_data.append(tmp_data)
        return random_block_data

    '''
    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        if self.samples_remaining > 0:
            if self.samples_remaining < self.batch_size and self.drop_last:
                self._next_indices()
            else:
                self.index_queue.put((self.send_idx, self._next_indices()))
                self.batches_outstanding += 1
                self.send_idx += 1
    '''


    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")


class HTKDataLoader(DataLoader):
    """
     HTK Data loader. Combines a dataset and a sampler, and provides
     single- or multi-process iterators over the dataset.
     Arguments:
         dataset (Dataset): dataset from which to load the data.
         batch_size (int, optional): how many samples per batch to load (default: 1). In frame_mode,
            batch_size means number of samples, otherwise, it means number of utterances.
         num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
         collate_fn (callable, optional)
         pin_memory (bool, optional)
         drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If False and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
         frame_mode(bool, optional): set to ``False`` to enable (utterance) sequence
         random_size(int, optional): defines the number of frames to load for once
         epoch_size(int, optional): defines the number of frames for each epoch
         truncate_size(int, optional): defines the truncate length in RNN
    """
    def __init__(self, dataset, batch_size=1, num_workers=0,
                 collate_fn=default_collate, pin_memory=False, drop_last=False,
                 frame_mode=False, random_size=None, epoch_size=None, truncate_size=0):
        self.dataset = dataset
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last  = drop_last

        self.frame_mode  = frame_mode
        self.batch_size  = batch_size
        self.random_size = random_size if random_size else self.dataset.inputs[0]['total_nframes']
        self.epoch_size  = epoch_size if epoch_size else self.random_size
        self.truncate_size = truncate_size


    def __iter__(self):
        return HTKDataLoaderIter(self)


    def __len__(self):
        """Epoch size."""
        if self.drop_last:
            return self.dataset.inputs[0]['total_nframes'] // self.epoch_size
        else:
            return (self.dataset.inputs[0]['total_nframes'] + self.epoch_size - 1) // self.epoch_size