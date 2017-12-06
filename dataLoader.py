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
import time
import logging
import threading
import traceback
import numpy as np
from HTK_IO import HTKFeat_read, HTKFeat_write
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
    'int64'  : torch.LongTensor,
    'int32'  : torch.IntTensor,
    'int16'  : torch.ShortTensor,
    'int8'   : torch.CharTensor,
    'uint8'  : torch.ByteTensor,
}


def convert_utts_list_tensor(data, tensor, uttsLength):
    "Convert data[batch_size][uttLen]([context_window])[vec_len] to a tensor."
    dim_cnt = 2
    ctx_len = 1 # Context Window Length
    vec_len = 1
    if type(data[0]).__module__ == np.__name__: # numpy ndarray list
        if len(data[0].shape) >= 2:
            dim_cnt = len(data[0].shape) + 1
            vec_len = data[0].shape[-1]
            ctx_len = data[0].shape[-2]
    else:   # ``List'' type in python
        subdata = data[0][0]
        while (type(subdata) == list):
            dim_cnt += 1
            ctx_len = vec_len
            vec_len = len(subdata)
            subdata = subdata[0]

    maxLen = max(uttsLength)
    if (dim_cnt == 3):  # Context Window = 0
        ctx_len = 1
    dataTensor = tensor(len(data), maxLen, ctx_len, vec_len).zero_()
    dataTensor.squeeze_(dataTensor.dim()-2) # Context Window Squeeze
    dataTensor.squeeze_(dataTensor.dim()-1) # Vector Length Squeeze

    #for utt in range(len(data)):
    #    if type(data[0]).__module__ == np.__name__: # numpy ndarray list
    #        dataTensor[utt][0:uttsLength[utt]].copy_(torch.from_numpy(data[utt]))
    #    else: # ``List'' type in python
    #        dataTensor[utt][0:uttsLength[utt]].copy_(tensor(data[utt]))
    for data_idx, data_item in enumerate(data):
        if type(data_item).__module__ == np.__name__:   # List of numpy ndarray
            dataTensor[data_idx][0:uttsLength[data_idx]].copy_(torch.from_numpy(data_item))
        else:   # ``List'' type in python
            dataTensor[data_idx][0:uttsLength[data_idx]].copy_(tensor(data_item))
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
        elif isinstance(batch[0][0][0], string_classes):
            return batch

    valid_type = {'tensor', 'number', 'dict', 'list'}
    raise TypeError(("batch must contain {%r}; found {}".format(valid_type, type(batch[0]))))


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
    "Iterates once over the DataLoader's dataset"
    """ Methods:
            next()
            normalize()
    """
    def __init__(self, loader):
        self.logger = loader.logger
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.frame_mode = loader.frame_mode
        self.eval_mode  = loader.eval_mode
        self.random_size = min(loader.random_size, self.dataset.inputs[0]['total_nframes'])
        self.epoch_size = min(loader.epoch_size, self.dataset.inputs[0]['total_nframes'])
        self.context_window = [
            [input['context_window'] for input in self.dataset.inputs],
            [target['context_window'] for target in self.dataset.targets]
        ]

        self.truncate_size = loader.truncate_size
        self.collate_fn = loader.collate_fn
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.drop_last = loader.drop_last
        self.done_event = threading.Event()
        self.random_seed = loader.random_seed
        self.permutation = loader.permutation
        np.random.seed(self.random_seed)
        self.all_keys = loader.dataset.all_keys

        self.epoch_samples_remaining = self.epoch_size
        self.random_samples_remaining = 0
        self.random_utts_remaining = 0
        self.random_utt_idx = loader.random_utt_idx

        # In frame_mode, the start/end index of the utterance for each frame.
        self.utt_start_index = None
        self.utt_end_index = None
        self.random_block_keys = None

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

        self.logger.info('HTKDataLoaderIterator initialization close.')


    def __len__(self):
        if self.frame_mode:
            data_cnt = self.dataset.inputs[0]['total_nframes']
        else:
            data_cnt = self.dataset.inputs[0]['nUtts']
        if self.drop_last:
            return data_cnt // self.batch_size
        else:
            return (data_cnt + self.batch_size - 1) // self.batch_size


    def _get_default_tensor(self, data):
        tmp_data = data
        while (True):
            if type(tmp_data)==list:
                tmp_data = tmp_data[0]
                continue
            elif type(tmp_data).__module__ == np.__name__:
                return numpy_type_map[str(tmp_data.dtype)]
            elif isinstance(tmp_data, int):
                return torch.IntTensor
            elif isinstance(tmp_data, float):
                return torch.FloatTensor
            else:
                raise Exception("DataLoaderIter only support numpy, int and float type.")


    def _frame_feature_augmentation(self, dataidx, blockidx, batchidxs):
        if self.context_window[dataidx][blockidx] is None: # MLF have no context
            return self.block_data[dataidx][blockidx][batchidxs]

        left_context = self.context_window[dataidx][blockidx][0]
        right_context = self.context_window[dataidx][blockidx][0]
        context_len = left_context + right_context + 1

        # Context Window = 0
        if (left_context == 0 and right_context == 0):
            return self.block_data[dataidx][blockidx][batchidxs]

        # Context Window > 0
        subdata = self.block_data[dataidx][blockidx]
        vec_len = subdata[0].shape[0]
        cnt = 0
        ret = np.zeros((len(batchidxs),context_len,vec_len), dtype=subdata.dtype)
        for i in batchidxs:
            left_cnt = min(left_context, i-self.utt_start_index[i])
            right_cnt = min(right_context, self.utt_end_index[i]-i)
            left_aug_idx = left_context - left_cnt
            right_aug_idx = left_context + 1 + right_cnt
            ret[cnt][left_aug_idx:right_aug_idx] = np.copy(subdata[i-left_cnt:i+right_cnt+1])
            cnt += 1
        return ret


    def _utterance_feature_augmentation(self, dataidx, blockidx, batchidxs):
        if self.context_window[dataidx][blockidx] is None:  # MLF have no context
            return self.block_data[dataidx][blockidx][batchidxs]

        left_context  = self.context_window[dataidx][blockidx][0]
        right_context = self.context_window[dataidx][blockidx][1]
        context_len = left_context + 1 + right_context

        # Context Window = 0
        if (left_context == 0 and right_context == 0):
            return self.block_data[dataidx][blockidx][batchidxs]

        # Context Window > 0
        subdata = self.block_data[dataidx][blockidx]
        vec_len = subdata[0].shape[1]
        ret = []
        for i in batchidxs:
            utt_aug = np.zeros((subdata[i].shape[0],context_len,vec_len), dtype=subdata[0].dtype)
            for j in range(subdata[i].shape[0]):
                left_cnt = min(left_context, j-self.utt_start_index[i][j])
                right_cnt = min(right_context, self.utt_end_index[i][j] - j)
                left_aug_idx = left_context - left_cnt
                right_aug_idx = left_context + 1 + right_cnt
                utt_aug[j][left_aug_idx:right_aug_idx] = np.copy(subdata[i][j-left_cnt:j+right_cnt+1])
            ret.append(utt_aug)
        return ret


    def __next__(self):
        if self.num_workers == 0:   # same_process loading
            if self.drop_last and self.epoch_samples_remaining < self.batch_size:
                self.epoch_samples_remaining = self.epoch_size
                raise StopIteration
            if self.epoch_samples_remaining <= 0:
                self.epoch_samples_remaining = self.epoch_size
                raise StopIteration
            if self.random_samples_remaining == 0:  # Load next block data
                self.block_data, self.random_block_keys, self.random_samples_remaining = self._next_random_block()

            indices, lengths, keys = self._next_batch_indices(self.random_block_keys)
            if not self.frame_mode:
                sorted_lengths, order = torch.sort(torch.IntTensor(lengths), 0, descending=True)
                keys = [keys[i] for i in order]

            inputs  = [] if self.block_data[0] else None
            targets = [] if self.block_data[1] else None
            batch   = [inputs, targets]
            for i in range(len(self.block_data)):
                if batch[i] is None: continue

                for j in range(len(self.block_data[i])):
                    tmp_batch = []
                    if self.frame_mode:
                        tmp_batch = self._frame_feature_augmentation(i, j, indices)
                        batch[i].append(torch.from_numpy(tmp_batch))
                    else:
                        tmp_batch = list(self._utterance_feature_augmentation(i, j, indices))
                        #batch[i].append(self.collate_fn(tmp_batch, self.frame_mode))
                        defaultTensor = self._get_default_tensor(tmp_batch)
                        batch[i].append(convert_utts_list_tensor(tmp_batch, defaultTensor, lengths)[order])

            if self.pin_memory:
                batch = pin_memory_batch(batch)
        if self.frame_mode:
            return batch, None, None
        else:
            return batch, list(sorted_lengths), keys

    next = __next__     # Python 2 compatibility

    def __iter__(self):
        return self

    def _next_batch_indices(self, random_block_keys):
        batch_size = min(self.epoch_samples_remaining, self.batch_size, self.random_samples_remaining) if self.frame_mode else min(self.random_utts_remaining, self.batch_size)
        batch_indices = [next(self.perm_indices) for _ in range(batch_size)]
        batch_lengths = []
        batch_keys = None

        if self.frame_mode:
            self.epoch_samples_remaining -= batch_size
            self.random_samples_remaining -= batch_size
            return batch_indices, None, None
        else:
            batch_keys = []
            self.random_utts_remaining -= batch_size
            for i in range(batch_size):
                key = random_block_keys[batch_indices[i]]
                key2idx0 = self.dataset.inputs[0]['name2idx'][key]
                uttLength = self.dataset.inputs[0]['nframes'][key2idx0]
                batch_lengths.append(uttLength)
                batch_keys.append(key)

                self.epoch_samples_remaining  -= uttLength
                self.random_samples_remaining -= uttLength
            return batch_indices, batch_lengths, batch_keys


    def _random_block_keys(self, norm_mode):
        """Load the next random block keys."""
        random_block_keys = []
        random_samples_remaining = 0
        self.utt_start_index = []
        self.utt_end_index   = []
        num_utts = len(self.all_keys)
        while True:
            key = self.all_keys[self.random_utt_idx]
            key2idx0 = self.dataset.inputs[0]['name2idx'][key]
            if (random_samples_remaining + self.dataset.inputs[0]['nframes'][key2idx0]) <= self.random_size:
                random_block_keys.append(key)
                self.random_utt_idx = (self.random_utt_idx + 1) % num_utts
                utt_len = self.dataset.inputs[0]['nframes'][key2idx0]
                utt_start = random_samples_remaining
                random_samples_remaining += utt_len

                if (norm_mode):
                    if self.random_utt_idx == 0: break
                    continue

                if self.frame_mode:
                    self.utt_start_index += [utt_start] * utt_len
                    self.utt_end_index   += [utt_start + utt_len - 1] * utt_len
                else:
                    self.utt_start_index.append([0] * utt_len)
                    self.utt_end_index.append([utt_len - 1] * utt_len)
            else:
                break
        return random_block_keys, random_samples_remaining


    def _next_random_block(self, norm_mode=False):
        """Load the next random block when the current block is drained."""
        random_block_keys, random_samples_remaining = self._random_block_keys(norm_mode)

        data_cnt = random_samples_remaining if self.frame_mode else len(random_block_keys)
        self.random_utts_remaining = None if self.frame_mode else data_cnt
        self.perm_indices = iter(np.random.permutation(data_cnt)) if self.permutation else iter(np.arange(data_cnt))

        # Read HTK Data of keys list
        data = [self.dataset.inputs, self.dataset.targets]
        inputs_block  = [] if data[0] else None
        targets_block = [] if data[1] else None
        data_block    = [inputs_block, targets_block]
        for i in range(len(data_block)):
            if data_block[i] is None: continue

            for j in range(len(data[i])):
                tmp_block_data = None
                if data[i][j]['data_type'] == 'SCP':
                    tmp_block_data = self._get_SCP_block(data[i][j], random_block_keys)
                elif not norm_mode and data[i][j]['data_type'] == 'MLF':
                    tmp_block_data = self._get_MLF_block(data[i][j], random_block_keys)
                data_block[i].append(tmp_block_data)
        return data_block, random_block_keys, random_samples_remaining


    def _get_SCP_block(self, subdataset, block_keys):
        # Read the HTK feats of a list in a random block
        # :params: subdataset: one input or target in SCP format
        random_block_data = []
        dimension = subdataset['dim']

        for key in block_keys:
            key2idx0 = subdataset['name2idx'][key]
            feat_path = subdataset['data'][key2idx0]
            feat_start = subdataset['start_f'][key2idx0]
            feat_len = subdataset['nframes'][key2idx0]

            htk_reader = HTKFeat_read(feat_path)
            tmp_data = htk_reader.getsegment(feat_start, feat_start+feat_len-1)
            if (tmp_data.shape[1] != dimension):
                raise Exception("Dimension does not match, %d in configure vs. %d in data" % (dimension, tmp_data.shape[1]))
            htk_reader.close()
            random_block_data.append(tmp_data)
        if self.frame_mode:
            return np.concatenate(random_block_data, axis=0)
        else:
            return np.array(random_block_data)


    def _get_MLF_block(self, subdataset, block_keys):
        # Read the HTK feats of a list in a random block
        # :params: subdataset: one input or target in MLF format
        random_block_data = []
        for key in block_keys:
            key2idx0 = subdataset['name2idx'][key]
            tmp_data = subdataset['data'][key2idx0]

            if self.frame_mode:
                random_block_data += tmp_data
            else:
                random_block_data.append(tmp_data)
        return np.array(random_block_data)


    def normalize(self, mode=None):
        """ return mean_data, std_data
        """
        mean_data_block = [[None] * len(self.dataset.inputs),
                           [None] * len(self.dataset.targets)]
        std_data_block  = [[None] * len(self.dataset.inputs),
                           [None] * len(self.dataset.targets)]

        valid_mode = {None, 'globalMean', 'globalVar', 'globalMeanVar'}
        if mode not in valid_mode:
            raise ValueError("normalization must be one of %r" % valid_mode)

        def Mean():
            self.logger.info("DataLoaderIterator: Normalization -- means")
            while True:
                data_block, _, nsamples = self._next_random_block(norm_mode=True)
                for data_idx, data in enumerate(data_block):
                    for block_idx, block in enumerate(data):
                        if block is None: continue
                        total_nframes = dataset[data_idx][block_idx]['total_nframes']
                        if mean_data_block[data_idx][block_idx] is None:
                            mean_data_block[data_idx][block_idx] = np.zeros(block.shape[1])
                        mean_data_block[data_idx][block_idx] += np.mean(block, axis=0) * (nsamples / total_nframes)
                if self.random_utt_idx == 0:
                    break
            return mean_data_block

        def Std():
            self.logger.info("DataLoaderIterator: Normalization -- standard variance")
            while True:
                data_block, _, nsamples = self._next_random_block(norm_mode=True)
                for data_idx, data in enumerate(data_block):
                    for block_idx, block in enumerate(data):
                        if block is None: continue
                        total_nframes = dataset[data_idx][block_idx]['total_nframes']
                        if std_data_block[data_idx][block_idx] is None:
                            std_data_block[data_idx][block_idx] = np.zeros(block.shape[1])
                        tmp_block = block - mean_data_block[data_idx][block_idx]
                        std_data_block[data_idx][block_idx] += np.var(tmp_block, axis=0) * (nsamples / total_nframes)
                if self.random_utt_idx == 0:
                    break
            for data_idx, std_data in enumerate(std_data_block):
                for block_idx, std_block in enumerate(std_data):
                    if not std_block is None:
                        std_data_block[data_idx][block_idx] = np.sqrt(std_block)
            return std_data_block

        # None
        if mode is None:
            return mean_data_block, std_data_block

        frame_mode_backup = self.frame_mode
        self.frame_mode = True
        dataset = [self.dataset.inputs, self.dataset.targets]

        mean_data_block = Mean()
        if mode != 'globalMean': std_data_block = Std()

        self.frame_mode = frame_mode_backup
        return mean_data_block, std_data_block


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
     HTK Data loader. Combines a dataset and provides
     single- or multi-process iterators over the dataset.
     Arguments:
         dataset (Dataset): dataset from which to load the data.
         batch_size (int, optional): how many samples per batch to load (default: 1). In frame_mode,
            batch_size means number of samples, otherwise, it means number of utterances.
         num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
         eval_mode (bool, optional): dataset used for train or evaluation if eval_mode is True, then frame_mode and batch_size will be set to False and 1 respectively.
         collate_fn (callable, optional)
         pin_memory (bool, optional)
         drop_last (bool, optional): set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
         frame_mode(bool, optional): set to ``False`` to enable (utterance) sequence
         permutation(bool, optional): whether permutate the indices in the iterator
         random_size(int, optional): defines the number of frames to load for once
         epoch_size(int, optional): defines the number of frames for each epoch
         truncate_size(int, optional): defines the truncate length in RNN
         random_utt_idx(int, optional): defines the beginning utterance index to be loaded
         random_seed(int, optional): defines the random seed
    """
    def __init__(self, dataset, batch_size=1, num_workers=0, eval_mode=False,
                 collate_fn=default_collate, pin_memory=False, drop_last=False,
                 frame_mode=False, permutation=True, random_size=None,
                 epoch_size=None, truncate_size=0, random_utt_idx=0,
                 random_seed=19931225, logger=None):
        logging.basicConfig(format='[%(name)s] %(levelname)s %(asctime)s: %(message)s', datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)
        self.logger = logger if logger else logging.getLogger('HTKDataLoader')

        self.dataset = dataset
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last  = drop_last
        self.frame_mode  = frame_mode
        self.random_utt_idx = random_utt_idx
        self.random_seed = random_seed
        self.permutation = permutation

        self.batch_size  = batch_size
        self.random_size = random_size if random_size else self.dataset.inputs[0]['total_nframes']
        self.epoch_size  = epoch_size if epoch_size else self.random_size
        self.truncate_size = truncate_size

        self.eval_mode = eval_mode
        if self.eval_mode:
            self.frame_mode = False
            self.permutation = False
        self.logger.info('HTKDataLoader initialization close.')


    def __iter__(self):
        return HTKDataLoaderIter(self)


    def __len__(self):
        """Epoch size."""
        if self.drop_last:
            return self.dataset.inputs[0]['total_nframes'] // self.epoch_size
        else:
            return (self.dataset.inputs[0]['total_nframes'] + self.epoch_size - 1) // self.epoch_size
