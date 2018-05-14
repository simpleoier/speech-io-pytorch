#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.data as torch_data
import collections
import math
import sys
import io
import re
import os
import time
import logging
import threading
import traceback
import numpy as np
import tqdm
from HTK_IO import HTKFeat_read, HTKFeat_write
sys.path.append('./kaldi-io-for-python')
from kaldi_io import read_mat
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_use_shared_memory = False
"""Whether to use shared memory in default_collate"""

logging.basicConfig(format='[%(name)s] %(levelname)s %(asctime)s: %(message)s', datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)



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


def convertUttsList2Tensor(data, tensor, uttsLength, padding_value=0):
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
    dataTensor = tensor(len(data), maxLen, ctx_len, vec_len).fill_(padding_value)
    dataTensor.squeeze_(dataTensor.dim()-2) # Context Window Squeeze
    dataTensor.squeeze_(dataTensor.dim()-1) # Vector Length Squeeze

    for data_idx, data_item in enumerate(data):
        if type(data_item).__module__ == np.__name__:   # List of numpy ndarray
            #dataTensor[data_idx][0:uttsLength[data_idx]].copy_(torch.from_numpy(data_item))
            dataTensor[data_idx][0:data_item.shape[0]].copy_(torch.from_numpy(data_item))
        else:   # ``List'' type in python
            #dataTensor[data_idx][0:uttsLength[data_idx]].copy_(tensor(data_item))
            dataTensor[data_idx][0:len(data_item)].copy_(tensor(data_item))
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


class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset"
    """ Methods:
            next()
            normalize()
    """
    def __init__(self, dataloader):
        self.logger = dataloader.logger
        self.logger.info('DataLoaderIterator initialization close.')

        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.frame_mode = dataloader.frame_mode
        self.padding_value = dataloader.padding_value
        self.random_size = min(dataloader.random_size, self.dataset.inputs[0]['total_nframes'])
        self.epoch_size  = min(dataloader.epoch_size,  self.dataset.inputs[0]['total_nframes'])
        self.context_window = [
            [feature['context_window'] for feature in self.dataset.features],
            [target['context_window'] for target in self.dataset.targets]
        ]

        self.truncate_size = dataloader.truncate_size
        self.collate_fn = dataloader.collate_fn
        self.num_workers = dataloader.num_workers
        self.pin_memory = dataloader.pin_memory
        self.drop_last = dataloader.drop_last
        self.done_event = threading.Event()
        self.random_seed = dataloader.random_seed
        self.permutation = dataloader.permutation
        np.random.seed(self.random_seed)
        self.all_keys = dataloader.dataset.all_keys

        self.epoch_samples_remaining = self.epoch_size
        self.random_samples_remaining = 0
        self.random_utts_remaining = 0
        self.utt_iter_idx = dataloader.utt_iter_idx

        self.utt_start_index = []
        self.utt_end_index = []
        self.random_block_keys = []

        self._get_SCP_block = self._get_HTK_SCP_block

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
            data_cnt = self.dataset.features[0]['total_nframes']  # features[0] must be the acoustic features
        else:
            data_cnt = self.dataset.features[0]['nUtts']
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


    def _frame_feature_augmentation(self, data_item, batchidxs, context_window, utt_start_index, utt_end_index):
        # Context Window is None or = 0
        if context_window is None or sum(context_window)==0: # MLF have no context
            return data_item[batchidxs]

        # Context Window > 0
        left_context  = context_window[0]
        right_context = context_window[1]
        context_len = left_context + right_context + 1

        vec_len = data_item[0].shape[0]
        ret = np.zeros((len(batchidxs),context_len,vec_len), dtype=data_item.dtype)
        for i, idx in enumerate(batchidxs):
            left_cnt  = min(left_context, idx-utt_start_index[idx])
            right_cnt = min(right_context, utt_end_index[idx]-idx)
            aug_beg = left_context - left_cnt
            aug_end = left_context + 1 + right_cnt
            ret[i][aug_beg:aug_end] = np.copy(data_item[idx-left_cnt:idx+right_cnt+1])
        return ret


    def _utterance_feature_augmentation(self, data_item, batchidxs, context_window):
        # Context Window is None or = 0
        if context_window is None or sum(context_window)==0: # MLF have no context
            return data_item[batchidxs].tolist()

        # Context Window > 0
        left_context  = context_window[0]
        right_context = context_window[1]
        context_len = left_context + 1 + right_context

        vec_len = data_item[0].shape[1]
        ret = []
        for i, idx in enumerate(batchidxs):
            utt_len = data_item[idx].shape[0]
            utt_aug = np.zeros((utt_len,context_len,vec_len), dtype=data_item[idx].dtype)
            for j in range(utt_len):
                left_cnt  = min(left_context, j)
                right_cnt = min(right_context, utt_len-1 - j)
                aug_beg = left_context - left_cnt
                aug_end = left_context + 1 + right_cnt
                utt_aug[j][aug_beg:aug_end] = np.copy(data_item[idx][j-left_cnt:j+right_cnt+1])
            ret.append(utt_aug)
        return ret


    def _next_batch_frame_mode(self, batch_size):

        def _next_batch_indices(random_block_keys, batch_size):
            batch_size = min(self.epoch_samples_remaining, batch_size, self.random_samples_remaining)
            batch_indices = [next(self.perm_indices) for _ in range(batch_size)]
            self.epoch_samples_remaining -= batch_size
            self.random_samples_remaining -= batch_size
            return batch_indices, None, None

        indices, _, _ = _next_batch_indices(self.random_block_keys, batch_size)
        batch = [[], []]
        for i, data in enumerate(self.block_data):
            for j, data_item in enumerate(data):
                tmp_batch = self._frame_feature_augmentation(data_item, indices, self.context_window[i][j], utt_start_index=self.utt_start_index, utt_end_index=self.utt_end_index)
                batch[i].append(torch.from_numpy(tmp_batch))
        return batch, None, None

    def _next_batch_utts_mode(self, batch_size):

        def _next_batch_indices(random_block_keys, batch_size):
            batch_size = min(self.random_utts_remaining, batch_size)
            batch_indices = [next(self.perm_indices) for _ in range(batch_size)]
            batch_lengths = []
            batch_keys = []

            for i in range(batch_size):
                key = random_block_keys[batch_indices[i]]
                key2idx0 = self.dataset.features[0]['name2idx'][key]
                uttLength = self.dataset.features[0]['nframes'][key2idx0]
                batch_lengths.append(uttLength)
                batch_keys.append(key)
                self.epoch_samples_remaining  -= uttLength
                self.random_samples_remaining -= uttLength
            self.random_utts_remaining -= batch_size
            return batch_indices, batch_lengths, batch_keys

        indices, lengths, keys = _next_batch_indices(self.random_block_keys, batch_size)
        sorted_lengths, order = torch.sort(torch.IntTensor(lengths), 0, descending=True)
        keys  = [keys[i] for i in order]
        batch = [[], []]
        for i, data in enumerate(self.block_data):
            for j, data_item in enumerate(data):
                tmp_batch = self._utterance_feature_augmentation(data_item, indices, self.context_window[i][j])
                #batch[i].append(self.collate_fn(tmp_batch, self.frame_mode))
                defaultTensor = self._get_default_tensor(tmp_batch)
                batch[i].append(convertUttsList2Tensor(tmp_batch, defaultTensor, lengths, padding_value=self.padding_value)[order])
        return batch, list(sorted_lengths), keys

    def _next_batch(self):
        if self.num_workers == 0:   # same_process loading
            if self.drop_last and self.epoch_samples_remaining < self.batch_size:
                self.epoch_samples_remaining = self.epoch_size
                raise StopIteration
            if self.epoch_samples_remaining <= 0:
                self.epoch_samples_remaining = self.epoch_size
                raise StopIteration
            if self.random_samples_remaining == 0:  # Load next block data
                self.random_block_keys, self.random_samples_remaining = self._random_block_keys()
                self.block_data = self._get_block_data_from_keys(self.random_block_keys, frame_mode=self.frame_mode)
                data_cnt = self.random_samples_remaining if self.frame_mode else len(self.random_block_keys)
                self.random_utts_remaining = 0 if self.frame_mode else data_cnt
                self.perm_indices = iter(np.random.permutation(data_cnt)) if self.permutation else iter(np.arange(data_cnt))

            if self.frame_mode:
                 batch, lengths, keys = self._next_batch_frame_mode(self.batch_size)
            else:
                 batch, lengths, keys = self._next_batch_utts_mode(self.batch_size)
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch, lengths, keys
        else:
            raise Exception("NotImplementedError: multi-worker data loader iterator is not implemented.")

    __next__ = _next_batch
    next = __next__     # Python 2 compatibility
    def __iter__(self):
        return self


    def _random_block_keys(self):
        """Load the next random block keys."""
        block_keys = []
        samples_remaining = 0
        utt_start_index = []
        utt_end_index   = []
        total_nutts = len(self.all_keys)

        def start_end_index_frame_mode(samples_remaining, utt_len):
            utt_start_index += [samples_remaining] * utt_len
            utt_end_index   += [samples_remaining + utt_len - 1] * utt_len
            return samples_remaining+utt_len

        def start_end_index_utts_mode(samples_remaining, utt_len):
            utt_start_index.append([0] * utt_len)
            utt_end_index.append([utt_len - 1] * utt_len)
            return samples_remaining+utt_len

        if self.frame_mode:
            start_end_index_update = start_end_index_frame_mode
        else:
            start_end_index_update = start_end_index_utts_mode

        # Until the random block is full
        # TODO: this procedure needs to be modified to be an online loading version
        #       i.e. loading while consuming the data in _next()
        while True:
            key = self.all_keys[self.utt_iter_idx]
            # Here we use the features[0] as the standard utterance length
            key2idx0 = self.dataset.features[0]['name2idx'][key]
            if (samples_remaining + self.dataset.features[0]['nframes'][key2idx0]) <= self.random_size:
                block_keys.append(key)
                self.utt_iter_idx = (self.utt_iter_idx + 1) % total_nutts
                utt_len = self.dataset.features[0]['nframes'][key2idx0]
                samples_remaining = start_end_index_update(samples_remaining, utt_len)
            else:
                break

        self.utt_start_index = utt_start_index
        self.utt_end_index   = utt_end_index
        return block_keys, samples_remaining


    def _get_block_data_from_keys(self, block_keys, frame_mode):
        # Read Data of keys list
        dataset = [self.dataset.features, self.dataset.targets]
        data_block = [[], []]

        for i, data in enumerate(dataset):
            for item in data:
                if item['data_type'] == 'SCP':
                    tmp_data_block = self._get_SCP_block(item, block_keys, frame_mode=frame_mode)
                elif item['data_type'] == 'MLF':
                    tmp_data_block = self._get_MLF_block(item, block_keys, frame_mode=frame_mode)
                data_block[i].append(tmp_data_block)
        return data_block


    def _get_HTK_SCP_block(self, subdataset, block_keys, frame_mode):
        # Read the HTK feats of a list in a random block
        block_data = []
        dimension = subdataset['dim']

        for key in block_keys:
            key2idx0 = subdataset['name2idx'][key]
            feat_path = subdataset['data'][key2idx0]
            feat_start = subdataset['start_f'][key2idx0]
            feat_len = subdataset['nframes'][key2idx0]

            htk_reader = HTKFeat_read(feat_path)
            htk_data = htk_reader.getsegment(feat_start, feat_start+feat_len-1)
            if (htk_data.shape[1] != dimension):
                raise Exception("HTK SCP Block: dimension does not match, %d in configure vs. %d in data" % (dimension, htk_data.shape[1]))
            htk_reader.close()
            block_data.append(htk_data)

        if frame_mode:
            return np.concatenate(block_data, axis=0)
        else:
            return np.array(block_data)

    def _get_Kaldi_SCP_block(self, subdataset, block_keys, frame_mode):
        # Read the Kaldi feats of a list in a random block
        block_data = []
        dimension = subdataset['dim']

        for key in block_keys:
            key2idx0 = subdataset['name2idx'][key]
            feat_path = subdataset['data'][key2idx0]

            kaldi_data = read_mat(feat_path)
            if (kaldi_data.shape[1] != dimension):
                raise Exception("Kaldi SCP Block: dimension does not match, %d in configure vs. %d in data" % (dimension, htk_data.shape[1]))
            block_data.append(kaldi_data)

        if frame_mode:
            return np.concatenate(block_data, axis=0)
        else:
            return np.array(block_data)


    def _get_MLF_block(self, subdataset, block_keys, frame_mode):
        # Read the MLF feats of a list in a random block
        # :params: subdataset: one input or target in MLF format
        block_data = []
        if frame_mode:
            append_data = lambda mlf_data: block_data.extend(mlf_data)
        else:
            append_data = lambda mlf_data: block_data.append(mlf_data)

        for key in block_keys:
            key2idx0 = subdataset['name2idx'][key]
            append_data(subdataset['data'][key2idx0])
        return np.array(block_data)


    def priors(self):
        """ return prior for MLF."""
        self.logger.info("DataLoaderIterator: Priors")
        priors = [[None] * len(self.dataset.features),
                 [None] * len(self.dataset.targets)]
        dataset = [self.dataset.features, self.dataset.targets]
        # initialization
        for data_idx, data in enumerate(dataset):
            for item_idx, item in enumerate(data):
                if item['data_type'] != 'MLF': continue
                prior_item = np.ones(dataset[data_idx][item_idx]['dim'])
                for key in self.all_keys:
                    key2idx0 = dataset[data_idx][item_idx]['name2idx'][key]
                    prior_item[dataset[data_idx][item_idx]['data'][key2idx0]] += 1
                priors[data_idx][item_idx] = prior_item / (dataset[data_idx][item_idx]['total_nframes'] + dataset[data_idx][item_idx]['dim'])
        return priors


    def normalize(self, mode=None):
        """ return mean_data, std_data
        """
        mean_data_block = [[None] * len(self.dataset.features),
                           [None] * len(self.dataset.targets)]
        std_data_block  = [[None] * len(self.dataset.features),
                           [None] * len(self.dataset.targets)]

        valid_mode = {None, 'globalMean', 'globalVar', 'globalMeanVar'}
        if mode not in valid_mode:
            raise ValueError("normalization must be one of %r" % valid_mode)

        def _get_block_keys():
            step = int(len(self.all_keys) // (self.dataset.features[0]['total_nframes'] // self.random_size + 1))
            for i in range(0, len(self.all_keys), step):
                block_keys = self.all_keys[i:i+step]
                nsamples = 0
                for key in block_keys:
                    key2idx0  = self.dataset.features[0]['name2idx'][key]
                    nsamples += self.dataset.features[0]['nframes'][key2idx0]
                yield block_keys, nsamples

        def initialize(params):
            for data_idx, data in enumerate(dataset):
                for item_idx, item in enumerate(data):
                    if item['data_type'] == 'SCP':
                        params[data_idx][item_idx] = np.zeros(item['dim'])

        def Mean():
            self.logger.info("DataLoaderIterator: Normalization -- means")
            initialize(mean_data_block)

            for block_keys, nsamples in _get_block_keys():
                for data_idx, mean_data in enumerate(mean_data_block):
                    for block_idx, mean_block in enumerate(mean_data):
                        if mean_block is None: continue

                        total_nframes = dataset[data_idx][block_idx]['total_nframes']
                        block = self._get_SCP_block(dataset[data_idx][block_idx], block_keys, frame_mode=True)
                        mean_data_block[data_idx][block_idx] += np.mean(block, axis=0) * (nsamples / total_nframes)

            return mean_data_block

        def Std():
            self.logger.info("DataLoaderIterator: Normalization -- standard variance")
            initialize(std_data_block)

            for block_keys, nsamples in _get_block_keys():
                for data_idx, std_data in enumerate(std_data_block):
                    for block_idx, std_block in enumerate(std_data):
                        if std_block is None: continue

                        total_nframes = dataset[data_idx][block_idx]['total_nframes']
                        block = self._get_SCP_block(dataset[data_idx][block_idx], block_keys, frame_mode=True)
                        std_data_block[data_idx][block_idx] += np.var(block-mean_data_block[data_idx][block_idx], axis=0) * (nsamples / total_nframes)

            # Sqrt
            for data_idx, std_data in enumerate(std_data_block):
                for block_idx, std_block in enumerate(std_data):
                    if not std_block is None:
                        std_data_block[data_idx][block_idx] = np.sqrt(std_block)
            return std_data_block

        def MeanStd():
            self.logger.info("DataLoaderIterator: Normalization -- mean & standard variance")
            initialize(mean_data_block)
            initialize(std_data_block)

            for block_keys, nsamples in _get_block_keys():
                for data_idx, std_data in enumerate(std_data_block):
                    for block_idx, std_block in enumerate(std_data):
                        if std_block is None: continue

                        total_nframes = dataset[data_idx][block_idx]['total_nframes']
                        block = self._get_SCP_block(dataset[data_idx][block_idx], block_keys, frame_mode=False)
                        mean_data_block[data_idx][block_idx] += np.mean(block, axis=0) * (nsamples / total_nframes)
                        std_data_block[data_idx][block_idx] += np.var(block-mean_data_block[data_idx][block_idx], axis=0) * (nsamples / total_nframes)
            # Sqrt
            for data_idx, std_data in enumerate(std_data_block):
                for block_idx, std_block in enumerate(std_data):
                    if not std_block is None:
                        std_data_block[data_idx][block_idx] = np.sqrt(std_block)
            return std_data_block

        if mode is None:
            return mean_data_block, std_data_block

        dataset = [self.dataset.features, self.dataset.targets]
        mean_data_block = Mean()
        if mode != 'globalMean': std_data_block = Std()

        return mean_data_block, std_data_block


    def eval(self):
        for key in self.all_keys:
            batch_data = self._get_block_data_from_keys([key], frame_mode=False)
            length = self.dataset.features[0]['nframes'][ self.dataset.features[0]['name2idx'][key] ]
            batch = [[], []]
            for i, data in enumerate(batch_data):
                for j, data_item in enumerate(data):
                    tmp_batch = self._utterance_feature_augmentation(data_item, [0], self.context_window[i][j])
                    defaultTensor = self._get_default_tensor(tmp_batch)
                    batch[i].append(convertUttsList2Tensor(tmp_batch, defaultTensor, [length], padding_value=self.padding_value))
            yield batch, [length], [key]


    def eval_multi_utts(self, batch_size=2):
        for i in range(0, len(self.all_keys), batch_size):
            batch_keys = self.all_keys[i:i+batch_size]
            batch_data = self._get_block_data_from_keys(batch_keys, frame_mode=False)
            batch_idxs = [i for i in range(len(batch_keys))]
            lengths = [ self.dataset.features[0]['nframes'][ self.dataset.features[0]['name2idx'][key] ] for key in batch_keys ]
            sorted_lengths, order = torch.sort(torch.IntTensor(lengths), 0, descending=True)
            batch = [[], []]
            for i, data in enumerate(batch_data):
                for j, data_item in enumerate(data):
                    tmp_batch = self._utterance_feature_augmentation(data_item, batch_idxs, self.context_window[i][j])
                    defaultTensor = self._get_default_tensor(tmp_batch)
                    batch[i].append(convertUttsList2Tensor(tmp_batch, defaultTensor, lengths, padding_value=self.padding_value)[order])
            yield batch, list(sorted_lengths), batch_keys


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


class DataLoader(DataLoader):
    """
     Data loader. Combines a dataset and provides
     single- or multi-process iterators over the dataset.
     :param dataset (Dataset): dataset from which to load the data.
     :param batch_size (int, optional): how many samples per batch to load (default: 1). In frame_mode,
            batch_size means number of samples, otherwise, it means number of utterances.
     :param num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
     :param collate_fn (callable, optional)
     :param pin_memory (bool, optional)
     :param drop_last (bool, optional): set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
     :param frame_mode(bool, optional): set to ``False`` to enable (utterance) sequence
     :param permutation(bool, optional): whether permutate the indices in the iterator
     :param random_size(int, optional): defines the number of frames to load for once
     :param epoch_size(int, optional): defines the number of frames for each epoch
     :param truncate_size(int, optional): defines the truncate length in RNN
     :param utt_iter_idx(int, optional): defines the beginning utterance index to be loaded
     :param random_seed(int, optional): defines the random seed
     :param padding_value(int, optional): defines the value to be padded in sequence mode
    """
    def __init__(self, dataset, batch_size=1, num_workers=0,
                 collate_fn=default_collate, pin_memory=False, drop_last=False,
                 frame_mode=False, permutation=True, random_size=None,
                 epoch_size=None, truncate_size=0, utt_iter_idx=0,
                 random_seed=19931225, padding_value=0):
        self.logger = logging.getLogger('DataLoader')
        self.logger.info('DataLoader initialization')

        self.dataset = dataset
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last  = drop_last
        self.frame_mode  = frame_mode
        self.utt_iter_idx = utt_iter_idx
        self.random_seed = random_seed
        self.permutation = permutation
        self.padding_value = padding_value

        self.batch_size  = batch_size
        self.random_size = random_size if random_size else self.dataset.features[0]['total_nframes']
        self.epoch_size  = epoch_size if epoch_size else self.random_size
        self.truncate_size = truncate_size



    def __iter__(self):
        return DataLoaderIter(self)


    def __len__(self):
        """Epoch size."""
        if self.drop_last:
            return self.dataset.features[0]['total_nframes'] // self.epoch_size
        else:
            return (self.dataset.features[0]['total_nframes'] + self.epoch_size - 1) // self.epoch_size
