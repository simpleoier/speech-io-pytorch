#!/usr/bin/env python
# encoding: utf-8

import torch

class Sampler(object):
    """ Base class for all Samplers.
     Every Sampler subclass has to provide an __iter__ method, providing a way
     to iterate over indices of dataset elements, and a __len__ method that returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class FrameLevSampler(Sampler):
    "Samples elements in frame mode sequentially."
    def __init__(self, data_source):
        self.num_samples = data_source.inputs[0]['total_nframes']

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples

class UtteranceLevSampler(Sampler):
    "Samples elements in utterance mode sequentially."
    def __init__(self, data_source):
        self.num_samples = data_source.inputs[0]['nUtts']

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples

class BatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.
    
     Args:
        sampler (Sampler): Base sampler
        batch_size (int): Size of mini-batch. Number of frames in each mini-batch or number of utterances in each mini-batch.
        drop_last (bool): If ``True'', the sampler will drop the last batch if its size would be less than ``batch_size''.
    """
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size