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
import numpy as np

class HTKDataset(object):
    """
     HTK Data set. Provides an interface for features of HTK format
    """
    def __init__(self, input_config_parms, target_config_parms, max_utt_len=0):
        """
         [input, target]config_parms (dict or dict_list):
             file_name (string): Path to SCP file or MLF file
             type("MLF" or "SCP"),
                SCP type: dim(int), context_window(tuple),
                MLF type: dim(int), label_mapping(file_path), type("category"). Only in label.
             If only one argument is given for feat or label, then the output is an arg
             instead of a list.
         :max_utt_len: int, max utterance length permitted, so we can drop utterances longer than it

         Note:
             1. multiple input SCP should have compatible keys and the same order is preferrable 
                but not necessary.
             2. Attributes:
                :dim:       int, dimension, usually 39,40,80 for input feature, and 4009 for target
                :data_type: string, MLF or SCP
                :type:      string, from CNTK, only for target, "category"
                :data:      feature path or label
                :nUtts:     int, number of utterances
                :nframes:   int list, number of frames for each utt
                :context_window: tuple, context window size
                :label_mapping: label_mapping, from int to label
                :name2idx:  dictionary, 
        """
        """ If reading configures are not lists, convert them to lists """
        input_list_in = True
        if not (type(input_config_parms) is list):
            input_list_in = False
            input_config_parms = [input_config_parms]
        target_list_in = True
        if not (type(target_config_parms) is list):
            target_list_in = False
            target_config_parms = [target_config_parms]

        """ return data """
        input = []
        target = []
        self.ninputs   = len(input_config_parms)      # number of input  data
        self.ntargets  = len(target_config_parms)     # number of target data
        self.max_utt_len = max_utt_len

        """ Define all possible attributes for data and labels """
        base_attributes  = ['dim', 'data_type', 'type', 'data', 'nUtts', 'nframes',
                            'context_window', 'label_mapping', 'name2idx']
        nattributes = len(base_attributes)

        """ For loop convenience """
        data         = [input, target]
        data_prefix  = ["input", "target"]
        list_in      = [input_list_in, target_list_in]
        config_parms = [input_config_parms, target_config_parms]

        for data_idx in range(len(data)):     # loop of 2, input or target
            cur_data         = data[data_idx]
            cur_config_parms = config_parms[data_idx]
            data_cnt         = getattr(self, "n{0}s".format(data_prefix[data_idx])) # ninputs/ntargets
            
            """ initialize to None """
            for i in range(data_cnt):
                cur_data.append({})
                for j in range(nattributes):
                    cur_data[-1][base_attributes[j]] = None

            for i in range(data_cnt):   # input or output index, when multiple input or output
                file_name = cur_config_parms[i]['file_name']    # SCP or MLF File

                cur_data[i]['dim'] = cur_config_parms[i]['dim']
                cur_data[i]['data_type'] = cur_config_parms[i]['type']

                if cur_data[i]['cur_data_type'] == "MLF":
                    cur_data[i]['type'] = cur_config_parms[i]['label_type']
                    (cur_data[i]['data'], cur_data[i]['name2idx'], cur_data[i]['nframes'], cur_data[i]['label_mapping'], cur_data[i]['nUtts']) = self.read_MLF(file_name, cur_config_parms[i])
                elif cur_attr_dict['cur_data_type'][i] == "SCP":
                    cur_data[i]['context_window'] = cur_config_parms[i]['context_window'] if 'context_window' in cur_config_parms[i] else (0, 0)
                    (cur_data[i]['data'], cur_data[i]['name2idx'], cur_data[i]['nframes'], cur_data[i]['nUtts']) = self.read_SCP(file_name, cur_config_parms[i])

        """ convert list to the first item, when list is not needed """
        for i in range(len(data)):
            if not list_in[i]:
                data[i] = data[i][0]

        return (input, target)

    def read_MLF(self, file_name, config_parms):
        """ Function for MLF data type.
              file_name(string),
              config_parms(dictionary)
        """
        (labels, name2idx, lab_nframes) = self.read_HTK_MLF(file_name)
        if 'context_window' in config_parms:
            label_mapping_path = config_parms['label_mapping']
            (label_mapping, _) = self.read_label_mapping(label_mapping_path)
        else:
            label_mapping = []
        return (labels, name2idx, lab_nframes, label_mapping, len(labels))


    def read_SCP(self, file_name):
        """ Function for SCP data type.
              file_name(string),
              config_parms(dictionary)
        """
        (feats, name2idx, feat_nframes) = self.read_HTK_feats_SCP(file_name)
        return (feats, name2idx, feat_nframes, len(feats))


    def read_HTK_feats_SCP(self, file_name=None, max_utt_len=None):
        """
         Read the SCP files in HTK format: xxxxxxx.feats=/path/to/xxxxxx.feats[start_position, end_position]
         Output: code, feats(dictionary, ('name(xxxxxx)':['path(/path/to/xxxxxx.feats)', length] ) ), feats_list(list, all feats names), feat_nframes(int list, num of frames of each utt)
                code: 0: Success
                      1: File path does not exist
                      2: Format has problem
        """
        if (file_name == None or not os.path.exists(file_name)):
            raise Exception("An error occur while reading SCP file(%s), code:%d, File Non-exists" % (file_name, 1))

        feats = []
        name2idx = {}
        feat_nframes = []

        with open(file_name) as file:
            while 1:
                line = file.readline().strip()
                if (line==''): break

                feat_name, res_info = line.split('=')
                feat_name = feat_name.split(".")[: feat_name.rfind('.')]
                l_bracket_pos = res_info.find('[')
                if l_bracket_pos is None:
                    """ raise FileFormatError """
                    raise Exception("An error occur while reading SCP file(%s), code:%d, File Format Error in: %s" % (file_name, 2, line))
                    return ([], {}, 0, 0)

                feat_path = res_info[:l_bracket_pos]
                res_info  = res_info[l_bracket_pos+1:]
                comma_pos = res_info.find(',')
                start_frame = int(res_info[:comma_pos])
                end_frame   = int(res_info[comma_pos+1:-1])
                length = end_position - start_position + 1
                if (max_utt_len != None and length > max_utt_len): continue     # Omit the utterances that exceed the maximum length limit

                feats.append(feat_name)
                feat_nframes.append(length)
                name2idx[feat_name] = len(feats) - 1

        return (feats, name2idx, feat_nframes)


    def read_HTK_MLF(self, file_name=None, delete_toolong=True):
        """
         Read the MLF file in HTK format: #!MLF!# \n "xxxxxx.lab" \n start end lab \n ...
         Output: labels(list, label list), nframes(int list)
         :file_name: HTK_MLF file name
         :delete_toolong: delete the utterance that exceeds the maximum utterance length
        """
        if (file_name == None or not os.path.exists(file_name)):
            """ raise FileNotExistsError """
            raise Exception("An error occur while reading MLF file(%s), code:%d, File Non-exists" % (file_name, 1))
            return ({}, 0, 0)
        labels = []
        name2idx = {}
        lab_nframes = []

        start_mlf = False   # flag of "#!MLF!#"
        with open(file_name) as file:
            while 1:
                line = file.readline().strip()
                if (line==''): break
                if (start_mlf == False):
                    if (line=="#!MLF!#"):
                        start_mlf = True
                        lab_complete = True
                    else:
                        """ raise FileFormatError """
                        raise Exception("  An error occur while reading MLF file(%s), code:%d, File Format Error: no #!MLF!# found" % (file_name, 2))
                        return ({}, 0, 0)
                if (re.match('^\"[\.a-zA-Z0-9-_]+\.lab\"$', line)):   #label Name
                    if (not lab_complete):
                        raise Exception("  An error occur while reading MLF file(%s), code:%d, File Format Error, incomplete label" % (file_name, 2))
                        return ({}, 0, 0)
                    lab_length = 0
                    lab_complete = False
                    lab_name = line[1 : line.rfind('.lab')]
                    name2idx[lab_name] = len(labels)
                    labels.append([])
                elif (re.match('^[\d]+ [\d]+ [\d]+$', line)):     # labels
                    lst = line.split(' ')
                    start_pos  = int(lst[0]) // 100000
                    end_pos    = int(lst[1]) // 100000
                    lab_length = end_position
                    label_item = int(lst[2])
                    if (start_pos != lab_length):
                        """ raise FileFormatError """
                        raise Exception("An error occur while reading SCP file(%s), code:%d, File Format Error: missing labels" % (file_name, 2))
                        return ({}, 0, 0)
                    for i in range(start_pos, end_pos):
                        labels[-1].append(label)
                elif (re.match('^\.$', line)):      # End of current label
                    label_complete = True
                    if (self.max_utt_len > 0 and delete_toolong):   # delete the label if it exceed the maximum length
                        del name2idx[lab_name]
                        del labels[-1]
                    else:
                        lab_nframes.append(lab_length)
                else:
                    raise Exception("An error occur while reading SCP file(%s), code:%d, File Format Error: unknown format: %s" % (file_name, 2, line))
                    return ({}, 0, 0)

        if not label_complete:
            """ raise FileFormatError """
            raise Exception("  An error occur while reading MLF file(%s), code:%d, File Format Error, incomplete label" % (file_name, 2))
            return ({}, 0, 0)

        return (labels, name2idx, lab_nframes)


    def read_label_mapping(self, file_name=None):
        """
         Label mapping file: lists all the possible label values, one per line,
         which might be text or numeric. The line number is the identifier that will
         be used by CNTK to identify that label. This parameter is often moved to the
         root level to share with other Command Sections. For example, it’s important
         that the Evaluation Command Section share the same label mapping as the
         trainer, otherwise, the evaluation results will not be accurate.
        """
        if (file_name == None or not os.path.exists(file_name)):
            """ raise FileNotExistsError """
            raise Exception("An error occur while reading label mapping file(%s), code:%d, File Non-exists" % (file_name, 1))
            return 1, (None)
        label_mapping = []
        label_type_int = True
        with open(file_name, 'r') as file:
            while 1:
                line = file.readline().strip()
                if (line == ''): break
                if not re.match('\d+', line):
                    label_type_int = False
                label_mapping.append(line)
        if (label_type_int):
            label_mapping = [int(x) for x in label_mapping]
        return (label_mapping, label_type_int)


def default_collate():
    pass


class HTKDataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.collate_fn = loader.collate_fn
        self.sampler = loader.sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.drop_last = loader.drop_last
        self.done_event = threading.Event()

        self.samples_remaining = len(self.sampler)
        self.sample_iter = iter(self.sampler)
        self.cur_mb_index = loader


class HTKDataLoader(DataLoader):
    """
     HTK Data loader. Combines a dataset and a sampler, and provides
     single- or multi-process iterators over the dataset.
     Arguments:
         dataset (Dataset): dataset from which to load the data.
         batch_size (int, optional): how many samples per batch to load
             (default: 1).
         shuffle (bool, optional): set to ``True`` to have the data reshuffled
             at every epoch (default: False).
         sampler (Sampler, optional): defines the strategy to draw samples from
             the dataset. If specified, the ``shuffle`` argument is ignored.
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
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0,
                 collate_fn=default_collate, pin_memory=False, drop_last=False,
                 frame_mode=False, random_size=None, epoch_size=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.epoch_size = epoch_size
        self.frame_mode = frame_mode
        self.random_size = random_size

        self.feat_epoch = []     # [epoch, input_cnt]={}
        self.label_epoch = []     # [epoch, input_cnt]={}
        self.SplitEpoch()

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = torch_data.RandomSampler(dataset)
        elif not shuffle:
            self.sampler = torch_data.SequentialSampler(dataset)


    def __iter__(self):
        return HTKDataLoaderIter(self)


    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def SplitEpoch(self):
        pass