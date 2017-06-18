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
    def __init__(self, feat_config_parms, label_config_parms):
        """
         [feat,label]config_parms (dict or dict_list):
             file_name (string): Path to scp file or mlf file
             type("MLF" or "SCP"),
             SCP type: dim(int), context_window(tuple), max_utt_len(int),
             MLF type: dim(int), label_mapping(file_path), type("category")
         If only one argument is given for feat or label, then the output is an arg
         instead of a list.
        """
        feat_list_in = True
        if not (type(feat_config_parms) is list):
            feat_list_in = False
            feat_config_parms = [feat_config_parms]
        label_list_in = True
        if not (type(label_config_parms) is list):
            label_list_in = False
            label_config_parms = [label_config_parms]

        data_prefix      = ['feat', 'label']
        base_attributes  = ['_dim', '_data_type', '_type', 's', '_nUtts', '_nframes',
                            '_max_utt_len', '_context_window', '_label_mapping',
                            '_list']
        feat_attributes  = ['feat'  + x for x in base_attributes]
        label_attributes = ['label' + x for x in base_attributes]
        cur_attr_name    = ['cur'   + x for x in base_attributes]
        cur_attr_dict    = {}

        config_parms = [feat_config_parms, label_config_parms]
        attributes   = [feat_attributes,   label_attributes]
        list_in      = [feat_list_in,      label_list_in]

        for attr_idx in range(len(attributes)):
            cur_config_parms = config_parms[attr_idx]
            cur_attributes   = attributes[attr_idx]
            cur_prefix       = data_prefix[attr_idx]
            data_cnt         = len(cur_config_parms)
            for i in range(len(cur_attributes)):
                ''' Set the attribute to be a list of None [None, None, ...]'''
                setattr(self, cur_prefix + base_attributes[i], [None]*data_cnt)
                cur_attr_dict['cur' + base_attributes[i] ] = [None] * data_cnt

            for i in range(data_cnt):
                file_name = cur_config_parms[i]['file_name']

                cur_attr_dict['cur_dim'][i] = cur_config_parms[i]['dim']
                cur_attr_dict['cur_data_type'][i] = cur_config_parms[i]['type']
                if cur_attr_dict['cur_data_type'][i] == "MLF":
                    cur_attr_dict['cur_type'][i], cur_attr_dict['curs'][i], cur_attr_dict['cur_nUtts'][i], cur_attr_dict['cur_nframes'][i], cur_attr_dict['cur_label_mapping'][i] = self.read_mlf(file_name, cur_config_parms[i])
                elif cur_attr_dict['cur_data_type'][i] == "SCP":
                    cur_attr_dict['cur_context_window'][i], cur_attr_dict['curs'][i], cur_attr_dict['cur_list'][i], cur_attr_dict['cur_nframes'][i], cur_attr_dict['cur_nUtts'][i], cur_attr_dict['cur_max_utt_len'][i] = self.read_scp(file_name, cur_config_parms[i])

            for i in range(len(cur_attributes)):
                setattr(self, cur_attributes[i], cur_attr_dict[cur_attr_name[i]])

        for i in range(len(attributes)):
            if not list_in[i]:
                for attr in attributes[i]:
                    at = getattr(self, attr, None)
                    setattr(self, attr, at[0])

    def read_mlf(self, file_name, config_parms):
        """ Function for MLF data type.
              file_name(string),
              config_parms(dictionary)
        """
        label_mapping_path = config_parms['label_mapping']
        label_type = config_parms['label_type']
        code, (labels, nUtts, nframes) = self.read_htk_mlf(file_name)
        if (code > 0): raise Exception
        code, label_mapping = self.read_label_mapping(label_mapping_path)
        if (code > 0): raise Exception
        return label_type, labels, nUtts, nframes, label_mapping


    def read_scp(self, file_name, config_parms):
        """ Function for SCP data type.
              file_name(string),
              config_parms(dictionary)
        """
        context_window = config_parms['context_window'] if 'context_window' in config_parms else (0, 0)
        max_utt_len = config_parms['max_utt_len'] if 'max_utt_len' in config_parms else None
        code, (feats, feats_list, nframes) = self.read_htk_feats_scp(file_name, max_utt_len)
        nUtts = len(feats_list)
        return context_window, feats, feats_list, nframes, nUtts, max_utt_len


    def read_htk_feats_scp(self, file_name=None, max_utt_len=None):
        """
         Read the scp files in htk format: xxxxxxx.feats=/path/to/xxxxxx.feats[start_position, end_position]
         Output: code, feats(dictionary, ('name(xxxxxx)':['path(/path/to/xxxxxx.feats)', length] ) ), feats_list(list, all feats names), nframes(int, num of frames in total)
                code: 0: Success
                      1: File path does not exist
                      2: Format has problem
        """
        if (file_name == None or not os.path.exists(file_name)):
            """ raise FileNotExistsError """
            return 1, ({}, [], 0)
        feats = {}
        feats_list = []
        nframes = 0
        with open(file_name) as file:
            while 1:
                line = file.readline().strip()
                if (line==''): break
                feat_name, res_info = line.split('=')
                feat_name = ".".join(feat_name.split(".")[:-1])
                l_bracket_p = res_info.find('[')
                if l_bracket_p is None:
                    """ raise FileFormatError """
                    print("  Error: file format is not compatible %s" % line)
                    return 2, ({}, [], 0)
                feat_path = res_info[:l_bracket_p]
                res_info = res_info[l_bracket_p+1:]
                comma_p = res_info.find(',')
                start_position = int(res_info[:comma_p])
                end_position   = int(res_info[comma_p+1:-1])
                length = end_position - start_position + 1
                if (max_utt_len != None and length > max_utt_len): continue     # Omit the utterances that exceed the maximum length limit

                feats_list.append(feat_name)
                feats[feat_name] = [feat_path, length]
                nframes += length

        return 0, (feats, feats_list, nframes)


    def read_htk_mlf(self, file_name=None):
        """
         Read the MLF file in htk format: #!MLF!# \n "xxxxxx.lab" \n start end lab \n ...
         Output: code, labels(dictionary, ('name(xxxxxx)': numpy_array_lab)), nUtts(int), nframes(int)
        """
        if (file_name == None or not os.path.exists(file_name)):
            """ raise FileNotExistsError """
            return 1, ({}, 0, 0)
        labels = {}
        nUtts = 0
        nframes = 0

        start_mlf = False
        with open(file_name) as file:
            while 1:
                line = file.readline().strip()
                if (line==''): break
                if (start_mlf == False):
                    if (line=="#!MLF!#"):
                        start_mlf = True
                    else:
                        """ raise FileFormatError """
                        print("  Error: file format is not compatible in label file")
                        return 2, ({}, 0, 0)
                if (re.match('^\"[\.a-zA-Z0-9-_]+\.lab\"$', line)):   #label Name
                    nUtts += 1
                    feat_name = line[1:-4]
                    label_list = []
                    last_time_stamp = 0
                    label_complete = False
                elif (re.match('^[\d]+ [\d]+ [\d]+$', line)):     # labels
                    lst = line.split(' ')
                    start_position = int(lst[0]) // 100000
                    end_position   = int(lst[1]) // 100000
                    nframes += end_position - start_position
                    label          = int(lst[2])
                    if (start_position != last_time_stamp):
                        """ raise FileFormatError """
                        print("  Error: file format is not compatible in label file, missing labels")
                        return 2, ({}, 0, 0)
                    for i in range(start_position, end_position):
                        label_list.append(label)
                    last_time_stamp = end_position
                elif (re.match('^\.$', line)):      # End of current label
                    labels[feat_name] = np.array(label_list)
                    label_complete = True
        if not label_complete:
            """ raise FileFormatError """
            print("  Error: file format is not compatible in label file, incompleting labels")
            return 2, ({}, 0, 0)

        return 0, (labels, nUtts, nframes)


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
        return 0, label_mapping


def default_collate():
    pass


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
         max_utt_length(int, optional): defines the maximum utterance length to be loaded
         frame_mode(bool, optional): set to ``False`` to enable (utterance) sequence
         random_size(int, optional): defines the number of frames to load for once
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0,
                 collate_fn=default_collate, pin_memory=False, drop_last=False,
                 max_utt_length=None, frame_mode=False, random_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.max_utt_length = max_utt_length
        self.frame_mode = frame_mode
        self.random_size = random_size

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = torch_data.RandomSampler(dataset)
        elif not shuffle:
            self.sampler = torch_data.SequentialSampler(dataset)


    def __iter__(self):
        return torch_data.DataLoaderIter(self)


    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
