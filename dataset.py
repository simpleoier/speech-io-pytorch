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
from Kaldi_IO import read_ali_ark, write_vec_int, read_vec_flt_ark, write_vec_flt, read_mat_scp, write_mat
from sampler import FrameLevSampler, UtteranceLevSampler
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)


class HTKDataset(object):
    """
     HTK Data set. Provides an interface for features of HTK format
    """
    def __init__(self, input_config_parms, target_config_parms=None,
                 max_utt_len=0, verify_length=True, logger=None):
        """ Initialize parameters.
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
                :start_f:   int list, the start frame point in the feature file
                :nframes:   int list, number of frames for each utt
                :context_window: tuple, context window size
                :label_mapping: label_mapping, from int to label
                :name2idx:  dictionary,
        """
        """ If reading configures are not lists, convert them to lists """
        logging.basicConfig(format='[%(name)s] %(levelname)s(%(asctime)s): %(message)s', datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)
        self.logger = logging.getLogger('HTKDataset') if logger is None else logger

        input_config_parms = input_config_parms if isinstance(input_config_parms, list) else [input_config_parms]
        target_config_parms = target_config_parms if isinstance(target_config_parms, list) else [target_config_parms]

        self.max_utt_len = max_utt_len
        self.max_utt_len2 = 0
        self.first_input = True
        self.all_keys = None

        # number of inputs or targets data
        self.ninputs  = len(input_config_parms) if input_config_parms else 0
        self.ntargets = len(target_config_parms) if target_config_parms else 0
        self.inputs   = [] if self.ninputs>0 else None
        self.targets  = [] if self.ntargets>0 else None

        # Define all possible attributes for data and labels
        self.base_attributes = ['dim', 'data_type', 'type', 'data', 'nUtts',
                                'start_f', 'nframes', 'context_window',
                                'label_mapping', 'name2idx', 'total_nframes',]

        self._get_all_data(self.inputs,  input_config_parms)
        self._get_all_data(self.targets, target_config_parms)
        self._verify_data(verify_length)
        self.logger.info("HTKDataset initialization close.")


    def _verify_data(self, verify_length):
        """ Verify all keys have labels. If not, delete the key. """
        data_list = [self.inputs, self.targets]
        data_description = ['data', 'label']

        for data in data_list:
            for data_item in data:
                self.all_keys = list(filter(lambda x: x in data_item['name2idx'].keys(), self.all_keys))

        all_keys_set = set(self.all_keys)
        for d in range(len(data_list)):
            data = data_list[d]
            for i in range(len(data)):
                data_item = data[i]

                empty_keys_set = set(data_item['name2idx'].keys()) - all_keys_set
                empty_keys_list = list(filter(lambda x: x in empty_keys_set, data_item['name2idx'].keys()))

                not_found_info = ["[reject {} {} for incompleteness]".format('input', empty_keys_list[i]) for i in range(min(4, len(empty_keys_list)))]
                if (len(empty_keys_list)>0):
                    data_item['nUtts'] -= len(empty_keys_list)
                    for key in empty_keys_list:
                        key2idx0 = data_item['name2idx'][key]
                        data_item['total_nframes'] -= data_item['nframes'][key2idx0]
                    #self.logger.warning("{0}{1} {2} files in {3} {4} not found in other data or labels".format(' '.join(not_found_info), '.'*50, len(empty_keys_list), data_description[d], i))
                    self.logger.warning("{0}{1} {2} frames in {3} out of {4} utterances, {5} files of {6} {7} not found in other data or labels".format(' '.join(not_found_info), '.'*50, data_item['total_nframes'], data_item['nUtts'], data_item['nUtts']+len(empty_keys_list), len(empty_keys_list), data_description[d], i))

        """ Verify Lengths """
        data_list = [i for i in self.inputs] + [t for t in self.targets]
        if not verify_length or len(data_list)==0: return
        for key in self.all_keys:
            key2idx0 = data_list[0]['name2idx'][key]
            standard_len = data_list[0]['nframes'][key2idx0]
            for j in range(1, len(data_list)):
                key2idx0 = data_list[j]['name2idx'][key]
                if (standard_len != data_list[j]['nframes'][key2idx0]):
                    self.logger.error("Inconsistent length in input&label {0} utterance {1} {2} vs. {3} input0".format(j, key, data_list[j]['nframes'][key2idx0], standard_len))


    def _get_all_data(self, data, config_parms):
        """ Read inputs and targets."""
        ndata = len(config_parms) if config_parms[0] else 0

        for i in range(ndata):
            data.append({})
            for j in range(len(self.base_attributes)):
                data[-1][self.base_attributes[j]] = None     #initialize to None

        for i in range(ndata):
            file_name = config_parms[i]['file_name']    # SCP or MLF File
            data[i]['dim'] = config_parms[i]['dim']
            data[i]['data_type'] = config_parms[i]['type']
            data[i]['context_window'] = config_parms[i]['context_window'] if 'context_window' in config_parms[i] else None

            if data[i]['data_type'] == "MLF":
                data[i]['type'] = config_parms[i]['label_type']
                self._read_MLF(file_name, config_parms[i], data[i])
            elif data[i]['data_type'] == "SCP":
                self._read_SCP(file_name, data[i])

            if self.first_input:
                self.first_input = False
                self.all_keys = list(data[i]['name2idx'])

            self.logger.info("Dataset %d: %d frames in %d utterances" % (i, data[i]['total_nframes'], data[i]['nUtts']))


    def _read_SCP(self, file_name, data):
        """ Function for SCP data type.
              file_name(string),
              data(dictionary)
        """
        (feats, name2idx, feat_start_f, feat_nframes) = self._read_HTK_feats_SCP(file_name)
        total_nframes = 0
        for i in range(len(feat_nframes)): total_nframes += feat_nframes[i]
        self.logger.info("Reading script file %s ... %d entries." % (file_name, len(feats)))
        data['data']    = feats;        data['name2idx']      = name2idx
        data['start_f'] = feat_start_f; data['nframes']       = feat_nframes
        data['nUtts']   = len(feats);   data['total_nframes'] = total_nframes


    def _read_HTK_feats_SCP(self, file_name=None):
        """
         Read the SCP files in HTK format: xxxxxxx.feats=/path/to/xxxxxx.feats[start_position, end_position]
         Output: code, feats(dictionary, ('name(xxxxxx)':['path(/path/to/xxxxxx.feats)', length] ) ), feats_list(list, all feats names), feat_nframes(int list, num of frames of each utt)
                code: 0: Success
                      1: File path does not exist
                      2: Format has problem
        """
        if (file_name == None or not os.path.exists(file_name)):
            message = "An error occur while reading SCP file({0}), code:{1}, File Non-exists".format(file_name, 1)
            raise Exception(message)

        feats = []
        name2idx = {}
        feat_start_f = []
        feat_nframes = []
        linecnt = 0
        skip_cnt = 0

        with open(file_name, 'r') as file:
            while 1:
                line = file.readline().strip()
                linecnt += 1
                if (line==''): break

                feat_name, res_info = line.split('=')[0:2]
                feat_name = feat_name[: feat_name.rfind('.')]
                l_bracket_pos = res_info.find('[')
                if l_bracket_pos is None:
                    """ raise FileFormatError """
                    raise Exception("An error occur while reading SCP file(%s), code:%d, File Format Error in: %s" % (file_name, 2, line))

                feat_path = res_info[:l_bracket_pos]
                res_info  = res_info[l_bracket_pos+1:]
                comma_pos = res_info.find(',')
                start_frame = int(res_info[:comma_pos])
                end_frame   = int(res_info[comma_pos+1:-1])
                length = end_frame - start_frame + 1

                # Omit the utterances that exceed the maximum length limit
                if (self.max_utt_len != 0 and length > self.max_utt_len):
                    if self.first_input:
                        #self.logger.warning("minibatchutterancesource: skipping %d-th file (%d frames) because it exceeds maxUtteranceLength (%d frames)" % (linecnt, length, self.max_utt_len) )
                        skip_cnt += 1
                    continue
                if (length > self.max_utt_len2): self.max_utt_len2 = length
                if (not self.first_input) and (not feat_name in self.all_keys):
                    continue
                feats.append(feat_path)
                feat_start_f.append(start_frame)
                feat_nframes.append(length)
                if feat_name not in name2idx:
                    name2idx[feat_name] = len(feats) - 1
                else:
                    raise Exception("An error occur while reading SCP file(%s), code:%d, duplicate input name: %s" % (file_name, 2, feat_name))

        if self.first_input and skip_cnt > 0:
            self.logger.warning("minibatchutterancesource: skipping %d files because exceeding maxUtteranceLength (%d frames)" % (skip_cnt, self.max_utt_len) )
        return (feats, name2idx, feat_start_f, feat_nframes)


    def _read_MLF(self, file_name, config_parms, data):
        """ Function for MLF data type.
              file_name(string),
              config_parms(dictionary)
              data(dictionary)
        """
        (labels, name2idx, lab_nframes) = self._read_HTK_MLF(file_name)
        if 'label_mapping' in config_parms:
            label_mapping_path = config_parms['label_mapping']
            (label_mapping, _) = self._read_label_mapping(label_mapping_path)
            self.logger.info("Total %d state names in state list %s" % (len(label_mapping), label_mapping_path))
        else:
            label_mapping = None
        total_nframes = 0
        for i in range(len(lab_nframes)): total_nframes += lab_nframes[i]
        self.logger.info("Reading MLF file %s ... total %d entries." % (file_name, len(labels)))
        data['data']    = labels;       data['name2idx']      = name2idx
        data['nframes'] = lab_nframes;  data['label_mapping'] = label_mapping
        data['nUtts']   = len(labels);  data['total_nframes'] = total_nframes


    def _read_HTK_MLF(self, file_name=None, delete_toolong=True):
        """ Read the MLF file in HTK format: #!MLF!# \n "xxxxxx.lab" \n start end lab \n ...
         Output: labels(list, label list), nframes(int list)
         :file_name: HTK_MLF file name
         :delete_toolong: delete the utterance that exceeds the maximum utterance length
        """
        if (file_name == None or not os.path.exists(file_name)):
            message = "An error occur while reading MLF file({0}), code:{1}, File Non-exists".format(file_name, 1)
            raise Exception(message)

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
                        lab_name = ''
                        continue
                    else:
                        raise Exception("  An error occur while reading MLF file(%s), code:%d, File Format Error: no #!MLF!# found" % (file_name, 2))

                if (re.match('^\"[\.a-zA-Z0-9-_]+\.lab\"$', line)):   #label Name
                    if (not lab_complete):
                        raise Exception("  An error occur while reading MLF file(%s), code:%d, File Format Error, incomplete label %s" % (file_name, 2, lab_name))

                    lab_length = 0
                    lab_complete = False
                    lab_name = line[1 : line.rfind('.lab')]
                    name2idx[lab_name] = len(labels)
                    labels.append([])
                elif (re.match('^[\d]+ [\d]+ [\d]+$', line)):     # label Item
                    lst = line.split(' ')
                    start_pos  = int(lst[0]) // 100000
                    end_pos    = int(lst[1]) // 100000
                    label_item = int(lst[2])
                    if (start_pos != lab_length):
                        raise Exception("An error occur while reading SCP file(%s), code:%d, File Format Error: missing labels for %s" % (file_name, 2, lab_name))

                    lab_length = end_pos
                    for i in range(start_pos, end_pos):
                        labels[-1].append(label_item)
                elif (re.match('^\.$', line)):      # End of current label
                    lab_complete = True
                    if (self.max_utt_len > 0 and delete_toolong and lab_length > self.max_utt_len):
                        # delete the label if it exceed the maximum length
                        del name2idx[lab_name]
                        del labels[-1]
                    else:
                        lab_nframes.append(lab_length)
                else:
                    raise Exception("An error occur while reading SCP file(%s), code:%d, File Format Error: unknown format: %s" % (file_name, 2, line))

        if not lab_complete:
            raise Exception("An error occur while reading MLF file(%s), code:%d, File Format Error, incomplete label" % (file_name, 2))

        return (labels, name2idx, lab_nframes)


    def _read_label_mapping(self, file_name=None):
        """ Read label mapping file.
         Label mapping file: lists all the possible label values, one per line,
         which might be text or numeric. The line number is the identifier that will
         be used by CNTK to identify that label. This parameter is often moved to the
         root level to share with other Command Sections. For example, it’s important
         that the Evaluation Command Section share the same label mapping as the
         trainer, otherwise, the evaluation results will not be accurate.
        """
        if (file_name == None or not os.path.exists(file_name)):
            raise Exception("An error occur while reading label mapping file(%s), code:%d, File Non-exists" % (file_name, 1))

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
