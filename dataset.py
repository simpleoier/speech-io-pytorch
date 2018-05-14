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
import json
import numpy as np
from HTK_IO import HTKFeat_read, HTKFeat_write
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)

logging.basicConfig(format='[%(name)s] %(levelname)s(%(asctime)s): %(message)s', datefmt="%m-%d %H:%M:%S", level=logging.DEBUG)


def new_data_item(config_param):
    """
     Note:
        :dim:       int, dimension, such as 40-dimensional fbank, or 4009 dimensional targets
        :data_type: string, MLF or SCP or JSON
        :type:      string, from CNTK, only for target, "category"
        :data:      list, feature path or label data
        :nUtts:     int, number of utterances
        :format:    HTK, Kaldi, JSON
        :start_f:   int list, the start frame point in the feature file
        :nframes:   int list, number of frames for each utterance
        :context_window: tuple, context window size
        :label_mapping: label_mapping, from int to label
        :name2idx:  dictionary,
    """
    # Define all possible attributes for data and labels
    base_attributes = ['dim', 'data_type', 'data', 'nUtts', 'format'
                    'start_f', 'nframes', 'context_window',
                    'label_mapping', 'name2idx', 'total_nframes']
    item_dict = {}
    for key in base_attributes: item_dict[key] = None
    item_dict['dim'] = config_param['dim']
    item_dict['data_type'] = config_param['type']
    if 'format' in config_param:
        item_dict['format'] = config_param['format']
    if 'context_window' in config_param:
        item_dict['context_window'] = config_param['context_window']
    return item_dict


class Dataset(object):
    """
     Data set. Provides an interface for features of HTK format and Kaldi format.
     Input features are preferred in both HTK and Kaldi format.
     Targets (labels) are preferred in JSON format.
    """
    def __init__(self, feature_config_parms=None, target_config_parms=None,
                 max_utt_len=0, verify_length=True, logger=None):
        """ Initialize parameters.
         :param feature_config_params, target]config_parms:
            dict or dict_list including:
                file_name (string): Path to SCP file or MLF file
                type("MLF" or "SCP"),
                    SCP type: dim(int), context_window(tuple),
                    MLF type: dim(int), label_mapping(file_path), type("category"). Only in label.
         :param max_utt_len:
            int, max utterance length permitted, so we can drop utterances longer than it
         :param verify_length:
            verify whether the features have the same lengths with the corresponding targets (False for Seq2Seq models)
         :param logger:
            logger
         :return:
            features, targets, all_keys
         """

        self.logger = logger if logger else logging.getLogger('Dataset')
        self.logger.info("Dataset initialization")

        """ If reading configures are not lists, convert them to lists """
        feature_config_parms = feature_config_parms if isinstance(feature_config_parms, list) else [feature_config_parms]
        target_config_parms  = target_config_parms  if isinstance(target_config_parms,  list) else [target_config_parms]

        self.max_utt_len = max_utt_len
        self.features = self._get_all_data(feature_config_parms)
        self.targets  = self._get_all_data(target_config_parms)
        self.all_keys = self._verify_data(verify_length)


    def _verify_data(self, verify_length):
        """ Verify all keys have labels. If not, delete the key. """

        self.logger.info("Dataset verifying data")

        data_set = [self.features, self.targets]
        data_description = ['feature', 'target']

        # get the intersection of all the keys of dataset
        all_keys = data_set[0][0]['name2idx'].keys()
        for data in data_set:
            for item in data:
                all_keys = list(filter(lambda x: x in item['name2idx'].keys(), all_keys))

        # feedback the compatible information of keys in each feature and target item
        all_keys_set = set(all_keys)
        for d, data in enumerate(data_set):
            for i, item in enumerate(data):
                empty_keys_list = list( set( item['name2idx'].keys() ) - all_keys_set )

                if (len(empty_keys_list)>0):
                    item['nUtts'] -= len(empty_keys_list)
                    for key in empty_keys_list:
                        key2idx0 = item['name2idx'][key]
                        item['total_nframes'] -= item['nframes'][key2idx0]

                    not_found_info = empty_keys_list[:4] + ['.'*50]
                    self.logger.warning("reject\n{}\n\t total {} frames in {} out of {} utterances, {} files of {}[{}] not found in other data or labels".format('\n'.join(not_found_info), item['total_nframes'], item['nUtts'], item['nUtts']+len(empty_keys_list), len(empty_keys_list), data_description[d], i))

        """ Verify Lengths """
        data_set = self.features + self.targets
        if not verify_length:
            return all_keys

        for key in all_keys:
            key2idx0 = data_set[0]['name2idx'][key]
            standard_len = data_set[0]['nframes'][key2idx0]
            for j in range(1, len(data_set)):
                key2idx0 = data_set[j]['name2idx'][key]
                if (standard_len != data_set[j]['nframes'][key2idx0]):
                    self.logger.error("Inconsistent length in input&label {} utterance {} {} vs. {} input0".format(j, key, data_set[j]['nframes'][key2idx0], standard_len))
        return all_keys


    def _get_all_data(self, config_params):
        """ Read inputs and targets."""
        data = []
        for i, config in enumerate(config_params):
            data_item = new_data_item(config)

            file_name = config['file_name']

            if data_item['data_type'] == "SCP":
                self._read_SCP(file_name, data_item)
            elif data_item['data_type'] == "MLF":
                data_item['type'] = config['label_type']
                self._read_MLF(file_name, config, data_item)
            elif data_item['data_type'] == 'JSON':
                self._read_JSON(file_name, config, data_item)

            data.append(data_item)
            self.logger.info("Dataset %d: %d frames in %d utterances" % (i, data[i]['total_nframes'], data[i]['nUtts']))

        return data


    def _read_SCP(self, file_name, data):
        """ Function for SCP data type.
              file_name(string),
              data(dictionary)
        """
        if data['format'] == 'HTK':
            (feats, name2idx, feat_start_f, feat_nframes) = self._read_HTK_feats_SCP(file_name)
        elif data['format'] == 'Kaldi':
            (feats, name2idx, feat_start_f, feat_nframes) = self._read_Kaldi_feats_SCP(file_name)
        data['data']    = feats;        data['name2idx']      = name2idx
        data['start_f'] = feat_start_f; data['nframes']       = feat_nframes
        data['nUtts']   = len(feats);   data['total_nframes'] = sum(feat_nframes)
        self.logger.info("Reading script file %s ... %d entries." % (file_name, len(feats)))


    def _read_HTK_feats_SCP(self, scp_file_name=None):
        """
         Read the SCP files in HTK format: xxxxxxx.feats=/path/to/xxxxxx.feats[start_position, end_position]
         :return feats, name2idx, feat_start_f, feat_nframes:
             list, dictionary, list, list
        """
        if (scp_file_name == None or not os.path.exists(scp_file_name)):
            raise Exception("FileNonExistError: an error occur while reading SCP file({0})".format(scp_file_name))

        feats = []
        name2idx = {}
        feat_start_f = []
        feat_nframes = []
        skip_cnt = 0    # number of utterances exceeding the max utterance length

        with open(scp_file_name, 'r') as file:
            line = file.readline().strip()
            while line:
                feat_name, rest_info = line.split('=')[0:2]  # xxxxxx.feats, /path/to/xxxxxx.feats[start_f, end_f]
                feat_name = feat_name[: feat_name.rfind('.')]   # xxxxxx
                l_bracket_pos = rest_info.find('[')
                if l_bracket_pos is None:
                    raise Exception("FormatError: an error occur while reading SCP file(%s), File Format Error in: %s" % (scp_file_name, line))

                feat_path = rest_info[:l_bracket_pos]
                rest_info = rest_info[l_bracket_pos+1:]
                comma_pos = rest_info.find(',')
                start_frame = int(rest_info[:comma_pos])
                end_frame   = int(rest_info[comma_pos+1:-1])
                length = end_frame - start_frame + 1

                # read next line
                line = file.readline().strip()

                # Omit the utterances that exceed the maximum length limit
                if (self.max_utt_len != 0 and length > self.max_utt_len):
                    skip_cnt += 1
                    continue

                feats.append(feat_path)
                feat_start_f.append(start_frame)
                feat_nframes.append(length)
                if feat_name not in name2idx:
                    name2idx[feat_name] = len(feats) - 1
                else:
                    raise Exception("FormatError: an error occur while reading HTK SCP file(%s), duplicate input name: %s" % (scp_file_name, feat_name))

        if skip_cnt > 0:
            self.logger.info("minibatchutterancesource: skipping %d files because exceeding maxUtteranceLength (%d frames)" % (skip_cnt, self.max_utt_len) )
        return (feats, name2idx, feat_start_f, feat_nframes)


    def _read_Kaldi_feats_SCP(self, scp_file_name=None, feat_to_len_name=None):
        """
         Read the SCP files in Kaldi format: xxxxxxx /path/to/xxxxxx.ark:start_position
         Read the feat_to_len file in Kaldi format: xxxxxx length
         :return feats, name2idx, feat_start_f, feat_nframes:
             list, dictionary, list, list
        """
        def read_feat_to_len_file(feat_to_len_name):
            feat_to_len = {}
            with open(feat_to_len_name, 'r') as file:
                line = file.readline().strip()
                while line:
                    feat_name, length = line.split(' ')[0:2]
                    feat_to_len[feat_name] = int(length)
                    line = file.readline().strip()
            return feat_to_len

        if (feat_to_len_name == None or not os.path.exists(feat_to_len_name)):
            raise Exception("FileNonExistError: an error occur while reading feat_to_len file({0})".format(feat_to_len_name))
        if (scp_file_name == None or not os.path.exists(scp_file_name)):
            raise Exception("FileNonExistError: an error occur while reading SCP file({0})".format(scp_file_name))

        feat_to_len = read_feat_to_len_file(feat_to_len_name)
        feats = []
        name2idx = {}
        feat_start_f = []
        feat_nframes = []
        skip_cnt = 0    # number of utterances exceeding the max utterance length

        with open(scp_file_name, 'r') as file:
            line = file.readline().strip()
            while line:
                feat_name, feat_path = line.split(' ')[0:2]  # xxxxxx.feats, /path/to/xxxxxx.ark:start_f
                start_frame = rest_info[feat_path.rfind(':')+1:]  # start_f
                length = feat_to_len[feat_name]

                # read next line
                line = file.readline().strip()

                # Omit the utterances that exceed the maximum length limit
                if (self.max_utt_len != 0 and length > self.max_utt_len):
                    skip_cnt += 1
                    continue

                feats.append(feat_path)
                feat_start_f.append(start_frame)
                feat_nframes.append(length)
                if feat_name not in name2idx:
                    name2idx[feat_name] = len(feats) - 1
                else:
                    raise Exception("FormatError: an error occur while reading Kaldi SCP file(%s), duplicate input name: %s" % (scp_file_name, str(2), feat_name))

        if skip_cnt > 0:
            self.logger.info("minibatchutterancesource: skipping %d files because exceeding maxUtteranceLength (%d frames)" % (skip_cnt, self.max_utt_len) )
        return (feats, name2idx, feat_start_f, feat_nframes)


    def _read_MLF(self, mlf_file_name, config_parms, data):
        """ Function for MLF data type.
              mlf_file_name(string),
              config_parms(dictionary)
              data(dictionary)
        """
        (labels, name2idx, lab_nframes) = self._read_HTK_MLF(mlf_file_name)
        if 'label_mapping' in config_parms:
            label_mapping_path = config_parms['label_mapping']
            (label_mapping, _) = self._read_label_mapping(label_mapping_path)
            self.logger.info("Total %d state names in state list %s" % (len(label_mapping), label_mapping_path))
        else:
            label_mapping = None
        data['data']    = labels;       data['name2idx']      = name2idx
        data['nframes'] = lab_nframes;  data['label_mapping'] = label_mapping
        data['nUtts']   = len(labels);  data['total_nframes'] = sum(lab_nframes)
        self.logger.info("Reading MLF file %s ... total %d entries." % (mlf_file_name, len(labels)))


    def _read_HTK_MLF(self, mlf_file_name=None, delete_toolong=True):
        """ Read the MLF file in HTK format: #!MLF!# \n "xxxxxx.lab" \n start end lab \n ...
         Output: labels(list, label list), nframes(int list)
         :mlf_file_name: HTK_MLF file name
         :delete_toolong: deete the utterance that exceeds the maximum utterance length
        """
        if (mlf_file_name == None or not os.path.exists(mlf_file_name)):
            raise Exception("FileNonExistError an error occur while reading MLF file({0})".format(mlf_file_name))

        labels = []
        name2idx = {}
        lab_nframes = []

        start_mlf = False   # flag of "#!MLF!#"
        with open(mlf_file_name, 'r') as file:
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
                        raise Exception("FormatError: an error occur while reading MLF file(%s), no #!MLF!# found" % (mlf_file_name))

                if (re.match('^\"[\.a-zA-Z0-9-_]+\.lab\"$', line)):   #label Name
                    if (not lab_complete):
                        raise Exception("FormatError: an error occur while reading MLF file(%s), incomplete label %s" % (mlf_file_name, lab_name))
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
                        raise Exception("FormatError: an error occur while reading SCP file(%s), missing labels for %s" % (mlf_file_name, lab_name))

                    lab_length = end_pos
                    labels[-1].extend([label_item]*(end_pos - start_pos))
                elif (re.match('^\.$', line)):      # End of current label
                    lab_complete = True
                    if (self.max_utt_len > 0 and delete_toolong and lab_length > self.max_utt_len):
                        # delete the label if it exceed the maximum length
                        del name2idx[lab_name]
                        del labels[-1]
                    else:
                        lab_nframes.append(lab_length)
                else:
                    raise Exception("FormatError: an error occur while reading MLF file(%s), unknown format: %s" % (mlf_file_name, line))

        if not lab_complete:
            raise Exception("FormatError: an error occur while reading MLF file(%s), incomplete label" % (mlf_file_name))

        return (labels, name2idx, lab_nframes)


    def _read_label_mapping(self, mapping_file_name=None):
        """ Read label mapping file.
         Label mapping file: lists all the possible label values, one per line,
         which might be text or numeric. The line number is the identifier that will
         be used by CNTK to identify that label. This parameter is often moved to the
         root level to share with other Command Sections. For example, it’s important
         that the Evaluation Command Section share the same label mapping as the
         trainer, otherwise, the evaluation results will not be accurate.
        """
        if (mapping_file_name == None or not os.path.exists(mapping_file_name)):
            raise Exception("FileNonExistError: an error occur while reading label mapping file(%s)" % (mapping_file_name))

        label_mapping = []
        label_type_int = True
        with open(mapping_file_name, 'r') as file:
            while True:
                line = file.readline().strip()
                if (line == ''): break
                if not re.match('\d+', line):
                    label_type_int = False
                label_mapping.append(line)
        if (label_type_int):
            label_mapping = [int(x) for x in label_mapping]
        return (label_mapping, label_type_int)


    # TODO
    def _read_JSON(self, json_file_name, config_parm, data):
        """ Read the targets as well as other information in JSON format:
         {
           "utts": {
              "utt_name1": {
                "target": "A B C D",
                "targetid": "1 2 3 4",
                "olen": 4,
                ...
              },
           }
         }
        """
        def read_json_file(json_file_name):
            if (json_file_name == None or not os.path.exists(json_file_name)):
                raise Exception("FileNonExistError: an error occur while reading JSON file({0}), File Non-exists".format(json_file_name))

            labels = []
            name2idx = {}
            lab_nframes = []

            with open(json_file_name, 'r') as file:
                json_file = json.load(file)['utts']

            utts = list(json_file.keys())
            for i, utt in enumerate(utts):
                tgt_string_list = json_file[utt]['targetid'].split(' ')
                labels.append([int(tgt) for tgt in tgt_string_list])
                name2idx[utt] = i
                lab_nframes.append(int(json_file[utt]['olen']))
            return (labels, name2idx, lab_nframes)

        (labels, name2idx, lab_nframes) = read_json_file(json_file_name)
        if 'label_mapping' in config_parm:
            label_mapping_path = config_parm['label_mapping']
            (label_mapping, _) = self._read_label_mapping(label_mapping_path)
            self.logger.info("Total %d state names in state list %s" % (len(label_mapping), label_mapping_path))
        else:
            label_mapping = None
        data['data']    = labels;       data['name2idx']      = name2idx
        data['nframes'] = lab_nframes;  data['label_mapping'] = label_mapping
        data['nUtts']   = len(labels);  data['total_nframes'] = sum(lab_nframes)
        self.logger.info("Reading JSON file %s ... total %d entries." % (json_file_name, len(labels)))
