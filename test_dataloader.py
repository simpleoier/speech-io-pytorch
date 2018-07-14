# encoding: utf-8

# ====================================================================
# Copyright 2018 Shanghai Jiao Tong University (Xuankai Chang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Author      : Xuankai Chang
# Email       : netnetchangxk@gmail.com
# Filename    : test_dataloader.py
# Description :
# ====================================================================

from dataset import Dataset
from dataLoader import DataLoader
import os
from os import path
import torch
from torch.autograd import Variable
import numpy as np


if __name__ == '__main__':
    data_dir = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'test_data')
    print(data_dir)

    # HTK feats and mlf alignment
    scp_file_name = path.join(data_dir, 'feat.scp')
    feat_config_solo = [dict(file_name=scp_file_name, type='FEAT', format='HTK', dim=40)]
    target_mlf_name = path.join(data_dir, 'label.mlf')
    target_config_mlf_solo = [dict(file_name=target_mlf_name, type='ALI', format='HTK', label_type='category', dim=52)]
    target_json_name = path.join(data_dir, 'data.json')
    target_config_json_solo = [dict(file_name=target_json_name, type='ALI', format='JSON', label_type='category', dim=52)]
    target_configs = target_config_mlf_solo + target_config_json_solo
    dataset = Dataset(feature_config_parms=feat_config_solo, target_config_parms=target_configs, verify_length=False)

    # Kaldi feats and alignment
    #scp_file_name = path.join(data_dir, 'kaldi_feats.scp')
    #feat_to_len_file_name = path.join(data_dir, 'feats.lengths')
    #feat_config_solo = [dict(file_name=scp_file_name, feat_to_len_file_name=feat_to_len_file_name, type='FEAT', format='Kaldi', dim=40)]
    #target_ali_name = path.join(data_dir, 'kaldi_alignments.scp')
    #target_config_ali_solo = [dict(file_name=target_ali_name, type='ALI', format='Kaldi', label_type='category', dim=52)]
    #dataset = Dataset(feature_config_parms=feat_config_solo, target_config_parms=target_config_ali_solo, verify_length=False)

    dataloaderIter = iter(DataLoader(dataset, batch_size=4, frame_mode=False))
    print(target_configs)

    for batch_cnt, (batch, lengths, keys) in enumerate(dataloaderIter):
        #print(batch_cnt)
        print(lengths, keys)
        print(batch)
        #print(lengths)
        #print(batch[1]) # JSON
        exit()
