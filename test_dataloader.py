# encoding: utf-8

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

    scp_file_name = path.join(data_dir, 'feat.scp')
    feat_config_solo = [dict(file_name=scp_file_name, type='SCP', format='HTK', dim=40)]
    target_mlf_name = path.join(data_dir, 'label.mlf')
    target_config_mlf_solo = [dict(file_name=target_mlf_name, type='MLF', label_type='category', dim=52)]
    target_json_name = path.join(data_dir, 'data.json')
    target_config_json_solo = [dict(file_name=target_json_name, type='JSON', label_type='category', dim=52)]
    target_configs = target_config_mlf_solo + target_config_json_solo

    dataset = Dataset(feature_config_parms=feat_config_solo, target_config_parms=target_configs, verify_length=False)
    dataloaderIter = iter(DataLoader(dataset, batch_size=4, frame_mode=False))

    for batch_cnt, (batch, standard_lengths, keys, lengths) in enumerate(dataloaderIter):
        #print(batch_cnt)
        #print(standard_lengths, keys)
        #print(lengths)
        #print(batch[1]) # JSON
        exit()
