#!/usr/bin/env python
# encoding: utf-8

import sys, os
from dataset2 import HTKDataset
import dataLoader
import dataLoader2
import numpy as np
import torch
import time
sys.getdefaultencoding()

sys.path.append('/slfs1/users/xkc09/program/kaldi-io-pytorch')

data_dir1 = '/slfs1/users/xkc09/program/test_data'
data_dir2 = '/slfs1/users/xkc09/asr/wsj/data/train_si284/train_tr90'
feat_scp_name  = os.path.join(data_dir2, 'feats.scp')
label_mlf_name = os.path.join(data_dir2, 'labels.mlf')
label_mapping_name = os.path.join(data_dir1, 'label.mapping')
max_utt_len = 2000

leftFeatContext = 5
rightFeatContext = 5

label_mapping = open(label_mapping_name, 'w')
for i in range(1024):
    label_mapping.write(str(i)+'\n')
label_mapping.close()

feat_config_parms = [dict(file_name=feat_scp_name, type="SCP",
                          dim=40, context_window=(leftFeatContext, rightFeatContext))
                     ]
label_config_parms = [dict(file_name=label_mlf_name, type="MLF",
                           dim=1024, label_type="category",
                           label_mapping=label_mapping_name)
                      ]

data_set = HTKDataset(feat_config_parms, label_config_parms, max_utt_len)
#data_loader_frame1 = dataLoader.HTKDataLoader(data_set, batch_size=256, random_size=34560000, epoch_size=34560000, random_seed=19931225, frame_mode=True)
#data_iter1 = iter(data_loader_frame1)
#data_loader_frame2 = dataLoader2.HTKDataLoader(data_set, batch_size=256, random_size=345600, epoch_size=345600, random_seed=19931225, frame_mode=True)
#data_iter2 = iter(data_loader_frame2)
data_loader_utts2 = dataLoader2.HTKDataLoader(data_set, batch_size=1, random_size=345600, epoch_size=345600, random_seed=19931225, frame_mode=False)
data_iter2 = iter(data_loader_utts2)

#for batch, length, key in data_iter2.eval():
#    print(batch, length, key)
#    exit()

#mean1, std1 = data_iter1.normalize('globalMeanVar')
#mean2, std2 = data_iter2.normalize('globalMeanVar')
#mean, std = data_iter2.normalize('globalMeanVar')
mean, std = [[[np.zeros(3)]]], [[[np.zeros(4)]]]
#print(mean)
#print(std)
#for data_idx in range(len(mean)):
#    for item_idx in range(len(mean[data_idx])):
#        if mean[data_idx][item_idx] is None: continue
#        mean_item = mean[data_idx][item_idx]
#        std_item = std[data_idx][item_idx]
#        mean_item = np.tile(mean_item, (leftFeatContext + 1 + rightFeatContext))
#        std_item = np.tile(std_item, (leftFeatContext + 1 + rightFeatContext))
#        mean[data_idx][item_idx] = torch.from_numpy(mean_item).float()
#        std[data_idx][item_idx] = torch.from_numpy(std_item).float()

def normalize(input, mean, std):
    return (input - mean) / std

for epoch in range(1):
    print("Epoch: %d" % epoch)
    batch_cnt = 0
    for batch, lengths, keys in data_iter2:
        batch_cnt += 1
        #feat = batch[0][0].view(batch[0][0].size(0), -1)
        feat = batch[0][0]
        label = batch[1][0].long()
        #print(feat.size(), mean[0][0].size(), std[0][0].size())
        print(feat.size())

        #feat_norm = normalize(feat, mean[0][0], std[0][0])
        print(feat)
        print(feat.mean(dim=0))
        print(feat.std(dim=0))
        print(mean)
        print(std)
        exit()

        for i in range(feat.size(0)):
            if int(feat[i][200]) != int(label[i][0]):
                print("NO")

        #print(batch_cnt, feat.size(), label.size())

        #for i in range(feat.size(0)):
        #    if (int(feat[i][200]) != int(label[i])):
        #        print("False:", feat[i][0], label[i])
