#!/home/xkc09/Documents/xkc09/src/anaconda3/envs/common-py35/bin/python
# encoding: utf-8

import sys, os
from dataset import HTKDataset
from dataLoader3 import HTKDataLoader
import numpy as np
import torch

sys.path.append('/home/xkc09/Documents/xkc09/program/kaldi-io/kaldi-io-pytorch')

data_dir = '/home/xkc09/Documents/xkc09/program/kaldi-io/test_data'
feat_scp_name = os.path.join(data_dir, 'feat.scp')
label_mlf_name = os.path.join(data_dir, 'label.mlf')
label_mapping_name = os.path.join(data_dir, 'label.mapping')
max_utt_len = 220

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
data_loader_frame = HTKDataLoader(data_set, batch_size=256, random_size=34560000, epoch_size=34560000, random_seed=19931225, frame_mode=True)
data_iter = iter(data_loader_frame)
#data_loader_utts = HTKDataLoader(data_set, batch_size=1, random_size=34560000, epoch_size=34560000, random_seed=19931225, frame_mode=False)
#data_iter = iter(data_loader_utts)

mean, std = data_iter.normalize('globalMeanVar')
for data_idx in range(len(mean)):
    for item_idx in range(len(mean[data_idx])):
        if mean[data_idx][item_idx] is None: continue
        mean_item = mean[data_idx][item_idx]
        std_item = std[data_idx][item_idx]
        mean_item = np.tile(mean_item, (leftFeatContext + 1 + rightFeatContext))
        std_item = np.tile(std_item, (leftFeatContext + 1 + rightFeatContext))
        mean[data_idx][item_idx] = torch.from_numpy(mean_item).float()
        std[data_idx][item_idx] = torch.from_numpy(std_item).float()

def normalize(input, mean, std):
    output = input - mean
    output = output / std
    return output

for epoch in range(1):
    print("Epoch: %d" % epoch)
    batch_cnt = 0
    for batch, lengths in data_iter:
        batch_cnt += 1
        feat = batch[0][0].view(batch[0][0].size(0), -1)
        label = batch[1][0].long()

        feat_norm = normalize(feat, mean[0][0], std[0][0])
        print(feat)
        print(feat.mean(dim=0))
        print(feat.std(dim=0))
        print(feat_norm)
        print(feat_norm.mean(dim=0))
        print(feat_norm.std(dim=0))
        exit()

        for i in range(feat.size(0)):
            if int(feat[i][200]) != int(label[i][0]):
                print("NO")

        #print(batch_cnt, feat.size(), label.size())

        #for i in range(feat.size(0)):
        #    if (int(feat[i][200]) != int(label[i])):
        #        print("False:", feat[i][0], label[i])
