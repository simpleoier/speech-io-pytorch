#!/usr/bin/env python
# encoding: utf-8

from dataLoader import *
#from dataLoader2 import *
from time import clock

"""
 Test HTKDataset
"""
start = clock()
feat_config_parms = [dict(file_name='../test_data/tmp_htk.scp', type="SCP", dim=40, context_window=(0,0)), dict(file_name='', type="SCP", dim=40, context_window=(0,0))]
label_config_parms = [dict(file_name='../test_data/tmp_htk.mlf', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping'), dict(file_name='', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping')]
'''
data = HTKDataset(feat_config_parms, label_config_parms)
for i in range(len(feat_config_parms)):
    print(data.feat_nUtts[i], data.feat_nframes[i])
    print(data.label_nUtts[i], data.label_nframes[i])
    print(data.feats[i])
    print(data.labels[i])
    if data.labels[i] is None: continue
    for (key,value) in data.labels[i].items():
        print(value.shape)
    print(data.label_label_mapping[i])
'''
for i in range(len(feat_config_parms)):
    print(type(feat_config_parms[i]))
    data = HTKDataset(feat_config_parms[i], label_config_parms[i])
    print(data.inputs['nUtts'], data.inputs['nframes'])
    print(data.targets['nUtts'], data.targets['nframes'])
    print(data.inputs['data'])
    print(data.targets['data'])
    if data.targets['data'] is None: continue
    for (key,value) in data.targets['name2idx'].items():
        print(key,value)
    print(data.targets['label_mapping'])
finish = clock()
print((finish - start) / 1000000)
"""
 Test htk_io
"""
from htk_io import open_htk, FBANK, _O

read_file_path = "/slfs1/users/xkc09/asr/PIT/data/mixspeech/ami/data-fbank40/train_10/features_40dim/AMI_EN2001a_H00_MEE068_0000557_0000594_10_AMI_TS3006d_H01_MTD023UID_0158383_0158414.fbank"
htk_reader = open_htk(read_file_path,'rb')
data = htk_reader.getall()
write_file_path = "../test_data/test_write_htk"
htk_write = open_htk(write_file_path, 'wb', veclen=40, paramKind = (FBANK | _O))
htk_write.writeall(data)
