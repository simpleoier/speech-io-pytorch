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
feat_config_parms = [dict(file_name='../test_data/tmp_htk.scp', type="SCP", dim=40, context_window=(0,0)), dict(file_name='../test_data/tmp_htk.scp', type="SCP", dim=40, context_window=(0,0))]
label_config_parms = [dict(file_name='../test_data/tmp_htk.mlf', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping'), dict(file_name='', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping')]
label_config_parms = [dict(file_name='../test_data/tmp_htk.mlf', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping'), dict(file_name='../test_data/tmp_htk.mlf', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping')]
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
def test_single_config():
    for i in range(len(feat_config_parms)):
        print(type(feat_config_parms[i]))
        data = HTKDataset(feat_config_parms[i], label_config_parms[i])
        print(data.inputs[0]['nUtts'], data.inputs[0]['nframes'])
        print(data.targets[0]['nUtts'], data.targets[0]['nframes'])
        print(data.inputs[0]['data'])
        print(data.targets[0]['data'])
        if data.targets[0]['data'] is None: continue
        for (key,value) in data.targets[0]['name2idx'].items():
            print(key,value)
        print(data.targets[0]['label_mapping'])

def test_list_config():
    print(type(feat_config_parms))
    data = HTKDataset(feat_config_parms, label_config_parms)
    print(data.inputs[0]['nUtts'], data.inputs[0]['nframes'])
    print(data.targets[0]['nUtts'], data.targets[0]['nframes'])
    print(data.inputs[0]['data'])
    print(data.targets[0]['data'])
    for (key,value) in data.targets[0]['name2idx'].items():
        print(key,value)
    print(data.targets[0]['label_mapping'])

test_list_config()

finish = clock()
print((finish - start) / 1000000)
"""
 Test HTK_IO
"""
from HTK_IO import HTKFeat_read, HTKFeat_write, FBANK, _O

def test_HTK_IO():
    read_file_path = "/home/xkc09/Documents/xkc09/program/kaldi-io/test_data/AMI_EN2001a_H00_MEE068_0000557_0000594_10_AMI_TS3006d_H01_MTD023UID_0158383_0158414.fbank[0,34]"
    htk_reader = HTKFeat_read(read_file_path)
    data = htk_reader.getsegment(10, 34)
    print(data)
    write_file_path = "../test_data/test_write_htk"
    htk_writer = HTKFeat_write(write_file_path, veclen=40, paramKind = (FBANK | _O))
    htk_writer.writeall(data)

test_HTK_IO()