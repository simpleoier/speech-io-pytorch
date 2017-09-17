#!/home/xkc09/Documents/xkc09/src/anaconda3/envs/common-py35/bin/python
# encoding: utf-8

from dataset import HTKDataset
#from dataLoader2 import *
from time import clock

"""
 Test HTKDataset
"""
feat_config_parms = [dict(file_name='../test_data/tmp_htk.scp', type="SCP", dim=40, context_window=(0,0)), dict(file_name='', type="SCP", dim=40, context_window=(0,0))]
feat_config_parms = [dict(file_name='../test_data/tmp_htk.scp', type="SCP", dim=40, context_window=(0,0)), dict(file_name='../test_data/tmp_htk.scp', type="SCP", dim=40, context_window=(0,0))]
feat_config_parms = [dict(file_name='../test_data/tmp_htk.scp', type="SCP", dim=40, context_window=(0,0))]
label_config_parms = [dict(file_name='../test_data/tmp_htk.mlf', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping'), dict(file_name='', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping')]
label_config_parms = [dict(file_name='../test_data/tmp_htk.mlf', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping'), dict(file_name='../test_data/tmp_htk.mlf', type="MLF", dim=4009, label_type="category", label_mapping='../test_data/label_mapping')]

def test_single_config():
    for i in range(len(feat_config_parms)):
        print(type(feat_config_parms[i]))
        data = HTKDataset(feat_config_parms[i], label_config_parms[i])
        print(data.inputs[0]['nUtts'], data.inputs[0]['nframes'])
        print(data.targets[0]['nUtts'], data.targets[0]['nframes'])
        print(data.inputs[0]['data'])
        print(data.targets[0]['data'])
        if data.targets[0]['data'] is None: continue
        #for (key,value) in data.targets[0]['name2idx'].items():
        #    print(key,value)
        #print(data.targets[0]['label_mapping'])

        return data

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

    return data

#start = clock()

#dataset = test_single_config()
#dataset = test_list_config()

#finish = clock()
#print((finish - start) / 1000000)

"""
 Test HTK_IO
"""
from HTK_IO import HTKFeat_read, HTKFeat_write, FBANK, _O

def test_HTK_IO():
    read_file_path = "/home/xkc09/Documents/xkc09/program/kaldi-io/test_data/AMI_EN2001a_H00_MEE068_0000557_0000594_10_AMI_TS3006d_H01_MTD023UID_0158383_0158414.fbank[0,34]"
    htk_reader = HTKFeat_read(read_file_path)
    data = htk_reader.getsegment(10, 34)
    print(data)
    print(type(data))
    data_list = data.tolist()
    print(data_list)
    write_file_path = "../test_data/test_write_htk"
    htk_writer = HTKFeat_write(write_file_path, veclen=40, paramKind = (FBANK | _O))
    htk_writer.writeall(data)

#test_HTK_IO()

from dataLoader2 import HTKDataLoader

def test_dataLoader():
    #dataset = HTKDataset(feat_config_parms, label_config_parms)
    dataset = HTKDataset(feat_config_parms)
    dataloader = HTKDataLoader(dataset, 2, random_size=450, epoch_size=450, truncate_size=50)
    print(dataloader.batch_size, dataloader.random_size, dataloader.epoch_size, dataloader.truncate_size, len(dataloader))
    print(len(dataloader), dataloader.frame_mode, dataset.inputs[0]['total_nframes'], dataset.inputs[0]['nUtts'])
    dataloaderIter = iter(dataloader)
    print(dataloaderIter.epoch_size, dataloaderIter.random_size)
    for epoch in range(5):
        print("Epoch %d" % epoch)
        for batch, _ in dataloaderIter:
            print(batch[0][0].size())
            #size2 = (len(batch[1][0]), len(batch[1][0][0]))
    return dataloader

dataloader = test_dataLoader()
