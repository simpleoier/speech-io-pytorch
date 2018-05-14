#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from os import path

# Features
from HTK_IO import HTKFeat_write, FBANK, _O
def generate_features():
    index = -1
    dim = 40
    for i in range(file_num):
        file_name = path.join(datadir, file_name_prefix + str(i) + '.feat')
        writer = HTKFeat_write(file_name, veclen=dim, sampPeriod=16000)
        tmp_feat = np.zeros((file_len[i], dim))
        for j in range(file_len[i]):
            index += 1
            tmp_feat[j] = np.array([index] * dim)
        writer.writeall(tmp_feat)

# SCP
def generate_scp():
    index = -1
    scp_file_name = path.join(datadir, 'feat.scp')
    scp_file = open(scp_file_name, 'w')
    for i in range(file_num):
        scp_file.write("{0}.feat={1}/{0}.feat[0,{2}]\n".format(file_name_prefix+str(i), datadir, file_len[i]-1))
    scp_file.close()

# MLF
def generate_mlf():
    index = -1
    mlf_file_name = path.join(datadir, "label.mlf")
    mlf_file = open(mlf_file_name, 'w')
    mlf_file.write("#!MLF!#\n")
    for i in range(file_num):
        mlf_file.write("\"%s.lab\"\n" % (file_name_prefix + str(i)))
        for j in range(file_len[i]):
            index += 1
            mlf_file.write("{:06d} {:06d} {:d}\n".format(j*100000, (j+1)*100000, index))
        mlf_file.write(".\n")
    mlf_file.close()

    # Mapping
    mapping_file_name = path.join(datadir, 'label.mapping')
    mapping_file = open(mapping_file_name, 'w')
    for i in range(index):
        mapping_file.write("%d\n"%i)
    mapping_file.close()

# JSON
import json
def generate_json():
    utts_json = {}
    for i in range(file_num):
        # randomly generate data
        olen = np.random.randint(low=5, high=10)
        targetids = np.random.randint(low=0, high=51, size=olen).tolist()
        targetids = ' '.join([str(x) for x in targetids])
        utt_name = file_name_prefix + str(i)
        utts_json[utt_name] = {'olen': str(olen), 'targetid': targetids}
    data_json = {'utts': utts_json}
    json_file_name = path.join(datadir, 'data.json')
    json_string = json.dumps(data_json, indent=4, ensure_ascii=False)
    with open(json_file_name, 'w') as f:
        f.write(json_string)

if __name__ == "__main__":
    np.random.seed(19931225)
    datadir = path.join(path.dirname(path.dirname(path.realpath(__file__))), 'test_data') #'../test_data'
    file_name_prefix = 'artificial_feats'
    file_num = 10
    file_len = np.random.randint(100, high=250, size=file_num)
    #print(file_len)

    generate_features()
    generate_scp()
    generate_mlf()
    generate_json()