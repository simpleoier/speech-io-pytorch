#!/home/xkc09/Documents/xkc09/src/anaconda3/envs/common-py35/bin/python
# encoding: utf-8

from HTK_IO import HTKFeat_write, FBANK, _O
import numpy as np
from os import path

np.random.seed(19931225)
datadir = '../test_data'
file_name_prefix = 'artificial_feats'
file_num = 1000
file_len = np.random.randint(100, high=250, size=file_num)
print(file_len)

# Features
if True:
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
if True:
    index = -1
    scp_file_name = path.join(datadir, 'feat.scp')
    scp_file = open(scp_file_name, 'w')
    for i in range(file_num):
        scp_file.write("{0}.feat=/home/xkc09/Documents/xkc09/program/kaldi-io/test_data/{0}.feat[0,{1}]\n".format(file_name_prefix+str(i), file_len[i]-1))
    scp_file.close()

# MLF
if True:
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
