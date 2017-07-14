#!/usr/bin/env python
# encoding: utf-8

from struct import unpack, pack
import numpy as np
import os, re

LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11

_E = 0o000100    # has energy
_N = 0o000200    # absolute energy supressed
_D = 0o000400    # has delta coefficients
_A = 0o001000    # has acceleration (delta-delta) coefficients
_C = 0o002000    # is compressed
_Z = 0o004000    # has zero mean static coefficients
_K = 0o010000    # has CRC checksum
_O = 0o020000    # has 0th cepstral coefficient
_V = 0o040000    # has VQ data
_T = 0o100000    # has third differential coefficients

def HTK_open(f, mode=None, veclen=13, **kargs):
    """ Open an HTK format feature file for reading or writing.
        The mode parameter is 'rb' (reading) or 'wb' (writing).
    """
    if not (os.path.exists(f)):
        raise Exception("File %s does not exist!" % f)
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb'):
        return HTKFeat_read(f)  # veclen is ignored since it's in the file
    elif mode in ('w', 'wb'):
        return HTKFeat_write(f, veclen, **kargs)
    else:
        raise Exception("mode must be 'r', 'rb', 'w', or 'wb'")

class HTKFeat_read(object):
    """ Read HTK format feature files. """
    def __init__(self, file_name=None):
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        # print(re.match('^[\w\_\.\/-]+\[\d+,\d+\]$', file_name))
        if (re.match('^[\w\_\.\/-]+\[\d+,\d+\]$', file_name)):
            file_name = file_name[:-1]
            lst = file_name.split('[')
            file_name = lst[0]
            rest_info = lst[1]
            start_f, end_f = rest_info.split(',')
            self.start_f = int(start_f)
            self.end_f = int(end_f)
        if (file_name != None):
            self.open(file_name)

    def __iter__(self):
        self.fh.seek(12,0)
        return self

    def open(self, file_name):
        self.file_name = file_name
        self.fh = open(file_name, 'rb')
        self.readheader()

    def __del__(self):
        self.close()

    def close(self):
        self.fh.close()

    def readheader(self):
        self.fh.seek(0, 0)  # move the file pointer to the 0 byte
        spam = self.fh.read(12)
        self.nSamples, self.sampPeriod, self.sampSize, self.parmKind = unpack(">IIHH", spam)
        # Get coefficients for compressed data
        if self.parmKind & _C:
            self.dtype = 'h'
            self.veclen = self.sampSize // 2
            if self.parmKind & 0x3f == IREFC:
                self.A = 32767
                self.B = 0
            else:
                self.A = np.fromfile(self.fh, 'f', self.veclen)
                self.B = np.fromfile(self.fh, 'f', self.veclen)
                if self.swap:
                    self.A = self.A.byteswap()
                    self.B = self.B.byteswap()
        else:
            self.dtype = 'f'
            self.veclen = self.sampSize // 4
        self.hdrlen = self.fh.tell()

    def seek(self, idx):
        self.fh.seek(self.hdrlen + idx * self.sampSize, 0)

    def next(self):
        vec = np.fromfile(self.fh, self.dtype, self.veclen)
        if len(vec) == 0:
            raise StopIteration
        if self.swap:
            vec = vec.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            vec = (vec.astype('f') + self.B) / self.A
        return vec

    def readvec(self):
        return self.next()

    def getsegment(self, start_f, end_f):
        self.seek(self.start_f)
        data = np.zeros((end_f - start_f + 1, self.veclen), self.dtype)
        for i in range(end_f - start_f + 1):
            data[i] = self.next()
        return data

    def getall(self):
        self.seek(0)
        data = np.fromfile(self.fh, self.dtype)
        if self.parmKind & _K:  # Remove and ignore checksum
            data = data[:-1]
        data = data.reshape(len(data) // self.veclen, self.veclen)
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data

class HTKFeat_write(object):
    """ Write Sphinx-II format feature files """
    def __init__(self, file_name=None, veclen=13, sampPeriod=100000, paramKind=(FBANK | _O)):
        self.veclen = veclen
        self.sampPeriod = sampPeriod
        self.sampSize = veclen * 4
        self.paramKind = paramKind
        self.dtype = 'f'
        self.filesize = 0
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (file_name != None):
            self.open(file_name)

    def __del__(self):
        self.close()

    def open(self, file_name):
        self.file_name = file_name
        self.fh = open(file_name, 'wb')
        self.writeheader()

    def close(self):
        self.writeheader()
        self.fh.close()

    def writeheader(self):
        self.fh.seek(0, 0)
        self.fh.write(pack(">IIHH", self.filesize, self.sampPeriod, self.sampSize, self.paramKind))

    def writevec(self, vec):
        if len(vec) != self.veclen:
            raise Exception("Vector length must be %d" % self.veclen)
        if self.swap:
            np.array(vec, self.dtype).byteswap().tofile(self.fh)
        else:
            np.array(vec, self.dtype).tofile(self.fh)
        self.filesize = self.filesize + self.veclen

    def writeall(self, arr):
        for row in arr:
            self.writevec(row)
