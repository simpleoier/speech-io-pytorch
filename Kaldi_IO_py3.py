#!/usr/bin/env python
# encoding: utf-8

#!/usr/bin/env python
# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
import os
import sys
import re
import gzip
import struct

if sys.version_info[0] > 2:
    def str_or_bytes(bytes_string):
        return bytes_string.decode('utf-8') if isinstance(bytes_string, bytes) else bytes_string.encode('utf-8')
else:
    def str_or_bytes(bytes_string):
        return bytes_string


#################################################

IS_BIN = str_or_bytes('\x00B')
IS_EOL = str_or_bytes('\04')
IS_SPACE = str_or_bytes(' ')
IS_EMPTY = str_or_bytes('')
INT32 = str_or_bytes('\4')
FLOAT_VEC = str_or_bytes('FV ')
FLOAT_MAT = str_or_bytes('FM ')
DOUBLE_VEC = str_or_bytes('DV ')
DOUBLE_MAT = str_or_bytes('DM ')
COMPRESSED = str_or_bytes('CM')

# Omitting the int variable format, since it is not used for reading
# Support for compressed matrices from : https://github.com/vesis84/kaldi-io-for-python/pull/6/files?diff=split
# https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.cc
# https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
GLOBAL_HEADER = np.dtype([('minvalue', 'float32'), ('range', 'float32'), ('num_rows', '<i'), ('num_cols', '<i')])
PER_COL_HEADER = np.dtype([('percentile_0', 'uint16'), ('percentile_25', 'uint16'), ('percentile_75', 'uint16'),
                           ('percentile_100', 'uint16')])


#################################################
# Adding kaldi tools to shell path,

# Select kaldi,
if 'KALDI' not in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI'] = '/opt/kaldi'
    # Add kaldi tools to path,
    os.environ['PATH'] = os.popen('echo $KALDI/src/*bin | sed "s/ /:/g"').readline().strip() + ':' + os.environ['PATH']

#################################################
# Data-type independent helper functions,


def open_or_fd(file, mode='rb'):
    """ fd = open_or_fd(file)
    Open file, gzipped file, pipe, or forward the file-descriptor.
    Eventually seeks in the 'file' argument contains ':offset' suffix.
    """
    offset = None
    try:
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
            (prefix, file) = file.split(':', 1)
        # separate offset from filename (optional),
        if re.search(':[0-9]+$', file):
            (file, offset) = file.rsplit(':', 1)
        # input pipe?
        if file[-1] == '|':
            fd = os.popen(file[:-1], 'rb')
        # output pipe?
        elif file[0] == '|':
            fd = os.popen(file[1:], 'wb')
        # is it gzipped?
        elif file.split('.')[-1] == 'gz':
            fd = gzip.open(file, mode)
        # a normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    if offset is not None:
        fd.seek(int(offset))
    return fd


def read_key(fd):
    """ [key] = read_key(fd)
    Read the utterance-key from the opened ark/stream descriptor 'fd'.
    """
    key = IS_EMPTY
    while 1:
        char = fd.read(1)
        if char == IS_EMPTY:
            break
        if char == IS_SPACE:
            break
        key += char
    key = key.strip()
    if key == IS_EMPTY:
        return None  # end of file,
    assert(re.match(str_or_bytes('^[\.a-zA-Z0-9_-]+$'), key) is not None)  # check format,
    return key


#################################################
# Integer vectors (alignments, ...),

def read_ali_ark(file_or_fd):
    """ Alias to 'read_vec_int_ark()' """
    return read_vec_int_ark(file_or_fd)


def read_vec_int_ark(file_or_fd):
    """ generator(key,vec) = read_vec_int_ark(file_or_fd)
    Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.
    file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

    Read ark to a 'dictionary':
    d = { u:d for u,d in kaldi_io.read_vec_int_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_int(fd)
            yield str_or_bytes(key), ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_vec_int(file_or_fd):
    """ [int-vec] = read_vec_int(file_or_fd)
    Read kaldi integer vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2)
    if binary == IS_BIN:
        assert(fd.read(1) == INT32), 'Not an int32!'
        vec_size = struct.unpack('<i', str_or_bytes(fd.read(4)))[0]
        # Vectors are structured as (LENGTH_OF_POST,VALUE)
        vec = np.fromfile(fd, dtype=[('lenpost', np.int8), ('post', '<i')], count=vec_size)
        ans = vec[:]['post']
        return ans
    else:  # ascii,
        arr = (binary + fd.readline()).strip().split()
        try:
            arr.remove('[')
            arr.remove(']')
        except ValueError:
            pass
        ans = np.array(arr, dtype=int)
    if fd is not file_or_fd:
        fd.close()  # cleanup
    return ans


# Writing,
def write_vec_int(file_or_fd, v, key=IS_EMPTY):
    """ write_vec_int(f, v, key=IS_EMPTY)
    Write a binary kaldi integer vector to filename or stream.
    Arguments:
    file_or_fd : filename or opened file descriptor for writing,
    v : the vector to be stored,
    key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

    Example of writing single vector:
    kaldi_io.write_vec_int(filename, vec)

    Example of writing arkfile:
    with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
    """
    fd = open_or_fd(file_or_fd, mode='wb')
    try:
        if str_or_bytes(key) != IS_EMPTY:
            fd.write(str_or_bytes(key)+IS_SPACE)  # ark-files have keys (utterance-id)
        fd.write(IS_BIN)
        # dim,
        fd.write(INT32)
        fd.write(struct.pack('<i', v.shape[0]))
        # data,
        for i in range(len(v)):
            fd.write(INT32)
            fd.write(struct.pack('<i', v[i]))  # binary,
    finally:
        if fd is not file_or_fd:
            fd.close()


#################################################
# Float vectors (confidences, ivectors, ...),

def uncompress(value, p0, p25, p75, p100):
    if value <= 64:
        return p0 + (p25 - p0) * value * (1./64)
    elif value <= 192:
        return p25 + (p75 - p25) * (value - 64) * (1./128)
    else:
        return p75 + (p100 - p75) * (value - 192) * (1./63)


def read_compress_mat(fd, compress_type):
    global_header = np.fromfile(fd, dtype=GLOBAL_HEADER, count=1)
    global_range, global_min = global_header['range'][0], global_header['minvalue'][0]
    rows, cols = global_header['num_rows'][0], global_header['num_cols'][0]

    def uinttofloat(val):
        return global_min + global_range * 1.52590218966964e-05 * val
    mat = np.empty((rows, cols), dtype=float)
    # - cols because we firstly read the colheaders
    if compress_type == COMPRESSED + b' ':
        size = cols * (PER_COL_HEADER.itemsize + rows) - (PER_COL_HEADER.itemsize*cols)
    else:
        size = (2 * rows * cols) - (PER_COL_HEADER.itemsize*cols)
    # The data is structured as [Colheader, ... , Colheader, Data, Data , .... ]
    #                         {           cols           }{     size         }
    col_headers = np.fromfile(fd, dtype=PER_COL_HEADER, count=cols)
    data = np.fromfile(fd, dtype='B', count=size)
    for i, col_head in enumerate(col_headers):
        col_head = list(map(uinttofloat, col_head))
        mat[:, i] = [uncompress(data[j], *col_head) for j in range(i * rows, (i * rows) + rows)]
    return mat


# Reading
def read_vec_flt_scp(file_or_fd):
    """ generator(key,mat) = read_vec_flt_scp(file_or_fd)
    Returns generator of (key,vector) tuples, read according to kaldi scp.
    file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

    Iterate the scp:
    for key,vec in kaldi_io.read_vec_flt_scp(file):
     ...

    Read scp to a 'dictionary':
    d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key, rxfile) = line.split(IS_SPACE)
            vec = read_vec_flt(rxfile)
            yield str_or_bytes(key), vec
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_vec_flt_ark(file_or_fd):
    """ generator(key,vec) = read_vec_flt_ark(file_or_fd)
    Create generator of (key,vector<float>) tuples, reading from an ark file/stream.
    file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

    Read ark to a 'dictionary':
    d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_flt(fd)
            yield str_or_bytes(key), ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_vec_flt(file_or_fd):
    """ [flt-vec] = read_vec_flt(file_or_fd)
    Read kaldi float vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2)
    if binary == IS_BIN:
        # Data type
        dtype = fd.read(3)
        # CM and CM1 are possible values
        if dtype.startswith(COMPRESSED):
            return read_compress_mat(fd, dtype)
        elif dtype == FLOAT_VEC:
            sample_size = 4
        elif dtype == DOUBLE_VEC:
            sample_size = 8
        else:
            sys.exit("Vector is empty!")
        assert(fd.read(1) == INT32), 'Not int32!'
        vec_size = struct.unpack('<i', str_or_bytes(fd.read(sample_size)))[0]  # vector dim
        # Read whole vector,
        buf = fd.read(vec_size * sample_size)
        if sample_size == 4:
            ans = np.frombuffer(buf, dtype='float32')
        elif sample_size == 8:
            ans = np.frombuffer(buf, dtype='float64')
        else:
            raise ValueError('Bad sample size')
        return ans
    else:  # ascii,
        arr = (binary + fd.readline()).strip().split()
        try:
            arr.remove('[')
            arr.remove(']')
        except ValueError:
            pass
        ans = np.array(arr, dtype=float)
    if fd is not file_or_fd:
        fd.close()  # cleanup
    return ans


# Writing
def write_vec_flt(file_or_fd, v, key=IS_EMPTY):
    """ write_vec_flt(f, v, key=IS_EMPTY)
    Write a binary kaldi vector to filename or stream. Supports 32bit and 64bit floats.
    Arguments:
    file_or_fd : filename or opened file descriptor for writing,
    v : the vector to be stored,
    key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

    Example of writing single vector:
    kaldi_io.write_vec_flt(filename, vec)

    Example of writing arkfile:
    with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
    """
    fd = open_or_fd(file_or_fd, mode='wb')
    try:
        if str_or_bytes(key) != IS_EMPTY:
            fd.write(str_or_bytes(key)+IS_SPACE)  # ark-files have keys (utterance-id)
        fd.write(IS_BIN)
        # Data-type,
        if v.dtype == 'float32':
            fd.write(FLOAT_VEC)
        elif v.dtype == 'float64':
            fd.write(DOUBLE_VEC)
        else:
            raise TypeError
        # Dim
        fd.write(IS_EOL)
        fd.write(struct.pack('I', v.shape[0]))  # dim
        # Data
        v.tofile(fd, sep="")  # binary
    finally:
        if fd is not file_or_fd:
            fd.close()


#################################################
# Float matrices (features, transformations, ...),

# Reading,
def read_mat_scp(file_or_fd):
    """ generator(key,mat) = read_mat_scp(file_or_fd)
    Returns generator of (key,matrix) tuples, read according to kaldi scp.
    file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

    Iterate the scp:
    for key,mat in kaldi_io.read_mat_scp(file):
     ...

    Read scp to a 'dictionary':
    d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key, rxfile) = line.split(IS_SPACE)
            mat = read_mat(str_or_bytes(rxfile))
            yield str_or_bytes(key), mat
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_mat_ark(file_or_fd):
    """ generator(key,mat) = read_mat_ark(file_or_fd)
    Returns generator of (key,matrix) tuples, read from ark file/stream.
    file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

    Iterate the ark:
    for key,mat in kaldi_io.read_mat_ark(file):
     ...

    Read ark to a 'dictionary':
    d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            mat = read_mat(fd)
            yield str_or_bytes(key), mat
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_mat(file_or_fd):
    """ [mat] = read_mat(file_or_fd)
    Reads single kaldi matrix, supports ascii and binary.
    file_or_fd : file, gzipped file, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        binary = fd.read(2)
        if binary == IS_BIN:
            mat = read_mat_binary(fd)
        else:
            assert(binary == ' [')
            mat = read_mat_ascii(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()
    return mat


def read_mat_binary(fd):
    # Data type
    dtype = fd.read(3)
    # CM and CM1 are possible values
    if dtype.startswith(COMPRESSED):
        return read_compress_mat(fd, dtype)
    if dtype == FLOAT_MAT:
        sample_size = 4
    elif dtype == DOUBLE_MAT:
        sample_size = 8
    else:
        sys.exit("Vector is empty!")
    # Dimensions
    fd.read(1)
    rows = struct.unpack('<i', fd.read(sample_size))[0]
    fd.read(1)
    cols = struct.unpack('<i', fd.read(sample_size))[0]
    # Read whole matrix
    buf = fd.read(rows * cols * sample_size)
    if sample_size == 4:
        vec = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8:
        vec = np.frombuffer(buf, dtype='float64')
    else:
        raise ValueError('Bad sample size')
    mat = np.reshape(vec, (rows, cols))
    return mat


def read_mat_ascii(fd):
    rows = []
    while 1:
        line = fd.readline()
        if len(line) == 0:
            raise EOFError('Bad input format')
        if len(line.strip()) == 0:
            continue  # skip empty line
        arr = line.strip().split()
        if arr[-1] != ']':
            rows.append(np.array(arr, dtype='float32'))  # not last line
        else:
            rows.append(np.array(arr[:-1], dtype='float32'))  # last line
            mat = np.vstack(rows)
            return mat


# Writing
def write_mat(file_or_fd, m, key=IS_EMPTY):
    """ write_mat(f, m, key=IS_EMPTY)
    Write a binary kaldi matrix to filename or stream. Supports 32bit and 64bit floats.
    Arguments:
    file_or_fd : filename of opened file descriptor for writing,
    m : the matrix to be stored,
    key (optional) : used for writing ark-file, the utterance-id gets written before the matrix.

    Example of writing single matrix:
    kaldi_io.write_mat(filename, mat)

    Example of writing arkfile:
    with open(ark_file,'w') as f:
     for key,mat in dict.iteritems():
       kaldi_io.write_mat(f, mat, key=key)
    """
    fd = open_or_fd(file_or_fd, mode='wb')
    try:
        if str_or_bytes(key) != IS_EMPTY:
            fd.write(str_or_bytes(key) + IS_SPACE)  # ark-files have keys (utterance-id)
        fd.write(IS_BIN)
        # Data-type
        if m.dtype == 'float32':
            fd.write(FLOAT_MAT)
        elif m.dtype == 'float64':
            fd.write(DOUBLE_MAT)
        else:
            raise TypeError
        # Dims
        fd.write(IS_EOL)
        fd.write(struct.pack('I', m.shape[0]))  # rows
        fd.write(IS_EOL)
        fd.write(struct.pack('I', m.shape[1]))  # cols
        # Data
        #m.tofile(fd, sep="")  # binary
        fd.write(m.tobytes())
    finally:
        if fd is not file_or_fd:
            fd.close()


#################################################
# 'Posterior' kaldi type (posteriors, confusion network, nnet1 training targets, ...)
# Corresponds to: vector<vector<tuple<int,float> > >
# - outer vector: time axis
# - inner vector: records at the time
# - tuple: int = index, float = value
#

def read_cnet_ark(file_or_fd):
    """ Alias of function 'read_post_ark()', 'cnet' = confusion network """
    return read_post_ark(file_or_fd)


def read_post_ark(file_or_fd):
    """ generator(key,vec<vec<int,float>>) = read_post_ark(file)
    Returns generator of (key,posterior) tuples, read from ark file.
    file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

    Iterate the ark:
    for key,post in kaldi_io.read_post_ark(file):
     ...

    Read ark to a 'dictionary':
    d = { key:post for key,post in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            post = read_post(fd)
            yield str_or_bytes(key), post
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_post(file_or_fd):
    """ [post] = read_post(file_or_fd)
    Reads single kaldi 'Posterior' in binary format.

    The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
    the outer-vector is usually time axis, inner-vector are the records
    at given time,  and the tuple is composed of an 'index' (integer)
    and a 'float-value'. The 'float-value' can represent a probability
    or any other numeric value.

    Returns vector of vectors of tuples.
    """
    fd = open_or_fd(file_or_fd)
    ans = []
    binary = fd.read(2)
    assert(binary == IS_BIN), 'Vector is not in binary format'
    assert(fd.read(1) == INT32), 'Not an int32!'
    outer_vec_size = struct.unpack('<i', str_or_bytes(fd.read(4)))[0]  # number of frames (or bins)

    # Loop over 'outer-vector',
    for i in range(outer_vec_size):
        assert(fd.read(1) == INT32), 'Not an int32!'
        inner_vec_size = struct.unpack('<i', str_or_bytes(fd.read(4)))[0]  # number of records for frame (or bin)
        int_id = np.zeros(inner_vec_size, dtype=int)  # buffer for integer id's
        post = np.zeros(inner_vec_size, dtype=float)  # buffer for posteriors

        # Loop over 'inner-vector',
        for j in range(inner_vec_size):
            assert(fd.read(1) == INT32), 'Not an int32!'
            int_id[j] = struct.unpack('<i', str_or_bytes(fd.read(4)))[0]  # id
            assert(fd.read(1) == INT32), 'Not an int32!'
            post[j] = struct.unpack('<f', str_or_bytes(fd.read(4)))[0]  # post

        # Append the 'inner-vector' of tuples into the 'outer-vector'
        ans.append(zip(int_id, post))

    if fd is not file_or_fd:
        fd.close()
    return ans


#################################################
# Kaldi Confusion Network bin begin/end times,
# (kaldi stores CNs time info separately from the Posterior).
#

def read_cntime_ark(file_or_fd):
    """ generator(key,vec<tuple<float,float>>) = read_cntime_ark(file_or_fd)
    Returns generator of (key,cntime) tuples, read from ark file.
    file_or_fd : file, gzipped file, pipe or opened file descriptor.

    Iterate the ark:
    for key,time in kaldi_io.read_cntime_ark(file):
     ...

    Read ark to a 'dictionary':
    d = { key:time for key,time in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            cntime = read_cntime(fd)
            yield str_or_bytes(key), cntime
            key = read_key(fd)
    finally:
        if fd is not file_or_fd:
            fd.close()


def read_cntime(file_or_fd):
    """ [cntime] = read_cntime(file_or_fd)
    Reads single kaldi 'Confusion Network time info', in binary format:
    C++ type: vector<tuple<float,float> >.
    (begin/end times of bins at the confusion network).

    Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'

    file_or_fd : file, gzipped file, pipe or opened file descriptor.

    Returns vector of tuples.
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2)
    assert(binary == IS_BIN), 'Vector is not in binary format'
    assert(fd.read(1) == INT32), 'Not an int32!'
    # Get number of bins,
    vec_size = struct.unpack('<i', str_or_bytes(fd.read(4)))[0]  # number of frames (or bins)
    t_beg = np.zeros(vec_size, dtype=float)
    t_end = np.zeros(vec_size, dtype=float)

    # Loop over number of bins,
    for i in range(vec_size):
        assert(fd.read(1) == INT32), 'Not an int32!'
        t_beg[i] = struct.unpack('<f', str_or_bytes(fd.read(4)))[0]  # begin-time of bin
        assert(fd.read(1) == INT32), 'Not an int32!'
        t_end[i] = struct.unpack('<f', str_or_bytes(fd.read(4)))[0]  # end-time of bin

    # Return vector of tuples,
    ans = zip(t_beg, t_end)
    if fd is not file_or_fd:
        fd.close()
    return ans


#################################################
# Segments related,
#

# Segments as 'Bool vectors' can be handy,
# - for 'superposing' the segmentations,
# - for frame-selection in Speaker-ID experiments,
def read_segments_as_bool_vec(segments_file):
    """ [ bool_vec ] = read_segments_as_bool_vec(segments_file)
    using kaldi 'segments' file for 1 wav, format : '<utt> <rec> <t-beg> <t-end>'
    - t-beg, t-end is in seconds,
    - assumed 100 frames/second,
    """
    segs = np.loadtxt(segments_file, dtype='object,object,f,f', ndmin=1)
    # Sanity checks
    assert(len(segs) > 0), 'Empty segment'
    assert(len(np.unique([rec[1] for rec in segs])) == 1), 'Segment with only 1 wav-file'
    # Convert time to frame-indexes
    start = np.rint([100 * rec[2] for rec in segs]).astype(int)
    end = np.rint([100 * rec[3] for rec in segs]).astype(int)
    # Taken from 'read_lab_to_bool_vec', htk.py
    frms = np.repeat(np.r_[np.tile([False, True], len(end)), False],
                     np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat, 0])
    assert np.sum(end-start) == np.sum(frms)
    return frms
