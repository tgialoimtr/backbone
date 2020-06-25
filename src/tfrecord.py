import os, sys
import math
import numpy as np
import cv2 




def arr_feature(arr):
    if np.issubdtype(arr.dtype, np.integer):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = arr.reshape(-1)))
    elif arr.dtype in [np.float32, np.float64]:
        return tf.train.Feature(float_list = tf.train.FloatList(value = arr.reshape(-1)))

def value_feature(value):
    if isinstance(value, int):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    elif isinstance(value, float):
        return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def writeTfrecord0(writer, img, field, fg, words):
    feature = {
        'width':value_feature(img.shape[1]),\
        'height':value_feature(img.shape[0]),\
        'img':arr_feature(img),\
        'fg':arr_feature(fg),\
        'field':arr_feature(field),\
    }

    eg = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(eg.SerializeToString())



# filenames = ['/home/loitg/workspace/backbone/temp/a.tfrecord']
# raw_dataset = tf.data.TFRecordDataset(filenames)

# Create a description of the features.
feature_description = {
    'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'img': tf.io.VarLenFeature(tf.int64),
    'fg': tf.io.VarLenFeature(tf.float32),
    'field': tf.io.VarLenFeature(tf.float32),
}

def _restore_dim(flatten, width, height, channel):
    bb = tf.sparse.to_dense(flatten, default_value=0)
    return tf.reshape(bb,(height,width,channel))

def _parse_function(example_proto):
    parsed_record = tf.io.parse_single_example(example_proto, feature_description)
    width = parsed_record['width']
    height = parsed_record['height']
    parsed_record['img'] = _restore_dim(parsed_record['img'], width, height, 3)
    parsed_record['fg'] = _restore_dim(parsed_record['fg'], width, height, 1)
    parsed_record['field'] = _restore_dim(parsed_record['field'], width, height, 2)
    return parsed_record