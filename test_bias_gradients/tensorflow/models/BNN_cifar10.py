#coding=utf-8

from nnUtils import *


model = Sequential([
    BinarizedWeightOnlySpatialConvolution(128,3,3,1,1, padding='SAME', bias=True),
    BatchNormalization(scale=True,is_training=False,epsilon=1e-4,decay=0.9),
    HardTanh(),
    BinarizedSpatialConvolution(128,3,3, padding='SAME', bias=True),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(scale=True,is_training=False,epsilon=1e-4,decay=0.9),
    HardTanh(),
    BinarizedSpatialConvolution(256,3,3, padding='SAME', bias=True),
    BatchNormalization(scale=True,is_training=False,epsilon=1e-4,decay=0.9),
    HardTanh(),
    BinarizedSpatialConvolution(256,3,3, padding='SAME', bias=True),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(scale=True,is_training=False,epsilon=1e-4,decay=0.9),
    HardTanh(),
    BinarizedSpatialConvolution(512,3,3, padding='SAME', bias=True),
    BatchNormalization(scale=True,is_training=False,epsilon=1e-4,decay=0.9),
    HardTanh(),
    BinarizedSpatialConvolution(512,3,3, padding='SAME', bias=True),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(scale=True,is_training=False,epsilon=1e-4,decay=0.9),
    HardTanh(),
    BinarizedAffine2(1024, bias=True),
    BatchNormalization(scale=True,is_training=False,epsilon=1e-4,decay=0.9),
    HardTanh(),
    BinarizedAffine(1024, bias=True),
    BatchNormalization(scale=True,is_training=False,epsilon=1e-4,decay=0.9),
    HardTanh(),
    BinarizedAffine(10),
    BatchNormalization(scale=True,is_training=False,epsilon=1e-4,decay=0.9)
])
