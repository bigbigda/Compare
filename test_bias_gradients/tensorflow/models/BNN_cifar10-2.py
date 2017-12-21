#coding=utf-8

from nnUtils import *

model = Sequential([
    BinarizedWeightOnlySpatialConvolution(128,3,3,1,1, padding='SAME', bias=True),
    BatchNormalization(),
    HardTanh(),
    BinarizedSpatialConvolution(128,3,3, padding='SAME', bias=True),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    HardTanh(),
    BinarizedSpatialConvolution(256,3,3, padding='SAME', bias=True),
    BatchNormalization(),
    HardTanh(),
    BinarizedSpatialConvolution(256,3,3, padding='SAME', bias=True),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    HardTanh(),
    BinarizedSpatialConvolution(512,3,3, padding='SAME', bias=True),
    BatchNormalization(),
    HardTanh(),
    BinarizedSpatialConvolution(512,3,3, padding='SAME', bias=True),
    SpatialMaxPooling(2,2,2,2),
    BatchNormalization(),
    HardTanh(),
    BinarizedAffine(1024, bias=True),
    BatchNormalization(),
    HardTanh(),
    BinarizedAffine(1024, bias=True),
    BatchNormalization(),
    HardTanh(),
    BinarizedAffine(10),
    BatchNormalization()
])

