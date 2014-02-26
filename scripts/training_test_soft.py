import glob
import pandas as pd
from pandas import DataFrame
import numpy as np
import docs
import os
from scipy.io import wavfile
import subprocess
from jfsantos import timit_dataset
from pylearn2.datasets import mnist
import pickle
from pylearn2.utils import serial
import theano.tensor as T
import theano
import dataset_classes
from pylearn2.config import yaml_parse
from pylearn2.termination_criteria import MonitorBased


train = """!obj:pylearn2.train.Train {
    dataset: &train !obj:jfsantos.timit_dataset.TimitPhoneData {
        datapath: '/Users/alexis/university/ift6266/data/timit/raw/TIMIT',
        framelen: 1000,
        overlap: 800,
        start: 0,
        stop: 2000
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.mlp.Sigmoid {
                     layer_name: 'h0',
                     dim: 800,
                     sparse_init: 15,
                 }, !obj:pylearn2.models.mlp.Softmax {
                     layer_name: 'y',
                     n_classes: 61,
                     irange: 10.
                 }
                ],
        nvis: 1000,
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 1000000,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        updates_per_batch: 10,
        monitoring_dataset:
            {
                'train' : *train,
                'valid' : !obj:jfsantos.timit_dataset.TimitPhoneData {
                              datapath: '/Users/alexis/university/ift6266/data/timit/raw/TIMIT',
                              framelen: 1000,
                              overlap: 800,
                              start: 2000,
                              stop: 3000
                          },
                'test'  : !obj:jfsantos.timit_dataset.TimitPhoneData {
                              datapath: '/Users/alexis/university/ift6266/data/timit/raw/TIMIT',
                              framelen: 1000,
                              overlap: 800,
                              start: 3000,
                              stop: 4000
                          }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "valid_y_misclass"
        }
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: "results/mlp_soft_1000_Frames_800_overlap.pkl"
        },
    ]
}"""

train = yaml_parse.load(train)
train.main_loop()

model = serial.load('results/mlp_soft_1000_Frames_800_overlap.pkl' )
model.set_batch_size(1)
X = model.get_input_space().make_batch_theano()
Y = model.fprop(X)
f = theano.function( [X], Y )
y = f(t_te.X )

classified = []
for k, u in enumerate(y):
    m = u.max()
    max_u = [i for i, j in enumerate(u) if j == m]
    max_te = [i for i, j in enumerate(t_te.y[k]) if j == 1]
    classified.append(max_u == max_te)


classified = np.array(classified)
print 'classified correctly {}'.format(classified[classified].shape)
print 'missclassified {}'.format(classified[classified == False].shape)