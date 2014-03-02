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


# TimitDataA
train = """!obj:pylearn2.train.Train {
    dataset: &train !obj:dataset_classes.TimitDataA {
        datapath: '/Users/alexis/university/ift6266/data/timit/raw/TIMIT',
        framelen: 1000,
        overlap: 0,
        start: 0,
        stop: 2000
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.mlp.Sigmoid {
                     layer_name: 'h0',
                     dim: 1100,
                     sparse_init: 15,
                 }, !obj:pylearn2.models.mlp.Linear {
                     layer_name: 'y',
                     dim: 1,
                     irange: 10.
                 }
                ],
        nvis: 999,
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 1000000,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        updates_per_batch: 10,
        monitoring_dataset:
            {
                'train' : *train,
                'valid' : !obj:dataset_classes.TimitDataA {
                              datapath: '/Users/alexis/university/ift6266/data/timit/raw/TIMIT',
                              framelen: 1000,
                              overlap: 0,
                              start: 2000,
                              stop: 3000
                          },
                'test'  : !obj:dataset_classes.TimitDataA {
                              datapath: '/Users/alexis/university/ift6266/data/timit/raw/TIMIT',
                              framelen: 1000,
                              overlap: 0,
                              start: 3000,
                              stop: 4000
                          }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "valid_y_range_x_mean_u"
        }
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_range_x_mean_u',
             save_path: "results/mlp_best_tda_1000Frames.pkl"
        },
    ]
}"""


train = yaml_parse.load(train)
train.main_loop()