import os
import docs
import pandas as pd
from pandas import DataFrame, MultiIndex
import glob
from scipy.io import wavfile
import numpy as np
from datetime import datetime
import time
import re


class TIMITMultiplexer(object):
    """TIMITMultiplexer is a base class that is the foundation of further preprocessing. Inherit this class
    and modify self.df to your liking.
    """
    def __init__(self, raw_path, sex=False, dialect=False, 
        recdate=False, birthdate=False, height=False, race=False, edu=False):
        
        self.raw_path = raw_path
        self.sex = sex
        self.dialect = dialect
        self.recdate = recdate
        self.birthdate = birthdate
        self.height = height
        self.race = race
        self.edu = edu
        

        self.spkrs_info = docs.get_speakers_info(self.raw_path)
        self.spkrs_id = self.spkrs_info.id.values
        self.spkrs_augmented_id = self.spkrs_info.index.values
        self.spkrs_dialect = self.spkrs_info['DR'].values

        self.acoustic = self.get_acoustic()
        self.df = self.acoustic

        if sex:
            self.one_hot('Sex')
        if dialect:
            self.one_hot('DR')
        if recdate:
            self.date_to_timestamp('RecDate')
        if birthdate:
            self.date_to_timestamp('BirthDate')
        if height:
            self.attach_height()
        if race:
            self.one_hot('Race')
        if edu:
            self.one_hot('Edu')

        super(TIMITMultiplexer, self).__init__()

    def get_acoustic(self):
        acoustic = []
        index = []

        for id, aug_id, dialect in zip(self.spkrs_id, self.spkrs_augmented_id, self.spkrs_dialect):
            path = os.path.join(self.raw_path, 'TIMIT/TRAIN/DR{0}/{1}/*.WAV'.format(dialect, aug_id))
            wav_files_list = glob.glob(path)
            path = os.path.join(self.raw_path, 'TIMIT/TEST/DR{0}/{1}/*.WAV'.format(dialect, aug_id))
            wav_files_list.extend(glob.glob(path))
            
            for wav in wav_files_list:
                fileName, fileExtension = os.path.splitext(wav)
                fileName = fileName.split('/')[-1]
                rate, data = wavfile.read(wav)
                
                acoustic.append(data)
                index.append((aug_id, fileName))
                
        df = DataFrame({'acoustic':acoustic}, index=MultiIndex.from_tuples(index, names=['spkr','sent']))
        return df



    def date_to_timestamp(self, key):
        format = "%m/%d/%y"
        values = []

        for spkr_aug_id in self.spkrs_augmented_id:
            spkr = self.df.ix[spkr_aug_id]
            num_sent = spkr.shape[0]

            dt = self.spkrs_info.ix[spkr_aug_id][key]

            recDate = time.mktime(datetime.strptime(dt,format).timetuple())
            values.extend([recDate] * num_sent)

        self.df[key] = values


    def attach_height(self):
        values = []

        for spkr_aug_id in self.spkrs_augmented_id:
            spkr = self.df.ix[spkr_aug_id]
            num_sent = spkr.shape[0]

            ht = self.spkrs_info.ix[spkr_aug_id]['Ht']
            m = re.split('[\'\"]', ht)
            inches = int(m[0]) * 12 + int(m[1])
            for num in range(num_sent):
                values.append([inches])

        self.df['Ht'] = values


    def one_hot(self, key):
        unique_values = list(self.spkrs_info[key].unique())
        values = []
        for spkr_aug_id in self.spkrs_augmented_id:
            spkr = self.df.ix[spkr_aug_id]
            num_sent = spkr.shape[0]

            dr = self.spkrs_info.ix[spkr_aug_id][key]
            one_hot = np.zeros(len(unique_values))
            index = unique_values.index(dr)
            one_hot[index] = 1

            for num in range(num_sent):
                values.append(one_hot)


        self.df[key] = values

