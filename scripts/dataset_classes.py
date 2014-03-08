from pylearn2.datasets import DenseDesignMatrix
from jfsantos.timit_full import TimitFullCorpusReader
import numpy as np
import itertools
import os
import numpy as np
from TIMITMultiplexer import TIMITMultiplexer


class FrameData(DenseDesignMatrix):
    
    def __init__(self, framelen=0, overlap=0, start=0, stop=None, sex=False, dialect=False, race=False, edu=False, recdate=False, height=False):
        t_mult = TIMITMultiplexer('/Users/alexis/university/ift6266/data/timit/raw', sex=sex, dialect=dialect, race=race, edu=edu, recdate=recdate, height=height)

        samples = []
        for index in t_mult.df.index:
            s = t_mult.df.ix[index]
            acoustic = s['acoustic']
            if s.shape[0] > 1:
                rest = np.concatenate(s[1:].values)
            else:
                rest = np.array([])

            begin = 0
            jump = framelen - overlap
            for end in range(framelen, len(acoustic), jump):
                concat = np.concatenate((rest, acoustic[begin:end]))
                samples.append(concat)
                begin += jump

        samples = np.array(samples)

        if stop is None:
            X = samples[start:,:-1]
            y = samples[start:,-1]
        else:
            X = samples[start:stop,:-1]
            y = samples[start:stop,-1]

        y = y.reshape(y.shape[0], 1)

        super(FrameData,self).__init__(X=X, y=y)


class FrameDataB(DenseDesignMatrix):
    
    def __init__(self, framelen=0, overlap=0, start=0, stop=None):
        filename = "FrameData_{}_{}.npy".format(framelen, overlap)
        filepath = os.path.join('saved_dataset', filename)
        if os.path.exists(filename):
            samples = np.load(filename)
        else:
            t_mult = TIMITMultiplexer('/Users/alexis/university/ift6266/data/timit/raw')

            samples = []
            for index in t_mult.df.index:
                s = t_mult.df.ix[index]
                acoustic = s['acoustic']
                rest = np.concatenate(s[1:].values)
                
                start = 0
                jump = framelen - overlap
                for end in range(framelen, len(acoustic), jump):
                    samples.append(np.concatenate((rest, acoustic[start:end])))
                    start += jump

            samples = np.array(samples)
            np.save(open(filename,'wb'), samples)

        if stop is None:
            X = samples[start:,:-1]
            y = samples[start:,-1]
        else:
            X = samples[start:stop,:-1]
            y = samples[start:stop,-1]
        
        y = y.reshape(y.shape[0], 1)
        super(FrameData,self).__init__(X=X, y=y)



class TimitDataA(DenseDesignMatrix):
    """
    Dataset with frames and corresponding one-hot encoded
    phones.
    """
    def __init__(self, datapath, framelen, overlap, start=0, stop=None):
        """
        datapath: path to TIMIT raw data (using WAV format)
        framelen: length of the acoustic frames
        overlap: amount of acoustic samples to overlap
        start: index of first TIMIT file to be used
        end: index of last TIMIT file to be used
        """
        data = TimitFullCorpusReader(datapath)
        print datapath, framelen, overlap, start, stop
        # Some list comprehension/zip magic here (but it works!)
        if stop is None:
            self.utterances = data.utteranceids()[start:]
        else:
            self.utterances = data.utteranceids()[start:stop]
        self.spkrfr = [data.frames(z, framelen, overlap) for z in
                  self.utterances]

        self.fr, ph = zip(*[(x[0], x[1]) for x in self.spkrfr])



        X = np.vstack(self.fr)*2**-15
        self.ph = list(itertools.chain(*ph))
        # making y a one-hot output
        one_hot = np.zeros((len(self.ph),len(data.phonelist)),dtype='float32')
        idx = [data.phonelist.index(p) for p in self.ph]
        for i in xrange(len(self.ph)):
            one_hot[i,idx[i]] = 1.
        y = one_hot

        y = np.array([X[:,-1]]).T
        X = X[:,:-1]

        super(TimitDataA,self).__init__(X=X, y=y)


class TimitDataB(DenseDesignMatrix):
    """
    Dataset with frames and corresponding one-hot encoded
    phones.
    """
    def __init__(self, datapath, framelen, overlap, start=0, stop=None):
        """
        datapath: path to TIMIT raw data (using WAV format)
        framelen: length of the acoustic frames
        overlap: amount of acoustic samples to overlap
        start: index of first TIMIT file to be used
        end: index of last TIMIT file to be used
        """
        data = TimitFullCorpusReader(datapath)
        print datapath, framelen, overlap, start, stop
        # Some list comprehension/zip magic here (but it works!)
        if stop is None:
            self.utterances = data.utteranceids()[start:]
        else:
            self.utterances = data.utteranceids()[start:stop]
        self.spkrfr = [data.frames(z, framelen, overlap) for z in
                  self.utterances]

        self.fr, ph = zip(*[(x[0], x[1]) for x in self.spkrfr])



        X = np.vstack(self.fr)*2**-15
        self.ph = list(itertools.chain(*ph))
        # making y a one-hot output
        one_hot = np.zeros((len(self.ph),len(data.phonelist)),dtype='float32')
        idx = [data.phonelist.index(p) for p in self.ph]
        for i in xrange(len(self.ph)):
            one_hot[i,idx[i]] = 1.
        y = one_hot

        train_y = X[:,-1]
        temp_X = []
        for i, r in enumerate(X):
            temp_X.append(np.append(r[:-1], (y[i])))

        y = np.array([train_y]).T
        X = np.array(temp_X)
        super(TimitDataB,self).__init__(X=X, y=y)






