from pandas import DataFrame, Series
import pandas as pd
from rawFileParser import Parser
import glob

def get_prompts(file='raw/TIMIT/DOC/PROMPTS.TXT'):

    try:
        fp = open(file)
    except Exception, e:
        print "\033[1,31mCould not open the file\033[0m"
        raise e

    prompts = []
    names = []
    for line in fp.readlines():
        if line.startswith(';'):
            continue
        prompt, name = line.split(' (')
        name = name.strip(')\n')
        prompts.append(prompt)
        names.append(name)

    df = DataFrame(prompts, index=names, columns=['prompts'])
    return df


def get_timit_dictionnary(file='raw/TIMIT/DOC/TIMITDIC.TXT'):

    try:
        fp = open(file)
    except Exception, e:
        print "\033[1,31mCould not open the file\033[0m"
        raise e

    words = []
    phonetics = []
    for line in fp.readlines():
        if line.startswith(';'):
            continue
        word, phonetic = line.split(' /')
        phonetic = phonetic.strip('/\n')
        phonetic.split()
        words.append(word)
        phonetics.append(phonetic)

    df = DataFrame(phonetics, index=words, columns=['phonetic'])
    return df


def get_speakers_info(file='raw/TIMIT/DOC/SPKRINFO.TXT'):
    """
    From the speaker info file, the info of each speaker is parsed. 

    Param
        file: path to SPKRINFO.TXT. Default to raw/TIMIT/DOC/SPKRINFO.TXT

    Return
        DataFrame with the speaker id as the index and all their available info as columns
    """   
    try:
        fp = open(file)
    except Exception, e:
        print "\033[1,31mCould not open the file\033[0m"
        raise e

    ids = []
    infos = []
    for line in fp.readlines():
        if line.startswith(';'):
            continue
        
        line = line.split()
        id = line [1] + line[0]
        info = line[1:9]
        comment = " ".join(line[9:])
        info.append(comment)
        ids.append(id)
        infos.append(info)

    df = DataFrame(infos, index=ids, columns=['Sex', 'DR', 'Use',  'RecDate',   'BirthDate',  'Ht',    'Race', 'Edu',  'Comments'])
    return df


def get_speakers_sentences(file='raw/TIMIT/DOC/SPKRSENT.TXT'):
    """
    From the speaker sentence file, the id of each sentence they spoke is parsed

    Param
        file: path to SPKRSENT.TXT. Default to raw/TIMIT/DOC/SPKRSENT.TXT

    Return
        DataFrame with the speaker id as the index and the id of the ten sentences they spoke as columns
    """
    
    try:
        fp = open(file)
    except Exception, e:
        print "\033[1,31mCould not open the file\033[0m"
        raise e

    ids = []
    sentences = []
    for line in fp.readlines():
        if line.startswith(';'):
            continue
        
        line = line.split()
        id = line[0]
        sentence = line[1:]
        sentence[0] = 'sa' + sentence[0]
        sentence[1] = 'sa' + sentence[1]
        sentence[2] = 'sx' + sentence[2]
        sentence[3] = 'sx' + sentence[3]
        sentence[4] = 'sx' + sentence[4]
        sentence[5] = 'sx' + sentence[5]
        sentence[6] = 'sx' + sentence[6]
        sentence[7] = 'si' + sentence[7]
        sentence[8] = 'si' + sentence[8]
        sentence[9] = 'si' + sentence[9]
        ids.append(id)
        sentences.append(sentence)

    df = DataFrame(sentences, index=ids)
    return pd.concat({'SA':df.ix[:,0:1], 'SX':df.ix[:,1:7], 'SI':df.ix[:,7:10]}, axis=1)


def get_data_for_speaker(speaker, file_type):
    """
    This method is mainly meant to be called after getting the speakers with get_speakers_info.
    Parse each line of the resulting dataframe into this method to get whatever file type
    you want.
    Param
        speaker: pandas Series object containing the dialect region of the speaker and his id
            e.g.
                Sex                 M
                DR                  6
                Use               TRN
                RecDate      03/03/86
                BirthDate    06/17/60
                Ht              5'11"
                Race              WHT
                Edu                BS
                Comments             
                Name: MABC0, dtype: object
    Return
        List of DataFrames containing the phonemes for each sentences the speaker recorded

    TODO
        Possibly support more format for speaker. It may be slow to iterate over a DataFrame
        and use each row (that are Series) for this method.
    """
    if type(speaker) != Series:
        raise Exception("The datastructure provided must be a Series")

    speaker_id = speaker.name
    dialect = speaker['DR']
    path = "/Users/alexis/university/ift6266/data/timit/raw/TIMIT/TRAIN/DR{0}/{1}".format(dialect, speaker_id)

    if file_type == "PHN":
        path += "/*.PHN"
    elif file_type == "TXT":
        path += "/*.TXT"
    elif file_type == "WRD":
        path += "/*.WRD"
    else:
        raise Exception("File format not supported.")

    file_paths = glob.glob(path)

    data = []
    for file_path in file_paths:
        parser = Parser(file_path)
        data.append(parser.get_data())

    if file_type == "PHN":
        return data
    elif file_type == "TXT":
        return pd.concat(data)
    elif file_type == "WRD":
        return data
