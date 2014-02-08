from pandas import DataFrame
import pandas as pd

def get_prompts(file):

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


def get_timit_dictionnary(file):

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


def get_speakers_info(file):
    
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
        id = line[0]
        info = line[1:9]
        comment = " ".join(line[9:])
        info.append(comment)
        ids.append(id)
        infos.append(info)

    df = DataFrame(infos, index=ids, columns=['Sex', 'DR', 'Use',  'RecDate',   'BirthDate',  'Ht',    'Race', 'Edu',  'Comments'])
    return df


def get_speakers_sentences(file):
    
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




