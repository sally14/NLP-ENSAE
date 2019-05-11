"""

                        Utils

Utils, helps generating dataset, etc.

"""

import os
# import sys

import numpy as np
import gensim
from glob import glob

# sys.path.append(os.path.expanduser('~/nlp'))


def generate_dataset(filedir, n):
    """
    Generates dataset for the Masked Language task.
    Args
        filedir : a directory of files on which datasets must be
                    generated
        n : the number of token which must be masked per sentence
    Returns
        None

    Writes new file.sent, file.labels in a subdir 'mlm_dataset'.
    """

    # Retrieving list of filenames with glob:
    filenames = glob(os.path.join(filedir, '*'))
    # Getting dirname for writing operations:
    dirname = os.path.dirname(filenames[0])
    # Creating new subdir in which files will be written
    write_path = os.path.join(dirname, 'mlm_dataset')
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Start iterating on files:
    for file in filenames:
        basename = os.path.basename(file)
        with open(file, 'r', encoding='utf-8') as f:
            sents = f.readlines()  # In f, there is 1 sent per line

        sent_basename = os.path.join(write_path, basename+'.sent')
        label_basename = os.path.join(write_path, basename+'.labels')
        # Now that we have sents, mask n random words
        with open(sent_basename, 'w', encoding='utf-8') as s_write:
            with open(label_basename, 'w', encoding='utf-8') as l_write:
                for s in sents:
                    tokens = s.rstrip('\n').split(' ')
                    m = len(tokens)
                    if m <= n:
                        pass
                    else:
                        mask = np.random.choice(range(m),
                                                size=n,
                                                replace=False)
                        for i in range(m):
                            if i in mask:
                                s_write.write('<MASK> ')
                                l_write.write(str(tokens[i])+' ')
                            else:
                                s_write.write(str(tokens[i])+' ')
                        s_write.write('\n')
                        l_write.write('\n')
    return None


def create_google_embeddings(vocab, google_dir, write_dir):
    """
    Creates an embeddings.tsv, metadata.tsv file with vocabulary 'vocab'
    containing the Google word vectors
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(google_dir+'/GoogleNews-vectors-negative300.bin', binary=True)

    inv_vocab = {v: k for k, v in vocab.items()}
    METADATA_PATH = os.path.join(write_dir, 'metadata.tsv')
    VECTOR_PATHS = os.path.join(write_dir, 'embeddings.tsv') 
    with open(METADATA_PATH, 'w', encoding='utf-8') as metadata:
        with open(VECTOR_PATHS, 'w', encoding='utf-8') as vectors:
            metadata.write('WORD\tINDEX\n')
            for i in range(len(vocab)):
                try:
                    vector = model.wv[inv_vocab[i]]
                    metadata.write(str(inv_vocab[i])+'\t'+str(i)+'\n')
                    n = len(vector)
                    for j in range(n):
                        if j==(n-1):
                            vectors.write(str(vector[j])+'\n')
                        else:
                            vectors.write(str(vector[j])+'\t')
                except:
                    print('{0} not in vocabulary. Passing. \n'.format(inv_vocab[i]))
                    pass


def extract_vocabulary(train):
    """
    Extracts unique words and create an index dictionnary for the vocab.
    """
    with open(train, 'r', encoding='utf-8') as t:
        text = t.read()
    unique_tokens = list(set(text.replace('\n', '').split(' ')))
    vocab = {unique_tokens[i]: i for i in range(len(unique_tokens))}
    return vocab


def create_lookups(train, dirname, google_dir):
    """
    Creates vocab, metadata
    """
    vocab = extract_vocabulary(train)
    create_google_embeddings(vocab, google_dir, dirname)

