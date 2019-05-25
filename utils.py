"""

                        Utils

Utils, helps generating dataset, etc.

"""

import os

import numpy as np
from glob import glob
from nltk import FreqDist
from nltk.util import ngrams


def generate_dataset(filedir, logdir, n_grams, mode="lm", **kwargs):
    """
    Generates dataset for the Masked Language / LM task.
    Args
        filedir : a directory of files on which datasets must be
                    generated
        mode : masked language model (mlm) or classical language model (lm)
        n : the number of token which must be masked per sentence (mlm mode
            only)
    Returns
        None

    Writes new file.sent, file.labels in a subdir 'mlm_dataset'.
    """
    if mode == "mlm":
        n = kwargs["n"]
    # Retrieving list of filenames with glob:
    filenames = glob(os.path.join(filedir, "*"))
    # Getting dirname for writing operations:
    # dirname = os.path.dirname(os.path.dirname(filenames[0]))
    # Creating new subdir in which files will be written
    write_path = logdir

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Start iterating on files:
    for file in filenames:
        basename = os.path.basename(file)
        with open(file, "r", encoding="utf-8") as f:
            sents = f.readlines()  # In f, there is 1 sent per line
            if mode == "mlm":
                write_mlm(write_path, basename, sents, n)
            else:
                write_lm(write_path, basename, sents)
                if n_grams:
                    write_ngrams(write_path, basename, sents, n=5)
    return None


def write_mlm(write_path, basename, sents, n):
    sent_basename = os.path.join(write_path, basename + ".sents")
    label_basename = os.path.join(write_path, basename + ".labels")
    # Now that we have sents, mask n random words / generate lm model
    with open(sent_basename, "w", encoding="utf-8") as s_write:
        with open(label_basename, "w", encoding="utf-8") as l_write:
            for s in sents:
                tokens = s.rstrip("\n").split(" ")
                m = len(tokens)
                if m <= n:
                    print("m<n, passing.")
                    pass
                else:
                    mask = np.random.choice(range(m), size=n, replace=False)
                    for i in range(m):
                        if i in mask:
                            s_write.write("<MASK> ")
                            l_write.write(str(tokens[i]) + " ")
                        else:
                            s_write.write(str(tokens[i]) + " ")
                    s_write.write("\n")
                    l_write.write("\n")
    return None


def write_lm(write_path, basename, sents):
    sent_basename = os.path.join(write_path, basename + ".sents")
    label_basename = os.path.join(write_path, basename + ".labels")
    # Now that we have sents, mask n random words / generate lm model
    with open(sent_basename, "w", encoding="utf-8") as s_write:
        with open(label_basename, "w", encoding="utf-8") as l_write:
            for s in sents:
                tokens = s.rstrip("\n").split(" ")
                m = len(tokens)
                for i in range(1, m):
                    s_write.write(" ".join(tokens[:i]))
                    l_write.write(tokens[i])
                    s_write.write("\n")
                    l_write.write("\n")

    return None


def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


def write_ngrams(write_path, basename, sents, n=5):
    sents = list(map(lambda x: x.rstrip('\n')+, sents))
    sents = ' '.join(sents)
    tokens = sents.split(' ')
    n_grams = word_grams(tokens, min=n, max=n+1)
    assert len(n_grams) > 0, 'No ngrams generated'
    sent_basename = os.path.join(write_path, basename + ".sents")
    label_basename = os.path.join(write_path, basename + ".labels")
    # Now that we have sents, mask n random words / generate lm model
    with open(sent_basename, "a", encoding="utf-8") as s_write:
        with open(label_basename, "a", encoding="utf-8") as l_write:
            for i in range(len(n_grams)-1):
                s_write.write(n_grams[i] + '\n')
                l_write.write(n_grams[i+1].split(' ')[-1]+'\n')


def create_full_vocab(train, emb):
    with open(train, 'r', encoding='utf-8') as f:
        text = f.read().replace('\n', '')
    tokens = text.split(' ')
    vocab_words = FreqDist(tokens)
    vocab_chars = set(list(text+'</s>'))
    count = len(tokens)
    with open(os.path.join(emb, 'vocab.txt'), 'w', encoding='utf-8') as f:
        with open(os.path.join(emb, 'freq.txt'), 'w', encoding='utf-8') as freq:
            for i in vocab_words.keys():
                f.write(i + '\n')
                #print(vocab_words[i])
                freq.write(str(int(vocab_words[i])/count)+'\n')
    with open(os.path.join(emb, 'chars.txt'), 'w', encoding='utf-8') as c:
        for i in vocab_chars:
            c.write(i+'\n')



