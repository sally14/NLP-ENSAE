"""
                                Training NER

This script trains/train and evaluate NER, implementation with Tensorflow 1.12, tf.Estimator

Usage:
  NER_train.py <filepath> <embedding_path> <log_dir>
  [--mode=<md>] [--char_emb_size=<em>]
  [--hidden_size_char=<hsc>]  [--hidden_size_NER=<hsn>]
  [--dropout=<dp>]
  [--learning_rate=<lr>] [--batch_size=<bs>]
  [--n_epochs=<ne>] [--optimizer=<op>] [--checkpoints=<ckpt>]
  [--buffer_size=<bfs>]


Options:
  -h --help
  --version
  --filepath                File directories, where the training files are stored, files to predict in case mode=predict
  --embedding_path          The path to embeddings as .tsv 'token \t coord1 \t coord2 ...'
  --log_dir                 Logs directory, where the checkpoints will be stored
  --mode=<md>               The tensorflow mode, in ['train', 'train_eval', 'eval'] [default: train]
  --char_emb_size=<em>      The chosen character embedding size [default: 100]
  --hidden_size_char=<hsc>  The hidden size of the character embedding Bi-LSTM. [default: 25]
  --hidden_size_NER=<hsn>   The hidden size of the general NER Bi-LSTM. [default: 100]
  --dropout=<dp>            The dropout value. [default: 0.5]
  --learning_rate=<lr>      The learning rate for the optimizer [default: 1e-3]
  --batch_size=<bs>         The batch size for the training [default: 24]
  --n_epochs=<ne>           Number of epochs to train the network on [default: 100]
  --optimizer=<op>          The chosen tf optimizer, in ['sgd', 'adam', 'adagrad', 'rmsprop'] [default: sgd]
  --checkpoints=<ckpt>      Save checkpoints every ckpt steps [default:5000]
  --buffer_size=<bfs>       Buffer size for shuffling 
"""

import os
import sys
import time
import logging
from multiprocessing import cpu_count

sys.path.append(os.path.expanduser("./code"))

import numpy as np
from glob import glob
import tensorflow as tf
tf.enable_eager_execution()
from docopt import docopt
from unidecode import unidecode

from model_lm import model_fn

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

if __name__ == "__main__":
    # Get the arguments
    args = docopt(__doc__, version="0.3")
    print(args)
    start = time.time()
    # Transform it into a params dic
    params = {}
    for k in args.keys():
        k2 = k.replace("<", "").replace(">", "").replace("-", "")
        try:  # Convert strings to int or floats when required
            params[k2] = int(args[k])
        except:
            try:
                params[k2] = float(args[k])
            except:
                params[k2] = args[k]

    # Filling additional features in the parameters dict.
    # Path to the .tsv file containing the embeddings
    params["emb_tsv"] = os.path.join(
        params["embedding_path"], "embeddings.tsv"
    )
    # Path to vocabs for indexing
    params["word_emb_vocab"] = os.path.join(
        params["embedding_path"], "vocab.txt"
    )
    params["label_vocab"] = params["word_emb_vocab"]
    params["char_vocab"] = os.path.join(
        params["embedding_path"], "chars_vocab.txt"
    )

    # Loading .tsv file containing the embeddings:
    params["embedding"] = np.genfromtxt(params["emb_tsv"], delimiter="\t")
    logging.info("Loaded embeddings")

    # Get nb_chars, nb_labels, & nb_words for params (used in model):
    with open(params["word_emb_vocab"], "r", encoding="utf-8") as f:
        lines = f.readlines()
        params["vocab_size"] = len(lines)
        params["max_len_sent"] = max(map(len, lines))
    with open(params["label_vocab"], "r", encoding="utf-8") as f:
        params["nb_tags"] = len(f.readlines())
    with open(params["char_vocab"], "r", encoding="utf-8") as f:
        params["nb_chars"] = len(f.readlines())

    cores = cpu_count()

    def lower_tensor(tens):
        lower = tf.py_func(
            lambda x: x.lower(), tens, tf.string, stateful=False
        )
        return lower

    def decode_tensor(tens):
        lower = tf.py_func(
            lambda x: x.lower(), tens, tf.string, stateful=False
        )
        return lower

    def extract_char(token, default_value="<pad_char>"):
        # Split characters
        out = tf.string_split(token, delimiter="")
        # Convert to Dense tensor, filling with default value
        out = tf.sparse_tensor_to_dense(out, default_value=default_value)
        return out

    def extract_char_V2(tens):
        out = tens.map(lambda x: extract_char(x))
        return out

    # INPUT FUNCTIONS
    def input_fn():
        """
        input_fn
            Generates a dataset to be fed to the tf.Estimator model
            As the input_fn function cannot take arguments in tf.Estimator,
            it is assumed that parameters are fixed while running the script,
            declared outside the function. One could have curryfied the
            function in order to be cleaner, but I have choosen to keep
            it simple for readability.
        Args
            None

        Returns
            dataset
                A tf.dataset object which yields
                ((dataset_words, dataset_chars), dataset_labels)
                where :
                - dataset_words is a [batch_size, max_l] tensor (max_l= maximum sentence
                length over the current batch). Words are NOT embedded before entering
                the model, their are encoded as their index in the vocabulary.
                - dataset_chars is a [batch_size, max_l, max_l_char] tensor. (max_l 
                defined as before, max_l_char = max length of a word in terms
                of characters over the current batch). Chars are embedded as their 
                index in the character vocabulary.
                - dataset_labels is a [batch_size, max_l] tensor containing
                the labels ids for each words for each sentence in the batch.
        """
        # Get the important parameters to generate datasets
        batch_size = params["batch_size"]
        repeat_ = params["n_epochs"]
        buffer_size = params["buffer_size"]

        # Get the files list:
        sent_files = glob(os.path.join(params["filepath"], "*.train.txt.sents"))
        label_files = glob(os.path.join(params["filepath"], "*.train.txt.labels"))

        # Tensorflow TextLineDataset reads files, file per file,
        # line per line, and outputs tensors containing the string
        # for each line.
        sent_lines = tf.data.TextLineDataset(
            sent_files, buffer_size=params["buffer_size"]
        )
        label_lines = tf.data.TextLineDataset(
            label_files, buffer_size=params["buffer_size"]
        )
        nb_cores = tf.data.experimental.AUTOTUNE
        # WARNING : custom decode_tensor & lower_tensor functions
        # might have reproductibility issues using tf.save because of pyfunc
        # Encoding forcing through unidecode
        decoded_tokens = sent_lines.map(
            lambda token: decode_tensor([token]), num_parallel_calls=nb_cores
        )
        # Lower words before indexing for tokens,
        # for compatibility with word embeddings
        lowered_tokens = decoded_tokens.map(
            lambda token: lower_tensor([token])
        )
        # Split tokens along whitespace
        dataset_tokens = lowered_tokens.map(
            lambda string: tf.string_split([string], delimiter=" ").values,
            num_parallel_calls=nb_cores,
        )
        # No lowering for characters to keep maximum information,
        # split a first time along whitespaces
        decoded_chars = decoded_tokens.map(
            lambda string: tf.string_split([string], delimiter=" ").values,
            num_parallel_calls=nb_cores,
        )

        dataset_labels = label_lines.map(
            lambda string: tf.string_split([string], delimiter=" ").values,
            num_parallel_calls=nb_cores,
        )
        dataset_chars = decoded_chars.apply(extract_char_V2)

        # Vocabulary, label vocab, char vocab
        words = tf.contrib.lookup.index_table_from_file(
            params["word_emb_vocab"], num_oov_buckets=1
        )
        chars = tf.contrib.lookup.index_table_from_file(
            params["char_vocab"], num_oov_buckets=1
        )

        # Embed words, labels, chars with their indexes in vocabularies.
        dataset_tokens = dataset_tokens.map(
            lambda tokens: words.lookup(tokens), num_parallel_calls=cores
        )
        dataset_seq_length = dataset_tokens.map(
            lambda tokens: tf.size(tokens), num_parallel_calls=cores
        )
        dataset_labels = dataset_labels.map(
            lambda tokens: words.lookup(tokens), num_parallel_calls=cores
        )
        dataset_chars = dataset_chars.map(
            lambda tokens: chars.lookup(tokens), num_parallel_calls=cores
        )

        # Now needs zipping, padding, batching, shuffling
        dataset_input = tf.data.Dataset.zip((dataset_tokens, dataset_chars))
        padded_shapes = (
            tf.TensorShape([None]),  # padding the words
            tf.TensorShape([None, None]),
        )
        padding_values = (
            words.lookup(tf.constant(["<pad_word>"]))[0],
            chars.lookup(tf.constant(["<pad_char>"]))[0],
        )
        dataset_input = dataset_input.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
        )

        # padding the characters for each word
        padded_shapes = tf.TensorShape([None])
        # arrays of labels padded on the right with <pad>

        dataset_seq_length = dataset_seq_length.batch(batch_size)

        dataset = tf.data.Dataset.zip(
            ((dataset_input, dataset_seq_length), dataset_labels)
        )

        # Shuffle the dataset and repeat:
        dataset = dataset.repeat(repeat_)

        return dataset
    
    dataset = input_fn()
    i=0
    for d in dataset:
        i=i+1
        if i>290 and i<310:
            print(d)
        else:
            break
    assert True==False


    # Create configs
    # sess_config = tf.ConfigProto(device_count = {'GPU': 0})
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True
    )
    # config.intra_op_parallelism_threads = 16
    # config.inter_op_parallelism_threads = 16
    config.gpu_options.allow_growth = True
    # distribution = tf.contrib.distribute.MirroredStrategy()

    config = (
        tf.estimator.RunConfig(save_checkpoints_steps=params["checkpoints"])
        .replace(save_summary_steps=1)
        .replace(session_config=config)
    )

    # Generate estimator object
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params["log_dir"],
        params=params,
        config=config,
    )
    # hook = tf.train.ProfilerHook(save_steps=1000, output_dir=params["log_dir"])

    if params["mode"] == "train":
        train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
        estimator.train(input_fn)  #, hooks=[hook])

    elif params["mode"] == "train_eval":
        train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=input_eval, steps=700, throttle_secs=10
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    elif params["mode"] == "eval":
        eval_spec = tf.estimator.EvalSpec(
            input_fn=input_eval, steps=700, throttle_secs=10
        )
        estimator.evaluate(input_eval)
    else:
        # Getting sentences to print a clear, human-readable output
        with open(
            glob(os.path.join(params["filepath"], "*_pred.sent"))[0],
            "r",
            encoding="utf-8",
        ) as pred:
            lines = list(
                map(lambda x: x.rstrip("\n").split(" "), pred.readlines())
            )

        # Getting labels to do this.
        with open(params["label_vocab"], "r", encoding="utf-8") as lab:
            labels = list(map(lambda x: x.rstrip("\n"), lab.readlines()))
            dic_lab = {str(i): labels[i] for i in range(len(labels))}

        for idx, predictions in enumerate(estimator.predict(input_predict)):
            # for k, l in zip(sentences_copy[idx], predictions['label'].tolist()[:len(sentences_copy[idx])]):
            # l = predictions['label'].tolist()#[:len(sentences_copy2[idx])]
            # print(idx, predictions)
            seq = list(predictions["label"])
            acc = ""
            for i in range(len(lines[idx])):
                try:
                    if seq[i] == 0:  # TODO : correct it
                        acc = acc + lines[idx][i] + " "
                    else:
                        acc = (
                            acc
                            + lines[idx][i]
                            + " [{}] ".format(dic_lab[str(seq[i])])
                        )
                except:
                    pass
            print(acc)
            print("\n")
