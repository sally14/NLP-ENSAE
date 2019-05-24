"""
                                Training Language Model

This script trains/train and evaluate a word-level
language model, implementation with Tensorflow 1.13, tf.Estimator API

Usage:
  train.py <filepath> <log_dir>
  [--gpu_train=<gpu>] [--mode=<md>]
  [--embedding_size=<ems>] [--char_emb_size=<em>]
  [--num_LSTM_layers=<nmlstm>] [--num_layers_char=<nmchar>]
  [--hidden_size_chars=<hsc>]  [--hidden_size_LSTM=<hsn>]
  [--add_char_emb=<add>] [--num_heads=<nmh>]
  [--dropout=<dp>]
  [--learning_rate=<lr>] [--batch_size=<bs>]
  [--n_epochs=<ne>] [--optimizer=<op>] [--checkpoints=<ckpt>]
  [--buffer_size=<bfs>]
  [--deepness_finish=<dpsf>] [--activation_finish=<actf>]
  [--intern_size=<intf>] [--weighted_loss=<wgth>]
  [--add_encoder=<addenc>]

Options:
  -h --help
  --version
  --filepath                 File directories, where the training, and eventually testing files are stored, files to predict in case mode=predict
  --log_dir                  Logs directory, where the checkpoints will be stored
  --gpu_train=<gpu>          Boolean, True is using a gpu. (Requires tensorflow-gpu) [default: True]
  --mode=<md>                The tensorflow mode, in ['train', 'train_eval', 'eval'] [default: train_eval]
  --embedding_size=<ems>     The word embedding dimension [default: 300]
  --char_emb_size=<em>       The chosen character embedding size [default: 100]
  --num_LSTM_layers=<nmlstm> The number of LSTM units [default:2]
  --num_layers_char=<nmchar> The number of LSTM units for char embedding [default:2]
  --hidden_size_chars=<hsc>  The hidden size of the character embedding Bi-LSTM. [default: 25]
  --hidden_size_LSTM=<hsn>   The hidden size of the LSTM. [default: 600]
  --add_char_emb=<add>       Boolean, True if char embedding is needed [default: True]
  --num_heads=<nmh>          The number of head for the attention encoder layer [default: 10]
  --dropout=<dp>             The dropout value. [default: 0.5]
  --learning_rate=<lr>       The learning rate for the optimizer [default: 1e-3]
  --batch_size=<bs>          The batch size for the training [default: 24]
  --n_epochs=<ne>            Number of epochs to train the network on [default: 100]
  --optimizer=<op>           The chosen tf optimizer, in ['sgd', 'adam', 'adagrad', 'rmsprop'] [default: sgd]
  --checkpoints=<ckpt>       Save checkpoints every ckpt steps [default: 5000]
  --buffer_size=<bfs>        Buffer size for shuffling  [default: 500]
  --deepness_finish=<dpsf>   The layer depth for the dense finish [default: 2]
  --activation_finish=<actf> The activation function for the dense finish [default: leaky_relu]
  --intern_size=<intf>       The internal dimension of finish dense layers [default: 3000]
  --weighted_loss=<wgth>     Boolean, indicates if loss must be weighted or not [default: True]
  --add_encoder=<addenc>     Boolean, indicates if encodre must be kept [default: True]
"""


import os
import logging

import tensorflow as tf
from docopt import docopt
from functools import partial
from numpy import genfromtxt
from glob import glob

from utils import generate_dataset, create_full_vocab
from reader import input_fn_gen
from model import model_fn

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError


if __name__ == "__main__":
    # Get the arguments
    args = docopt(__doc__, version="0.3")
    # print(args)
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
                try:
                    params[k2] = str_to_bool(args[k])
                except:
                    params[k2] = args[k]
    print(params)
    if params['num_LSTM_layers'] is None or params['num_layers_char'] is None:
        params['num_LSTM_layers'] = 2
        params['num_layers_char'] = 2
    # Checking if logdir already has the emb files:
    params['embedding_path'] = os.path.join(params['log_dir'], 'embedding')
    if not os.path.isdir(params['log_dir']):
        os.makedirs(params['log_dir'])
    if not os.path.isdir(params['embedding_path']):
        os.makedirs(params['embedding_path'])

    # Path to vocabs for indexing
    params["word_emb_vocab"] = os.path.join(
        params["embedding_path"], "vocab.txt"
    )
    params["char_vocab"] = os.path.join(
        params["embedding_path"], "chars.txt"
    )
    cond = os.path.isfile(params["word_emb_vocab"]) and os.path.isfile(params["char_vocab"])
    if not cond:
        train_path = glob(os.path.join(params['filepath'], '*train*'))[0]
        create_full_vocab(train_path, params['embedding_path'])

    # Create dataset files if not already done:
    params['data_path'] = os.path.join(params['log_dir'], 'dataset_lm')
    if not os.path.isdir(params['data_path']):
        generate_dataset(
              filedir=params['filepath'],
              mode='lm',
              logdir=params['data_path'])


    # Get nb_chars, nb_labels, & nb_words for params (used in model):
    with open(params["word_emb_vocab"], "r", encoding="utf-8") as f:
        lines = f.readlines()
        params["vocab_size"] = len(lines)
        params["max_len_sent"] = max(map(len, lines))
    with open(params["char_vocab"], "r", encoding="utf-8") as f:
        params["nb_chars"] = len(f.readlines())

    params['frequencies'] = genfromtxt(os.path.join(
                                  params['embedding_path'],
                                  'freq.txt'),
                                delimiter="\t")
    params['activation_finish'] = tf.nn.leaky_relu
    # Create input functions
    input_fn = partial(input_fn_gen, mode='train', params=params)
    input_eval = partial(input_fn_gen, mode='eval', params=params)

    # Session config for tensorflow
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True
    )
    config.gpu_options.allow_growth = True

    config = (
        tf.estimator.RunConfig(save_checkpoints_steps=params["checkpoints"])
        .replace(save_summary_steps=1)
        .replace(session_config=config)
    )
    print(params)
    # Generate estimator object
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params["log_dir"],
        params=params,
        config=config,
    )

    if params["mode"] == "train":
        train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
        estimator.train(input_fn)

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

