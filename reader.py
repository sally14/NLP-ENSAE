'''
Reader functions to input data in tf.estimator
'''
import os
import tensorflow as tf
from glob import glob


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


def extract_char(token, default_value='<pad_char>'):
    # Split characters
    out = tf.string_split(token, delimiter='')
    # Convert to Dense tensor, filling with default value
    out = tf.sparse_tensor_to_dense(out, default_value=default_value)
    return out


def extract_char_V2(tens):
    out = tens.map(lambda x: extract_char(x))
    return out


# INPUT FUNCTIONS
def input_fn_gen(mode, params, **kwargs):
    '''
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
    '''
    # Get the important parameters to generate datasets
    buffer_size = params['buffer_size']
    batch_size = params['batch_size']

    def input_fn():
        # Get the files list:
        if mode == 'train':
            sent_files = glob(os.path.join(params['filepath'], '*.train.txt.sents'))
            label_files = glob(os.path.join(params['filepath'], '*.train.txt.labels'))
            # Tensorflow TextLineDataset reads files, file per file,
            # line per line, and outputs tensors containing the string
            # for each line.
            sent_lines = tf.data.TextLineDataset(
                sent_files, buffer_size=buffer_size
            )
            label_lines = tf.data.TextLineDataset(
                label_files, buffer_size=buffer_size
            )
            repeat_ = params['n_epochs']

        elif mode == 'eval':
            sent_files = glob(os.path.join(params['filepath'], '*.test.txt.sents'))
            label_files = glob(os.path.join(params['filepath'], '*.test.txt.labels'))
            # Tensorflow TextLineDataset reads files, file per file,
            # line per line, and outputs tensors containing the string
            # for each line.
            sent_lines = tf.data.TextLineDataset(
                sent_files, buffer_size=buffer_size
            )
            label_lines = tf.data.TextLineDataset(
                label_files, buffer_size=buffer_size
            )
            repeat_ = 1
        elif mode == 'predict':
            sent = kwargs['sent']
            sent_lines = tf.data.Dataset.from_tensor_slices(
                            tf.constant([sent],
                                        dtype=tf.string))
            label_lines = tf.data.Dataset.from_tensor_slices(
                            tf.constant(['useless'],
                                        dtype=tf.string))
            repeat_ = 1

        else:
            print('mode error')

        cores = tf.data.experimental.AUTOTUNE
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
            lambda string: tf.string_split([string], delimiter=' ').values,
            num_parallel_calls=cores,
        )
        # No lowering for characters to keep maximum information,
        # split a first time along whitespaces
        decoded_chars = decoded_tokens.map(
            lambda string: tf.string_split([string], delimiter=' ').values,
            num_parallel_calls=cores,
        )

        dataset_labels = label_lines.map(
            lambda string: tf.string_split([string], delimiter=' ').values,
            num_parallel_calls=cores,
        )
        dataset_chars = decoded_chars.apply(extract_char_V2)

        # Vocabulary, label vocab, char vocab
        words = tf.contrib.lookup.index_table_from_file(
            params['word_emb_vocab'], num_oov_buckets=1
        )
        chars = tf.contrib.lookup.index_table_from_file(
            params['char_vocab'], num_oov_buckets=1
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

        # Now needs zipping,shuffling, unzipping (for padding reasons)
        dataset_input = tf.data.Dataset.zip((dataset_tokens, dataset_chars))
        dataset = tf.data.Dataset.zip(
            ((dataset_input, dataset_seq_length), dataset_labels)
        ).shuffle(buffer_size)

        intermediate = dataset.map(lambda a, b: a)
        dataset_labels = dataset.map(lambda a, b: b)
        dataset_input = intermediate.map(lambda a, b: a)
        dataset_seq_length = intermediate.map(lambda a, b: b)

        padded_shapes = (
            tf.TensorShape([None]),  # padding the words
            tf.TensorShape([None, None]),
        )
        padding_values = (
            words.lookup(tf.constant(['<pad_word>']))[0],
            chars.lookup(tf.constant(['<pad_char>']))[0],
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

        dataset_labels = dataset_labels.batch(batch_size)

        dataset = tf.data.Dataset.zip(
            ((dataset_input, dataset_seq_length), dataset_labels)
        )

        # Shuffle the dataset and repeat:
        dataset = dataset.repeat(repeat_)
        return dataset
    return input_fn()
