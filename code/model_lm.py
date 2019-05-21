"""
                                Model for language model

This script defines model_fn function for the LM, implementation with
Tensorflow 1.13, tf.Estimator


Model :

Pre-training task : Masked LM

            | Word2Vec |    | Char Embedding |
                 |                  |
                |     Embedding      |
                          |
                       Bi-LSTMs
"""

import tensorflow as tf


def model_fn(features, labels, mode, params):
    # Get the params
    batch_size = params["batch_size"]
    char_emb_size = params["char_emb_size"]
    nb_chars = params["nb_chars"]
    hidden_size_char = params["hidden_size_char"]
    hidden_size_NER = params["hidden_size_NER"]
    dropout = params["dropout"]
    W2Vembedding = params["embedding"]
    # nb_tags = params['nb_tags']
    lr = params["learning_rate"]
    optim = params["optimizer"]
    vocab_size = params["vocab_size"]

    # Get the inputs
    # input_sentences, input_char, seq_length = features
    (input_sentences, input_char), seq_length = features
    training = mode == tf.estimator.ModeKeys.TRAIN

    # First embed words

    _W2V_embedding = tf.Variable(
        W2Vembedding, dtype=tf.float32, trainable=False
    )
    W2V_embedded = tf.nn.embedding_lookup(
        _W2V_embedding, input_sentences, name="W2V_embedded"
    )
    # Then Bi-LSTM for chars:
    _char_embedding = tf.get_variable(
        name="char_embedding",
        dtype=tf.float32,
        shape=[nb_chars, char_emb_size],
    )
    char_embedded = tf.nn.embedding_lookup(
        _char_embedding, input_char, name="char_embedded"
    )

    # put the time dimension on axis=1
    s = tf.shape(char_embedded)
    char_embedded = tf.reshape(
        char_embedded, shape=[s[0] * s[1], s[-2], char_emb_size]
    )

    # Bi-LSTM
    LSTM = tf.contrib.cudnn_rnn.CudnnLSTM(
        1, hidden_size_char, direction="bidirectional"
    )
    outputs, output_states = LSTM(tf.transpose(char_embedded, [1, 0, 2]))
    # tf.transpose = CudnnLSTM handles differently time axis
    output_fw = output_states[0][1]
    output_bw = output_states[1][1]
    output = tf.concat([output_fw, output_bw], axis=-1)
    # shape = (batch size, max sentence length, char hidden size)
    output = tf.reshape(output, shape=[s[0], s[1], 2 * hidden_size_char])
    word_embeddings = tf.concat([W2V_embedded, output], axis=-1)

    word_embeddings = tf.nn.dropout(word_embeddings, dropout)

    # NER
    # Bi-LSTM

    LSTM = tf.contrib.cudnn_rnn.CudnnLSTM(
        1, hidden_size_NER, direction="bidirectional"
    )
    outputs, output_states = LSTM(tf.transpose(word_embeddings, [1, 0, 2]))
    output = outputs
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    logits = tf.layers.dense(output, vocab_size)

    labels = tf.one_hot(labels, depth=vocab_size)

    loss = tf.losses.softmax_cross_entropy(labels, logits)
    

    if mode == tf.estimator.ModeKeys.PREDICT:

        # predictions = {"label": labels_pred, "logits": logits}
        # export_outputs = {
        #     "predictions": tf.estimator.export.PredictOutput(predictions)
        # }
        # return tf.estimator.EstimatorSpec(
        #     mode, predictions=predictions, export_outputs=export_outputs
        # )
        pass
    else:
        loss = tf.reduce_mean(loss)
        ppxl = tf.exp(loss)
        tf.summary.scalar(ppxl)
        # weights = tf.sequence_mask(seq_length)
        # # weights_flat = tf.reshape(labels, [bs[0]*m])
        # acc = tf.metrics.accuracy(labels=labels,
        #                           predictions=labels_pred,
        #                           weights=weights,
        #                           name='accuracy')
        # prec = tf.metrics.precision(labels=labels,
        #                             predictions=labels_pred,
        #                             weights=weights,
        #                             name='precision')
        # rec = tf.metrics.recall(labels=labels,
        #                         predictions=labels_pred,
        #                         weights=weights,
        #                         name='recall')
        # # op = tf.div(2*tf.matmul(prec[1], rec[1]), tf.add(prec[1], rec[1]))
        # f1 = [0, 2*prec[0]*rec[0]/(prec[0]+rec[0])]
        # print(f1[0])
        # metrics = {'accuracy': acc,
        #            'precision' : prec,
        #            'recall' : rec,
        #            'f1' : f1}
        # For Tensorboard
        # for k, v in metrics.items():
        #     # v[1] is the update op of the metrics object
        #     tf.summary.scalar(k, v[1])
        if mode == tf.estimator.ModeKeys.EVAL:
            # metrics = {'accuracy': acc,
            #             'precision' : prec,
            #             'recall' : rec}
            # return tf.estimator.EstimatorSpec(
            #     mode, loss=loss, eval_metric_ops=metrics
            # )
            pass

        # 5. Define EstimatorSpecs for TRAIN
        elif mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            epoch = global_step // batch_size
            tf.summary.scalar("epoch", epoch)
            if optim == "adam":
                opt = tf.train.AdamOptimizer(lr)
            elif optim == "adagrad":
                opt = tf.train.AdagradOptimizer(lr)
            elif optim == "sgd":
                opt = tf.train.GradientDescentOptimizer(lr)
            elif optim == "rmsprop":
                opt = tf.train.RMSPropOptimizer(lr)
            train_op = opt.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op
            )
