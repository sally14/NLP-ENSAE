"""

"""

import tensorflow as tf

from submodels import CharacterEmbedding, LSTMSequenceEmbedding
from submodels import EncoderLayer
from submodels import FinishDense
from submodels import LossConstructor


def model_fn(features, labels, mode, params):
    # Get the params
    # Hardware config
    gpu_train = params['gpu_train']

    # Learning params
    dropout = params["dropout"]
    lr = params["learning_rate"]
    optim = params["optimizer"]

    # Embedding params
    vocab_size = params["vocab_size"]
    embedding_size = params['embedding_size']

    # LSTM params
    num_LSTM_layers = params["num_LSTM_layers"]
    hidden_size_LSTM = params["hidden_size_LSTM"]

    # Char embedding parameters
    add_chars_emb = params['add_char_emb']
    if add_chars_emb:
        num_layers_chars = params['num_layers_char']
        hidden_size_chars = params['hidden_size_chars']
        nb_chars = params['nb_chars']

    # Add encoder:
    add_encoder = params['add_encoder']

    # TODO : positionnal embedding parameters

    # Encoder parameters
    num_heads = params['num_heads']

    # Finish parameters
    deepness_finish = params['deepness_finish']
    activation_finish = params['activation_finish']
    inter_size = params['intern_size']

    # Loss parameters
    weighted_loss = params['weighted_loss']
    frequencies = params['frequencies']

    # Get the inputs
    (input_sentences, input_char), seq_length = features
    training = mode == tf.estimator.ModeKeys.TRAIN

    # First embed words
    W2V_embedding = tf.get_variable(
                        name='W2V_embedding',
                        shape=[vocab_size+1, embedding_size],
                        dtype=tf.float32,
                        trainable=True
                        )
    word_embeddings = tf.nn.embedding_lookup(
                        W2V_embedding,
                        input_sentences,
                        name="W2V_embedded"
                        )

    # Depending on parameters, do a character embedding
    if add_chars_emb:
        char_emb = CharacterEmbedding(
            num_layers=num_layers_chars,
            dim_hidden_state=hidden_size_chars,
            nb_chars=nb_chars+1,
            gpu_train=gpu_train,
            is_training=training,
            dropout=dropout)
        chars_embedded = char_emb(inputs=input_char)
        word_embeddings = tf.concat([word_embeddings, chars_embedded], axis=-1)

    word_embeddings = tf.nn.dropout(word_embeddings, dropout)
    if add_chars_emb:
        embedding_shape = embedding_size + 2*hidden_size_chars
    else:
        embedding_shape = embedding_size

    # Encoder phase :
    if add_encoder:
        sample_encoder_layer = EncoderLayer(embedding_shape,
                                            num_heads, 2048)
        out_sequence = sample_encoder_layer(word_embeddings, False, None)
    else:
        out_sequence = word_embeddings

    # LSTM sequence embedding
    seq_emb = LSTMSequenceEmbedding(
                num_layers=num_LSTM_layers,
                dim_hidden_state=hidden_size_LSTM,
                gpu_train=gpu_train,
                is_training=training,
                dropout=dropout)
    output = seq_emb(out_sequence)

    dense = FinishDense(
                deepness_finish=deepness_finish,
                activation_finish=activation_finish,
                intern_size=inter_size,
                is_training=training,
                dropout=dropout)
    logits = dense(output, vocab_size+1)

    labels_pred = tf.math.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        top_labels = tf.math.top_k(logits, k=10).indices
        predictions = {"label": labels_pred, "top_k": top_labels}
        export_outputs = {
            "predictions": tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs
        )
    else:
        loss = LossConstructor(
                vocab_size+1,
                weighted_loss=weighted_loss,
                frequencies=frequencies)
        loss_mean, weigthed = loss(logits, labels)
        tvars = tf.trainable_variables()
        if weighted_loss:
            grads, _ = tf.clip_by_global_norm(
                        tf.gradients(weigthed, tvars),
                        20)
        else:
            grads, _ = tf.clip_by_global_norm(
                        tf.gradients(loss_mean, tvars),
                        20)
        ppxl = tf.exp(loss_mean)

        acc = tf.metrics.accuracy(labels=labels,
                                  predictions=labels_pred,
                                  name='accuracy')

        metrics = {'accuracy': acc}
        # For Tensorboard
        for k, v in metrics.items():
            # v[1] is the update op of the metrics object
            tf.summary.scalar(k, v[1])
        tf.summary.scalar('perplexity', ppxl)
        if weighted_loss:
            tf.summary.scalar('weighted_loss', weigthed)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss_mean, eval_metric_ops=metrics
            )
        # 5. Define EstimatorSpecs for TRAIN
        elif mode == tf.estimator.ModeKeys.TRAIN:

            if optim == "adam":
                opt = tf.train.AdamOptimizer(lr)
            elif optim == "adagrad":
                opt = tf.train.AdagradOptimizer(lr)
            elif optim == "sgd":
                opt = tf.train.GradientDescentOptimizer(lr)
            elif optim == "rmsprop":
                opt = tf.train.RMSPropOptimizer(lr)
            
            train_op = opt.apply_gradients(
                        zip(grads, tvars),
                        global_step=tf.train.get_or_create_global_step())

            return tf.estimator.EstimatorSpec(
                mode, loss=loss_mean, train_op=train_op
            )
