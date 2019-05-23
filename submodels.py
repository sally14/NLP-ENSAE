"""
Submodels for the tensorflow graph
"""
import tensorflow as tf


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
          tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
      ])


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead) 
      but it must be broadcastable for addition.

      Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.

      Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth_v)

    return output, attention_weights


class CharacterEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 dim_hidden_state,
                 nb_chars,
                 gpu_train=True,
                 is_training=True,
                 dropout=0):
        super(CharacterEmbedding, self).__init__()
        self.num_layers = num_layers
        self.dim_hidden_state = dim_hidden_state
        self.nb_chars = nb_chars
        if gpu_train:
            self.RNN = tf.contrib.cudnn_rnn.CudnnLSTM(
                            self.num_layers,
                            self.dim_hidden_state,
                            direction="bidirectional"
                        )
        else:
            self.RNN = tf.contrib.rnn.BasicLSTMCell(
                            self.dim_hidden_state,
                            forget_bias=0.0,
                            state_is_tuple=True,
                            reuse=not is_training)
        # if is_training:
        #     self.RNN = tf.contrib.rnn.DropoutWrapper(
        #                     self.RNN,
        #                     output_keep_prob=dropout)

    def call(self, inputs):
        self.char_embedding = tf.get_variable(
                                name="char_embedding",
                                dtype=tf.float32,
                                shape=[self.nb_chars, self.dim_hidden_state],
                                )
        self.char_embedded = tf.nn.embedding_lookup(
                                self.char_embedding,
                                inputs,
                                name="char_embedded"
                                )
        # put the time dimension on axis=1
        s = tf.shape(self.char_embedded)
        self.char_embedded = tf.reshape(
                                self.char_embedded,
                                shape=[s[0] * s[1], s[-2], self.dim_hidden_state]
                                )

        outputs, output_states = self.RNN(tf.transpose(self.char_embedded, [1, 0, 2]))
        # tf.transpose = LSTM handles differently time axis
        self.output_fw = output_states[0][1]
        self.output_bw = output_states[1][1]
        output = tf.concat([self.output_fw, self.output_bw], axis=-1)
        # shape = (batch size, max sentence length, char hidden size)
        self.output = tf.reshape(output, shape=[s[0], s[1], 2 * self.dim_hidden_state])
        return output


class LSTMSequenceEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 dim_hidden_state,
                 gpu_train=True,
                 is_training=True,
                 dropout=0):
        super(LSTMSequenceEmbedding, self).__init__()
        self.num_layers = num_layers
        self.dim_hidden_state = dim_hidden_state
        if gpu_train:
            self.RNN = tf.contrib.cudnn_rnn.CudnnLSTM(
                            self.num_layers,
                            self.dim_hidden_state,
                            direction="bidirectional"
                        )
        else:
            self.RNN = tf.contrib.rnn.BasicLSTMCell(
                            self.dim_hidden_state,
                            forget_bias=0.0,
                            state_is_tuple=True,
                            reuse=not is_training)
        # if is_training:
        #     self.RNN = tf.contrib.rnn.DropoutWrapper(
        #                     self.RNN,
        #                     output_keep_prob=dropout)

    def call(self, inputs):
        outputs, output_states = self.RNN(tf.transpose(inputs, [1, 0, 2]))
        # tf.transpose = LSTM handles differently time axis
        self.output_fw = output_states[0][1]
        self.output_bw = output_states[1][1]
        self.output = tf.concat([self.output_fw, self.output_bw], axis=-1)
        return self.output


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.contrib.layers.layer_norm
        self.layernorm2 = tf.contrib.layers.layer_norm

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class FinishDense(tf.keras.layers.Layer):
    def __init__(self,
                 deepness_finish,
                 activation_finish=tf.nn.leaky_relu,
                 intern_size=300,
                 is_training=True,
                 dropout=0):
        super(FinishDense, self).__init__()
        self.deepness_finish = deepness_finish
        self.activation_finish = activation_finish
        self.is_training = is_training
        self.dropout = dropout

    def call(self, input, vocab_size):
        nb_dense = self.deepness_finish
        for i in range(nb_dense-1):
            input = tf.layers.dense(
                        input,
                        self.intern_size,
                        activation=self.activation_finish)
        output = tf.layers.dense(
                        input,
                        vocab_size)
        return output


class LossConstructor(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 weighted_loss=False,
                 frequencies=None):
        super(LossConstructor, self).__init__()
        """
        frequencies must be a tensor [vocab_size, 1] indicating the frequencies of 
        each word over the training corpus
        """
        self.weighted_loss = weighted_loss
        self.frequencies = frequencies
        self.vocab_size = vocab_size
        if self.weighted_loss:
            self.frequencies = tf.get_variable(
                                self.frequencies,
                                name='frequencies',
                                shape=[self.vocab_size, 1],
                                dtype=tf.float32,
                                trainable=False
                                )
            self.inv_freq = 1/self.frequencies

    def call(self, logits, labels):
        labels_one_hot = tf.squeeze(tf.one_hot(labels, 
                                    depth=self.vocab_size))
        if self.weighted_loss:
            freq_weights = tf.nn.embedding_lookup(
                                self.inv_freq,
                                labels,
                                name="freq_weights"
                                )
            losses = tf.losses.softmax_cross_entropy(labels_one_hot, logits)
            self.weighted_loss = tf.losses.compute_weighted_loss(losses, freq_weights)
        else:
            self.weighted_loss = None
        loss = tf.losses.softmax_cross_entropy(labels_one_hot, logits)
        self.loss = tf.reduce_mean(loss)
        return self.loss, self.weighted_loss

