import numpy as np
import tensorflow as tf


class Transformer(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 pe_input,
                 dropout=0.1,
                 use_lstm=False
                 ):
        super(Transformer, self).__init__()

        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm1 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(1024, return_sequences=True)
            )
            self.lstm2 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(1024)
            )

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, dropout)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, input, training):
        input, enc_padding_mask = input
        enc_output = self.encoder(input, training, enc_padding_mask)
        hidden = enc_output[:, 0, :]
        if self.use_lstm:
            hidden = self.lstm1(enc_output)
            hidden = self.lstm2(hidden)
        hidden = self.dense1(hidden)
        hidden = self.dropout1(hidden, training=training)
        hidden = self.dense2(hidden)
        hidden = self.dropout2(hidden, training=training)
        final_output = self.final_layer(hidden)

        return final_output

    def return_embedding(self, input):
        input, enc_padding_mask = input
        enc_output = self.encoder(input, False, enc_padding_mask)

        return enc_output


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 max_position_encoding,
                 dropout=0.1
                 ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.pos_encoding = position_encoding(max_position_encoding, self.d_model)

        self.EncoderLayers = [EncoderLayer(d_model, num_heads, dff, dropout)
                              for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.EncoderLayers[i](x, training, mask)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.MultiHeadAttention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_network = feed_forward_network(d_model, dff)

        self.NormLayer1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.NormLayer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        attn_output, weights = self.MultiHeadAttention(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.NormLayer1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.feed_forward_network(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.NormLayer2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


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
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth)

        scaled_attention, weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, weights


def feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask):
    mul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    score = mul_qk / tf.math.sqrt(dk)
    if mask is not None:
        rev_mask = tf.cast(tf.math.equal(mask, 0)[:, tf.newaxis, tf.newaxis, :], tf.float32)
        score += (rev_mask * -1e9)
    weights = tf.nn.softmax(score, axis=-1)
    output = tf.matmul(weights, v)

    return output, weights


def position_encoding(max_position_encoding, d_model):
    angle = get_angle(np.arange(max_position_encoding)[:, np.newaxis],
                      np.arange(d_model)[np.newaxis, :],
                      d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])

    pos_encoding = angle[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angle(pos, i, d_model):
    angle = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle
