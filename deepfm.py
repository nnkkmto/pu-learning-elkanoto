from collections import OrderedDict
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, features_info, emb_dim, name_prefix=''):
        """
        sequence対応のembedding layer
        """
        super(EmbeddingLayer, self).__init__()
        self.features_info = features_info
        self.feature_to_embedding_layer = OrderedDict()
        for feature in features_info:
            initializer = tf.keras.initializers.RandomNormal(stddev=0.01, seed=None)
            if feature['is_sequence']:
                # sequenceのembedding
                self.feature_to_embedding_layer[feature['name']] = tf.keras.layers.Embedding(
                    feature['dim'],
                    emb_dim,
                    mask_zero=True,
                    name=f"embedding_{name_prefix}{feature['name']}",
                    embeddings_initializer=initializer)
            else:
                self.feature_to_embedding_layer[feature['name']] = tf.keras.layers.Embedding(
                    feature['dim'],
                    emb_dim,
                    name=f"embedding_{name_prefix}{feature['name']}",
                    embeddings_initializer=initializer)

    def concatenate_embeddings(self, embeddings, name_prefix=''):
        if len(embeddings) >= 2:
            embeddings = tf.keras.layers.Concatenate(axis=1, name=name_prefix+'embeddings_concat')(embeddings)
        else:
            embeddings = embeddings[0]
        return embeddings

    def call(self, inputs):
        embeddings = []
        for feature_input, feature in zip(inputs, self.features_info):
            # embeddingの作成
            embedding = self.feature_to_embedding_layer[feature['name']](feature_input)
            if feature['is_sequence']:
                # sequenceの場合はaverage pooling
                embedding = tf.math.reduce_mean(embedding, axis=1, keepdims=True)
            embeddings.append(embedding)

        # concatenate
        embeddings = self.concatenate_embeddings(embeddings)
        return embeddings


class FmLayer(tf.keras.layers.Layer):
    def __init__(self, features_info):
        super(FmLayer, self).__init__()

        # linear
        self.linear_embedding = EmbeddingLayer(features_info, 1, 'fm_linear_')
        self.linear_dense = tf.keras.layers.Dense(1, activation='relu', name='linear_dense')

    def call(self, inputs, embeddings):
        linear_embeddings = self.linear_embedding(inputs)
        linear_output = tf.squeeze(linear_embeddings, axis=2, name='linear_embeddings_squeeze')
        linear_output = self.linear_dense(linear_output)

        # cross
        sum_then_square = tf.square(tf.reduce_sum(embeddings, axis=1))
        square_then_sum = tf.reduce_sum(tf.square(embeddings), axis=1)
        cross_output = tf.subtract(sum_then_square, square_then_sum)

        # merge linear and cross
        output = tf.keras.layers.Concatenate(axis=1, name='output_concat')([cross_output, linear_output])

        return output


class DeepLayer(tf.keras.layers.Layer):
    def __init__(self, dim, dropout_rate):
        super(DeepLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.dense_1 = tf.keras.layers.Dense(dim, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(dim, activation='relu', name='dense_2')

    def call(self, inputs):
        output = self.dense_1(inputs)
        output = tf.keras.layers.Dropout(self.dropout_rate)(output)
        output = self.dense_2(output)
        return output


class DeepFmLayer(tf.keras.layers.Layer):
    def __init__(self, features_info, dim, dropout_rate, is_add_fm=True):
        super(DeepFmLayer, self).__init__()
        self.is_add_fm = is_add_fm

        self.fm = FmLayer(features_info)
        self.deep = DeepLayer(dim, dropout_rate)
        self.out_dense = tf.keras.layers.Dense(1, activation='sigmoid', name='out_dense')

    def call(self, inputs, embeddings):
        # deep component
        reshaped_embeddings = tf.keras.layers.Flatten(name='deep_embeddings_flatten')(embeddings)
        deep_output = self.deep(reshaped_embeddings)

        if self.is_add_fm:
            # fm component
            fm_output = self.fm(inputs, embeddings)
            # merge fm and deep
            output = tf.keras.layers.Concatenate(axis=-1, name='output_concat')([fm_output, deep_output])
        else:
            output = deep_output

        return self.out_dense(output)


class DeepFM(tf.keras.Model):
    def __init__(self, features_info, emb_dim=10, dropout_rate=0.25, is_add_fm=True):
        super(DeepFM, self).__init__()
        self.emb_dim = emb_dim
        self.embedding = EmbeddingLayer(features_info, emb_dim)
        self.layer = DeepFmLayer(features_info, emb_dim, dropout_rate, is_add_fm)


    def call(self, inputs):
        # embedding
        embeddings = self.embedding(inputs)
        # deep fm
        output = self.layer(inputs, embeddings)

        return output














