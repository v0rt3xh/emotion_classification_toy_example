import  tensorflow as tf
import streamlit as st
import keras
import pickle
import numpy as np
from keras import layers


class transformerBlock(layers.Layer):
    def __init__(self, embedding_dimension, num_heads, feedfoward_dimension, dropout=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dimension)
        self.feedfoward = keras.Sequential(
            [layers.Dense(feedfoward_dimension, activation='relu'), layers.Dense(embedding_dimension),]
        )
        self.normLayer1 = layers.LayerNormalization(epsilon=1e-6)
        self.normLayer2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        output1 = self.normLayer1(attention_output + inputs)
        feedforward_output = self.feedfoward(output1)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        return self.normLayer2(feedforward_output + output1)

class tokenPostionEmbedding(layers.Layer):
    def __init__(self, maxLen, vocab_size, embedding_dim):
        super().__init__()
        self.token_embed  = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.pos_embed = layers.Embedding(input_dim=maxLen, output_dim=embedding_dim)

    def call(self, x):
        maxLen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxLen)
        positions = self.pos_embed(positions)
        x = self.token_embed(x)
        return x + positions
    
def init_model():
    tokenizer_from_disk = pickle.load(open("textVec_layer.pkl", "rb"))
    textVec_layer = layers.TextVectorization.from_config(tokenizer_from_disk['config'])
    textVec_layer.set_weights(tokenizer_from_disk['weights'])
    model = keras.models.load_model('transformer_classifier.keras', custom_objects={'tokenPostionEmbedding': tokenPostionEmbedding, 'transformerBlock': transformerBlock})
    return model, textVec_layer

def get_prediction(model, text2Vec, user_input):
    vectorized_input = text2Vec(tf.expand_dims([user_input], -1))
    scores = model.predict(vectorized_input)[0]
    index = np.argmax(scores)
    return index, scores[index]