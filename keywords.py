import os
os.environ['JAVA_HOME'] = r'D:\jdk-19.0.1\bin'

import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

doc1 = """
최근 LED(Light-Emitting Diode) TV 시장의 급성장에 따라 LED BLU(Back Light Unit)를 제어 하는 IC 또한 중요성이 커지고 있으며,
DC-DC 컨버터는 효율이 높은 LLC 공진컨버터로 대체되고, 단가를 낮추기 위헤 PCB 에 실장 되는 개별 소자들로 이루어진 블록을 하나의 IC로 집적하는 연구가 활발하게 진행되고 있다.
"""

okt = Okt()

# doc 핵심 단어
tokenized_doc = okt.pos(doc1)

tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])


# 품사추출
def get_noun(text):
    tokenized_doc = okt.pos(text)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])
    return tokenized_nouns
print(get_noun(doc1))

# 문장 추출
def get_sentence(text):
    tokenized_doc = okt.pos(text)
    tokenized_sentence = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Punctuation'])
    return tokenized_sentence

print(get_sentence(doc1))

# 중요 단어
def get_important_word(text):
    tokenized_doc = okt.pos(text)
    tokenized_important_word = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Verb'])
    return tokenized_important_word
print(get_important_word(doc1))

# 명사 추출
def get_nouns(text):
    tokenized_doc = okt.pos(text)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])
    return tokenized_nouns
print(get_nouns(doc1))

# tensorflow 모델 학습
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# 데이터셋
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

# 데이터셋 전처리
max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


# 데이터셋 만들기
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# 모델 만들기
embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

model.summary()

# 모델 컴파일
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# 모델 훈련
epochs = 10
history = model.fit(
    raw_train_ds,
    validation_data=doc1,
    epochs=epochs)


# 모델 평가
loss, accuracy = model.evaluate(doc1)

print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))

# 모델 예측
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')])
export_model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])
    