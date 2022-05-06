import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import pandas as pd

import align
import argparse
import codecs
import os, sys
from random import random, choice
import re
import glob
import matplotlib.pyplot as plt
import time
from losses_impl import *
from keras import backend


from IPython.utils import io
max_stem_length = 10
def read_data(filename):
    with codecs.open(filename, 'r', 'utf-8') as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split('\t')
        if l:
            inputs.append(list(l[0].strip()))
            outputs.append(list(l[1].strip()))
            tags.append(re.split('\W+', l[2].strip()))
    return inputs, outputs, tags

def find_good_range(a,b):
	mask = [(a[i]==b[i] and a[i] != u" ") for i in range(len(a))]
	if sum(mask) == 0:
		# Some times the alignment is off-by-one
		b = [' '] + b
		mask = [(a[i]==b[i] and a[i] != u" ") for i in range(len(a))]
	ranges = []
	prev = False
	for i,k in enumerate(mask):
		if k and prev:
			prev = True
		elif k and not prev:
			start = i
			prev = True
		elif prev and not k:
			end = i
			ranges.append((start, end))
			prev = False
		elif not prev and not k:
			prev = False
	if prev:
		ranges.append((start,i+1))
	ranges = [c for c in ranges if c[1]-c[0]>2]
	return ranges
def generate_stem():
	return "___"

def get_chars(l):
    flat_list = [char for word in l for char in word]
    return list(set(flat_list))
def best_range(ranges):
    longest_length = 0
    longest_index = 0
    for i,r in enumerate(ranges):
        length = r[1] - r[0]
        if length > longest_length:
            longest_length = length
            longest_index = i
    return ranges[i]

def augment(input_path):
    inputs,outputs,tags = np.array(read_data(input_path), dtype=object)
    temp = [(''.join(inputs[i]), ''.join(outputs[i])) for i in range(len(outputs))]
    aligned = align.Aligner(temp).alignedpairs
    vocab = list(get_chars(inputs + outputs))
    try:
        vocab.remove(u" ")
    except:
        pass

    new_inputs = []
    new_outputs = []
    new_tags = []
    for k,item in enumerate(aligned):
        i,o = item[0],item[1]
        good_range = find_good_range(i, o)
        if good_range:
            new_i, new_o = list(i), list(o)
            r = best_range(good_range)
            s = r[0]
            e = r[1]
            if (e-s>5): #arbitrary value
                s += 1
                e -= 1
            new_stem = generate_stem()
            new_i[s:e] = new_stem
            new_o[s:e] = new_stem
            new_i1 = [c for l,c in enumerate(new_i) if (c.strip() or (new_o[l]==' ' and new_i[l] == ' '))]
            new_o1 = [c for l,c in enumerate(new_o) if (c.strip() or (new_i[l]==' ' and new_o[l] == ' '))]
            new_inputs.append(new_i1)
            new_outputs.append(new_o1)
            new_tags.append(tags[k])
        else:
            new_inputs.append([])
            new_outputs.append([])
            new_tags.append([])
    return new_inputs, new_outputs, new_tags

def find_stems(input_path):
    inputs,outputs,tags = np.array(read_data(input_path), dtype=object)
    temp = [(''.join(inputs[i]), ''.join(outputs[i])) for i in range(len(outputs))]

    with io.capture_output() as captured:
        aligned = align.Aligner(temp).alignedpairs

    vocab = list(get_chars(inputs + outputs))
    try:
        vocab.remove(u" ")
    except:
        pass

    stems = []
    for k,item in enumerate(aligned):
        i,o = item[0],item[1]
        good_range = find_good_range(i, o)
        if good_range:
            r = best_range(good_range)
            s = r[0]
            e = r[1]
            if (e-s>5): #arbitrary value
                s += 1
                e -= 1
            stem = o[s:e]
            stems.append(stem)
        else:
            return inputs
    
    return stems

def get_vocab(strings):
    return sorted(list(get_chars(strings)))

def enumerate_sequence_characters(sequences, vocab):
    lut = {"0":0}
    count = 1
    for character in vocab:
        if character != "0":
            lut[character] = count
            count += 1
    new_sequences = []
    for sequence in sequences:
        new_sequences.append([lut[char] for char in sequence])
    return np.array(new_sequences)

def one_hot_encode_sequence(sequences, vocab):
    length = len(vocab)
    lut = {"0":0}
    count = 1
    for character in vocab:
        if character != "0":
            lut[character] = count
            count += 1
    new_sequences = []
    for sequence in sequences:
        new_sequences.append([[0] * lut[char] + [1] + [0] * ((length - lut[char]) - 1) for char in sequence])
    return np.array(new_sequences), {v: k for k, v in lut.items()}

def get_stem_data(language, set_type="train", hilo=None, data_dir="sigmorphon_data", pad=True):
    if hilo is None:
        if f'{data_dir}/{language}-{set_type}-high' in glob.glob(f'{data_dir}/{language}-{set_type}-*'):
            hilo = "high"
        else:
            hilo = "low"
    dpath = f'{data_dir}/{language}-{set_type}-{hilo}'
    padded_stems = sequence.pad_sequences(find_stems(dpath), dtype=str, maxlen=max_stem_length, padding="post", truncating="post")
    if pad:
        return padded_stems
    else:
        return np.array([np.array(stem) for stem in find_stems(dpath)])

def clean_stems(affirmative_stems, negative_stems):
    excluded_chars = [",","'","/","*","-","1","2","3","4","5","6","7","8","9"]
    affirmative_vocab = get_vocab(affirmative_stems)
    to_be_excluded = []
    for i,stem in enumerate(negative_stems):
        for char in stem:
            if char not in affirmative_vocab or char in excluded_chars:
                to_be_excluded.append(i)
                break
    new_negative_stems = []
    new_affirmative_stems = []
    for i,stem in enumerate(negative_stems):
        if i not in to_be_excluded:
            new_negative_stems.append([character.lower() for character in stem])
    for i,stem in enumerate(affirmative_stems):
        reject = False
        for char in stem:
            if char in excluded_chars:
                reject = True
        if not reject:
            new_affirmative_stems.append([character.lower() for character in stem])
    return new_affirmative_stems, new_negative_stems

def create_stem_dataset(reference_language, other_languages):
    reference_stems = get_stem_data(reference_language)
    other_stems = np.concatenate([get_stem_data(language) for language in other_languages])
    cleaned = clean_stems(reference_stems, other_stems)
    reference_stems = cleaned[0]
    other_stems = cleaned[1]
    combined_stems = np.concatenate([reference_stems, other_stems])
    combined_vocab = get_vocab(combined_stems)
    print(combined_vocab)
    X,lut = one_hot_encode_sequence(combined_stems, combined_vocab)
    labels = np.array([1] * len(reference_stems) + [0] * len(other_stems))
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33)
    return X_train, X_test, y_train, y_test, combined_vocab

def create_single_language_stem_dataset(language, hilo="high"):
    reference_stems = get_stem_data(language, set_type="train", hilo=hilo)
    cleaned = clean_stems(reference_stems, np.array([]))
    reference_stems = cleaned[0]
    combined_vocab = get_vocab(reference_stems)
    print(combined_vocab)
    X,lut = one_hot_encode_sequence(reference_stems, combined_vocab)
    labels = np.array([1] * len(reference_stems))
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.001)
    return X_train, X_test, y_train, y_test, combined_vocab, lut

language = "cornish"
X_train, X_test, y_train, y_test, X_vocab, lut = create_single_language_stem_dataset(language, hilo="low")
print(X_train.shape)
vocab_len = len(X_vocab)

def wasserstein_loss(y_true, y_pred):
    y_true_adjusted = tf.math.subtract(tf.math.multiply(y_true, 2), 1)
    y_pred_adjusted = tf.math.subtract(tf.math.multiply(y_pred, 2), 1)
    return backend.mean(y_true_adjusted * y_pred_adjusted)

generator = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.Dense(vocab_len, activation="softmax")
])
discriminator = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences=True),
    # keras.layers.Dropout(0.2),
    # keras.layers.LSTM(100, return_sequences=True),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss=wasserstein_loss, optimizer="adam", metrics="accuracy")
discriminator.trainable = False
gan.compile(loss=wasserstein_loss, optimizer="adam", metrics="accuracy")

batch_size = 1

dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

def distribution_to_sequence(batch_output):
    return tf.one_hot(tf.argmax(batch_output, axis=-1), depth = vocab_len)

def train_gan(gan, dataset, batch_size, vocab_len, n_epochs=50):
    generator, discriminator = gan.layers
    discriminator_history = {"loss":[],"accuracy":[]}
    generator_history = {"generated_sequences":[]}
    gan_history = {"loss":[],"accuracy":[]}
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        discriminator_metrics = []
        gan_metrics = []
        generated_images = None
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size,max_stem_length,vocab_len])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, tf.cast(X_batch, dtype="float32")], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator_metrics = discriminator.train_on_batch(tf.cast(X_fake_and_real, dtype="float32"), y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size,max_stem_length,vocab_len])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan_metrics = gan.train_on_batch(noise, y2)
        for i,v in enumerate(discriminator_metrics):
            discriminator_history[discriminator.metrics_names[i]].append(v)
        for i,v in enumerate(gan_metrics):
            gan_history[gan.metrics_names[i]].append(v)
        generator_history["generated_sequences"].append(generated_images[0])
        history_df = pd.DataFrame({
            "discriminator_loss":discriminator_history["loss"],
            "discriminator_accuracy":discriminator_history["accuracy"],
            "gan_loss":gan_history["loss"],
            "gan_accuracy":gan_history["accuracy"],
            "generated_sequences":["".join([lut[np.argmax(one_hot_char)] for one_hot_char in one_hot_sequence]) for one_hot_sequence in generator_history["generated_sequences"]],
        })
        history_df.to_csv(f'{language}_gan_history.csv')
        if epoch % 50 == 0:
            gan.save(f'checkpoints/{language}-epoch-{epoch}.h5', save_format="h5")
    return {"discriminator": discriminator_history, "generator": generator_history, "gan": gan_history}

history = train_gan(gan, dataset, batch_size, vocab_len, n_epochs=200000)