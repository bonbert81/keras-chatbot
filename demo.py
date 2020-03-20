import os
import re

import numpy as np
import tensorflow as tf
import yaml
from gensim.models.word2vec import Word2Vec
from tensorflow.keras import models, preprocessing
from tensorflow.keras.layers import Input

from procesar_texto import separar_oracion_tokens

dir_path = "data"
files_list = os.listdir(dir_path + os.sep)

questions = list()
resp = list()
answers = list()
vocab = []
input_characters = set()
target_characters = set()

print("Archivos a entrenar: {}".format(files_list))
for filepath in files_list:
    if filepath.endswith(".yaml"):
        stream = open(dir_path + os.sep + filepath, "rb")
        docs = yaml.safe_load(stream)
        conversations = docs["conversaciones"]
        for con in conversations:
            input_text = con[0]
            if len(con) > 2:
                pregunta = input_text
                pregunta = pregunta.lower()
                # pregunta = re.sub("[^a-zA-Z]", " ", pregunta)
                questions.append(pregunta)
                res = con[1:]

                ans = ""
                for rep in res:
                    r = str(rep).lower()
                    # r = re.sub("[^a-zA-Z]", " ", rep)
                    ans += " " + r
                resp.append(ans)
            elif len(con) > 1:
                questions.append(input_text)
                target_text = con[1]
                resp.append(target_text)

                for char in separar_oracion_tokens(input_text):
                    if char not in input_characters:
                        input_characters.add(char)
                for char in separar_oracion_tokens(target_text):
                    if char not in target_characters:
                        target_characters.add(char)

answers_with_tags = list()
for i in range(len(resp)):
    if type(resp[i]) == str:
        answers_with_tags.append(resp[i])
    else:
        questions.pop(i)

for i in range(len(answers_with_tags)):
    answers.append("<START> " + answers_with_tags[i] + " <END>")

palabras = set()
palabras.add('start')
palabras.add('end')
palabras = palabras.union(target_characters)
palabras = palabras.union(input_characters)
palabras = sorted(list(palabras))
# target_characters = sorted(list(target_characters))
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(palabras)
VOCAB_SIZE = len(tokenizer.word_index) + 1
print("vocab size : {}".format(VOCAB_SIZE))

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in questions])
max_decoder_seq_length = max([len(txt) for txt in resp])

print('Number of samples:', len(input_characters + target_characters))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

for word in tokenizer.word_index:
    if "mediante" in word:
        print("aqui vocab")
    vocab.append(word)


def tokenize(sentences):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append(tokens)

    for i, tok in enumerate(tokens_list):
        if tok == "mediante":
            print("aqui")
    return tokens_list, vocabulary


p = tokenize(questions + answers)
model = Word2Vec(p[0])

embedding_matrix = np.zeros((VOCAB_SIZE, 100))
for i in range(len(tokenizer.word_index)):
    try:
        embedding_matrix[i] = model[vocab[i]]
    except KeyError as identifier:
        pass
        # print("palabra no en vocab: {}".format(identifier))

# input_token_index = dict(
#     [(char, i) for i, char in enumerate(input_characters)])
# target_token_index = dict(
#     [(char, i) for i, char in enumerate(target_characters)])
#
# # Reverse-lookup token index to decode sequences back to
# # something readable.
# reverse_input_char_index = dict(
#     (i, char) for char, i in input_token_index.items())
# reverse_target_char_index = dict(
#     (i, char) for char, i in target_token_index.items())

model = tf.keras.models.load_model('model.h5')
model.summary()

encoder_inputs = model.input[0]
encoder_outputs, state_h, state_c = model.layers[4].output
encoder_states = [state_h, state_c]
encoder_model = models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(200,), name='input_3')
decoder_state_input_c = Input(shape=(200,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

embedding = model.layers[3]
decoder_input = model.input[1]
decoder_embedding = embedding(decoder_input)  # input_2

decoder_lstm = model.layers[5]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)

decoder_states = [state_h_dec, state_c_dec]

decoder_dense = model.layers[6]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = models.Model(
    [decoder_input] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_seq_encoded):
    states_value = encoder_model.predict(input_seq_encoded)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index["start"]

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:

        dec_outputs, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_char = None
        for palabra, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_sentence += " {}".format(palabra)
                sampled_char = palabra

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == "end" or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_word_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


def str_to_tokens(sentence: str):
    tokens_list = list()
    for word in separar_oracion_tokens(sentence):
        tokens_list.append(tokenizer.word_index[word] if word in tokenizer.word_index else 0)
    return preprocessing.sequence.pad_sequences(
        [tokens_list], maxlen=max_encoder_seq_length, padding="post"
    )


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    q = input("Enter question : ")
    input_seq = str_to_tokens(q)
    decoded_sentence = decode_sequence(input_seq)
    print('-' * 5)
    print('Input sentence:', q)
    print('Decoded sentence:', decoded_sentence)
